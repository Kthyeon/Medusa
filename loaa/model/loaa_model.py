import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
import math
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .loaa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

ADDRESS = {
    'lmsys/vicuna-7b-v1.3': '/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6'
}

HIDDEN_SIZE = {
    'lmsys/vicuna-7b-v1.3': 4096
}

VOCAB_SIZE = {
    'lmsys/vicuna-7b-v1.3': 32000
}

class SwiGLU(nn.Module):
    def forward(self, x):
        """
        Swish-Gated Linear Unit (SwiGLU) activation function.
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class LoaaConfig(PretrainedConfig):
    """
    Look-Ahead Adapter (Loaa) Configuration class.
    Configuration class for Loaa model.

    Args:
        loaa_num_heads (int, optional): Number of heads for the Loaa layer. Default is 2.
        loaa_num_layers (int, optional): Number of Loaa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        loaa_num_heads=9,
        loaa_num_layers=1,
        loaa_width=4,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        shortcut = True,
        cache_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loaa_num_heads = loaa_num_heads
        self.loaa_num_layers = loaa_num_layers
        self.loaa_width = loaa_width
        self.base_model_name_or_path = base_model_name_or_path
        self.address = ADDRESS[self.base_model_name_or_path]
        self.hidden_size = HIDDEN_SIZE[self.base_model_name_or_path]
        self.vocab_size = VOCAB_SIZE[self.base_model_name_or_path]
        self.shortcut = shortcut

class GroupedLinear(nn.Module):
    def __init__(self, input_features, output_features, groups, position=True):
        super(GroupedLinear, self).__init__()
        self.groups = groups
        # assert input_features % groups == 0, "input_features must be divisible by groups"
        # assert output_features % groups == 0, "output_features must be divisible by groups"

        self.input_features = input_features
        self.output_features = output_features

        self.weights = nn.Parameter(torch.Tensor(groups,  self.input_features, self.output_features,))
        self.position = position
        if position:
            # self.positional_embedding = self.get_sinusoidal_encoding()
            self.positional_embedding = nn.Parameter(torch.Tensor(self.input_features))
            nn.init.zeros_(self.positional_embedding)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)

    def get_sinusoidal_encoding(self):
        """
        Generates sinusoidal positional encodings for a given sequence length.

        Args:
            seq_length (int): The length of the sequence.

        Returns:
            torch.Tensor: Sinusoidal positional encodings with shape [seq_length, input_features].
        """
        seq_length = self.groups
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_features, 2) * -(math.log(10000.0) / self.input_features))
        pe = torch.zeros(seq_length, self.input_features)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        if len(x.shape) == 3:
            x = x.unsqueeze(2).repeat(1,1,self.groups,1) # (batch_size, seq_len, groups, input_features)
        elif len(x.shape) == 4:
            assert x.shape[2] == self.groups, "Number of groups in input tensor does not match the number of groups in the layer"
        if self.position:
            output = torch.einsum('blgp,gpo->blgo', x + self.positional_embedding.to(x.device), self.weights)
        else:
            output = torch.einsum('blgp,gpo->blgo', x, self.weights)
        return output

class SelfAttention(nn.Module):
    def __init__(self, o_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(o_dim, o_dim)
        self.key = nn.Linear(o_dim, o_dim)
        # self.value = nn.Linear(o_dim, o_dim)
    
    def forward(self, x):
        # x shape: (b, l, g, o)
        b, l, g, o = x.shape
        
        # Reshape x to merge b and l for simplicity in processing
        x = x.view(-1, g, o)  # Shape: (b*l, g, o)
        
        # Compute queries, keys, values
        Q = self.query(x)  # Shape: (b*l, g, o)
        K = self.key(x)    # Shape: (b*l, g, o)
        # V = self.value(x)  # Shape: (b*l, g, o)
        
        # Calculate attention scores
        # Perform batch matrix multiplication bmm(Q, K.transpose(-2, -1)) -> Shape: (b*l, g, g)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (o ** 0.5)  # Scaling by sqrt(dim)
        probs = F.softmax(scores, dim=-1)  # Shape: (b*l, g, g)
        
        # Apply attention to values
        out = torch.bmm(probs, x)  # Shape: (b*l, g, o)
        
        # Reshape to original dimensions
        out = out.view(b, l, g, o)  # Shape: (b, l, g, o)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, o_dim, num_heads=16):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.o_dim = o_dim
        self.d_k = o_dim * 4 // num_heads  # Dimension of each head

        # Ensure the output dimension is divisible by the number of heads
        assert o_dim % num_heads == 0, "Output dimension must be divisible by the number of heads."

        # Linear layers for queries, keys, and values
        self.query = nn.Linear(o_dim, o_dim * 4)
        self.key = nn.Linear(o_dim, o_dim * 4)
        self.value = nn.Linear(o_dim, o_dim * 4)

        # Final linear layer to transform the concatenated outputs
        self.out = nn.Linear(o_dim * 4, o_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, num_groups, o_dim)
        b, l, g, o = x.shape

        # Reshape x to merge batch and seq_length for simplicity in processing
        x = x.view(-1, g, o)  # Shape: (b*l, g, o)

        # Compute queries, keys, values
        Q = self.query(x).view(-1, g, self.num_heads, self.d_k).transpose(1, 2)  # Shape: (b*l, num_heads, g, d_k)
        K = self.key(x).view(-1, g, self.num_heads, self.d_k).transpose(1, 2)    # Shape: (b*l, num_heads, g, d_k)
        V = self.value(x).view(-1, g, self.num_heads, self.d_k).transpose(1, 2)  # Shape: (b*l, num_heads, g, d_k)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # Scaling by sqrt(dim_k)
        probs = F.softmax(scores, dim=-1)  # Shape: (b*l, num_heads, g, g)

        # Apply attention to values
        out = torch.matmul(probs, V)  # Shape: (b*l, num_heads, g, d_k)
        out = out.transpose(1, 2).contiguous().view(-1, g, self.num_heads * self.d_k)  # Concatenate heads

        # Final linear layer
        out = self.out(out)
        out = out.view(b, l, g, o)  # Reshape to original dimensions

        return out

class LoaBlock(nn.Module):
    """
    A Low-Rank Look-Ahead Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, width = 1.0, loaa = 9, shortcut=True):
        super().__init__()
        assert hidden_size % width == 0, "hidden_size must be divisible by width"

        self.loaa = 9
        self.proj = GroupedLinear(hidden_size, int(hidden_size * width), groups=self.loaa, position=True)
        self.layernorm1 = nn.LayerNorm([int(hidden_size * width)], bias=False)
        self.layernorm2 = nn.LayerNorm([int(hidden_size * width)], bias=False)

        self.att = MultiHeadAttention(int(hidden_size * width))

        exp_ratio = 4
        self.mlp = nn.Linear(int(hidden_size * width), int(hidden_size * width) * exp_ratio, bias=False)
        self.mlp2 = nn.Linear(int(hidden_size * width) * exp_ratio, int(hidden_size * width), bias=False)

        self.exp = GroupedLinear(int(hidden_size * width), hidden_size, groups=self.loaa, position=False)
        self.shortcut = shortcut

        # Initialize as an zero mapping when usin4g a shortcut
        if shortcut:
            torch.nn.init.zeros_(self.exp.weights)

        # Use SwiGLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the Low-Rank Look-Ahead Block.

        Args:
            x (torch.Tensor): Input tensor.
            i (int): The index of the positional encoding to add.

        Returns:
            torch.Tensor: Output after activation.
        """
        # Assuming x is of shape [batch_size, seq_length, embedding_size]
        # Add broadcasting to handle batch size and convert pe to the same device as x
        out = self.proj(x)
        out = self.att(self.layernorm1(out)) + out
        out = self.mlp2(self.act(self.mlp(self.layernorm2(out)))) + out

        if self.shortcut:
            return torch.einsum("blgo->gblo", self.exp(out) + x.unsqueeze(2).repeat(1,1,self.loaa,1))
        else:
            return torch.einsum("blgo->gblo", self.exp(out))


class LoaaModel(nn.Module):
    """The Loaa Language Model Head.

    This module creates a series of prediction heads (based on the 'loaa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
        model
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the LoaaModel.
        """
        super().__init__()

        loaa_num_heads = config.loaa_num_heads
        loaa_num_layers = config.loaa_num_layers
        base_model_name_or_path = config.base_model_name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.loaa = loaa_num_heads
        self.loaa_num_layers = loaa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer_address = config.address
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.base_model = model
        # Create a list of Loaa head
        self.loaa_head = LoaBlock(self.hidden_size, config.loaa_width, self.loaa, config.shortcut)

    # Add a link named base_model to self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the loaa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except Exception as e:
            print(f"Error: {e}")
            config = LoaaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.loaa_num_heads = 9
            base_model_config.loaa_num_layers = config.loaa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            loaa_head_path = os.path.join(pretrained_model_name_or_path, "loaa_lm_head.pt")
            if os.path.exists(loaa_head_path):
                filename = loaa_head_path
            else:
                # TODO (@taehyeonk): till the model is not uploaded yet to the hub
                raise ValueError("Loaa model is not uploaded to the hub yet!")
                filename = hf_hub_download(pretrained_model_name_or_path, "loaa_lm_head.pt")
            loaa_head_state_dict = torch.load(filename, map_location=model.device)
            model.loaa_head.load_state_dict(loaa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        # loaa_forward=False,
        **kwargs,
    ):
        """Forward pass of the LoaaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Loaa heads.
            (Optional) Original predictions from the base model's LM head.
        """

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        # (batch_size, seq_len, hidden_size)
        grouped_hiddens = self.loaa_head(outputs[0].clone())
        # sharing LM Heads
        loaa_logits = torch.einsum('gblo,ov->gblv', grouped_hiddens, self.base_model.lm_head.weight.T)

        if output_orig:
            return loaa_logits, outputs, self.base_model.lm_head(outputs[0].clone())
        return loaa_logits

    def get_loaa_choice(self, model_name):
        """
        TODO (@taehyeonk): find the optimal paths for the following benchmarks
        Currently, using the paths found in Loaa settings
        """
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please check whether Loaa choice configuration is used!')
        return mc_sim_7b_63

    def loaa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Loaa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        loaa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Loaa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            loaa_choices (list, optional): A list of integers indicating the number of choices for each Loaa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache loaa buffers (the fixed patterns for tree attention)
        if loaa_choices is None:
            loaa_choices = self.get_loaa_choice(self.base_model_name_or_path)

        if hasattr(self, "loaa_choices") and self.loaa_choices == loaa_choices:
            # Load the cached loaa buffer
            loaa_buffers = self.loaa_buffers
        else:
            # Initialize the loaa buffer
            loaa_buffers = generate_loaa_buffers(
                loaa_choices, device=self.base_model.device
            )
        self.loaa_buffers = loaa_buffers
        self.loaa_choices = loaa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_loaa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        loaa_logits, logits = initialize_loaa(
            input_ids, self, loaa_buffers["loaa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Loaa heads
            candidates, tree_candidates = generate_candidates(
                loaa_logits,
                logits,
                loaa_buffers["tree_indices"],
                loaa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # Use tree attention to verify the candidates and get predictions
            loaa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                loaa_buffers["loaa_position_ids"],
                input_ids,
                loaa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
            input_ids, logits, loaa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                loaa_buffers["retrieve_indices"],
                outputs,
                logits,
                loaa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


class LoaaModelLlama(LoaaModel, KVLlamaForCausalLM):
    pass

class LoaaModelMistral(LoaaModel, KVMistralForCausalLM):
    pass


class LoaaModel_X():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the loaa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # LOAA-v0.1 load
            config = LoaaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return LoaaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return LoaaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
