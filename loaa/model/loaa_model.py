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

class PositionEmbedding(nn.Module):

    def __init__(self, hidden_size):
        super(PositionEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        x[:, -1, :] += self.weight
        return x

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
        self.address = os.path.join(cache_dir, ADDRESS[self.base_model_name_or_path])
        self.hidden_size = HIDDEN_SIZE[self.base_model_name_or_path]
        self.vocab_size = VOCAB_SIZE[self.base_model_name_or_path]
        self.shortcut = shortcut

class LoaBlock(nn.Module):
    """
    A Low-Rank Look-Ahead Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, width = 0.25, loaa = 9, position = 0, shortcut=True):
        super().__init__()
        assert hidden_size % width == 0, "hidden_size must be divisible by width"

        self.proj = nn.Linear(hidden_size, int(hidden_size * width), bias = False)
        # self.pointwise = nn.Linear(hidden_size * width, hidden_size * width, bias = False)
        self.exp = nn.Linear(int(hidden_size * width), hidden_size, bias = False)
        self.loaa = 9
        self.position = position
        self.shortcut = shortcut

        # Initialize as an zero mapping when using a shortcut
        if shortcut:
            torch.nn.init.zeros_(self.exp.weight)

        # Use SwiGLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

        self.embedding_size = hidden_size
        self.pe = self.get_sinusoidal_encoding()
        
    def get_sinusoidal_encoding(self):
        """
        Generates sinusoidal positional encodings for a given sequence length.

        Args:
            seq_length (int): The length of the sequence.

        Returns:
            torch.Tensor: Sinusoidal positional encodings with shape [seq_length, embedding_size].
        """
        seq_length = self.loaa
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2) * -(math.log(10000.0) / self.embedding_size))
        pe = torch.zeros(seq_length, self.embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

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
        # add positional encoding
        x[:, -1, :] += self.pe.unsqueeze(0)[:, self.position, :].to(x.device)
        if self.shortcut:
            return self.exp(self.act(self.proj(x))) + x
        else:
            return self.exp(self.act(self.proj(x)))


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
        # Create a list of Loaa heads
        self.loaa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([LoaBlock(self.hidden_size, config.loaa_width, self.loaa, position, config.shortcut)] * loaa_num_layers),
                )
                for position in range(loaa_num_heads)
            ]
        )
        self.position_embedding = nn.ModuleList(
            [
                nn.Sequential(
                    PositionEmbedding(self.hidden_size),
                )
                for _ in range(loaa_num_heads)
            ]


        )

    # Add a link named base_model to self
    # @property
    # def base_model(self):
    #     return self
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
        # if not loaa_forward:
        #     return super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         past_key_values=past_key_values,
        #         position_ids=position_ids,
        #         **kwargs,
        #     )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )

        # Clone the output hidden states
        # (batch_size, seq_len, hidden_size)
        hidden_states = outputs[0].clone()
        _loaa_hidden = [hidden_states]
        # TODO (@taehyeonk): Consider parallelizing this loop for efficiency
        for i in range(self.loaa):
            # _pos_tmp = self.position_embedding[i](hidden_states.clone())
            _loaa_hidden.append(self.base_model.lm_head(self.loaa_head[i](hidden_states.clone())))

        # sharing LM Heads
        orig, loaa_logits = _loaa_hidden[0], _loaa_hidden[1:]
        if output_orig:
            return torch.stack(loaa_logits, dim=0), outputs, self.base_model.lm_head(orig)
        return torch.stack(loaa_logits, dim=0)


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
