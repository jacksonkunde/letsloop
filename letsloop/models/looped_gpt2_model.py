from letsloop.config import LoopedGPT2Config
from typing import Optional, Tuple, Union

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import torch


class LoopedGPT2ModelLMHead(GPT2LMHeadModel):
    def __init__(self, config: LoopedGPT2Config):
        super().__init__(config)
        if config.positional_embed_type == "NoPE":
            # Set positional embeddings to always be zero.
            self.transformer.wpe.weight.fill_(0.0)
            self.transformer.wpe.weight.requires_grad = False
        # Stopping Crit
        if config.stopping_criteria == "fixed_n":
            self.max_iterations = config.max_iterations
        elif config.stopping_criteria == "confidence":
            self.max_iterations = config.max_iterations
            self.confidence_threshold = config.confidence_threshold
        else:
            raise NotImplementedError(f"Stopping Criteria `{config.stopping_criteria}` is not supported. \
                                      Choose from `fixed_n` and `confidence`.")

        if self.max_iterations < 1:
            raise ValueError("`num_iterations` must be at least 1.")

        # TODO: Add other positional embeddings
        # TODO: ADD only apply positional embeddings on the first pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # we will pass inputs_embeds as the last output after the first iteration
        # for NoPE embed, we will set positonal embed to zeros

        hidden_states = None

        # First pass, we must also embed the input_ids
        for iteration in range(self.max_iterations):
            outputs = super().forward(
                input_ids=input_ids
                if iteration == 0
                else None,  # Input IDs only in the first iteration
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=hidden_states if iteration > 0 else inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # Extract the hidden states
            hidden_states = outputs.last_hidden_state

            if self.stopping_criteria == "confidence":
                # Pass through LM head
                pass

        return 0

        # Now we have hidden states
