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
from torch.nn import CrossEntropyLoss


class LoopedGPT2ModelLMHead(GPT2LMHeadModel):
    def __init__(self, config: LoopedGPT2Config):
        super().__init__(config)
        self.config = config
        self.stopping_criteria = config.stopping_criteria
        self.loss_fct = CrossEntropyLoss()
        # Positional Embeddings
        if config.positional_embed_type == "NoPE":
            # Set positional embeddings to always be zero.
            with torch.no_grad():  # Do this to avoid error `a leaf Variable that requires grad is being used in an in-place operation`
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
        if not self.max_iterations or self.max_iterations < 1:
            raise ValueError("`num_iterations` must be at least 1.")

        # TODO: Add other positional embeddings
        # TODO: ADD only apply positional embeddings on the first pass?

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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        n_loops: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # we will pass inputs_embeds as the last output after the first iteration
        # for NoPE embed, we will set positonal embed to zeros

        lm_logits = None
        hidden_states = None
        assert self.max_iterations is not None

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            # Create mask for the case where some, but not all, batch items have met the confidence threshold
            if self.stopping_criteria == "confidence":
                confidence_mask = torch.zeros(
                    (batch_size, seq_length), dtype=torch.bool, device=input_ids.device
                )
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            # Create mask for the case where some, but not all, batch items have met the confidence threshold
            if self.stopping_criteria == "confidence":
                confidence_mask = torch.zeros(
                    (batch_size, seq_length),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
        else:
            raise ValueError("You must provide either `input_ids` or `inputs_embeds`.")

        # If we are using max_iterations and the user passes in `n_loops` we will always loop for `n_loops` instead
        if n_loops is not None:
            max_iterations = int(n_loops.item())
        else:
            max_iterations = self.max_iterations

        # First pass, we must also embed the input_ids
        for iteration in range(max_iterations):
            transformer_outputs = self.transformer.forward(
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
            hidden_states = transformer_outputs[0]

            if self.stopping_criteria == "confidence":
                # Set device for model parallelism
                if self.model_parallel:
                    torch.cuda.set_device(self.transformer.first_device)
                    hidden_states = hidden_states.to(self.lm_head.weight.device)

                # Pass through LM head
                lm_logits = self.lm_head(hidden_states)

                # Calculate probabilities and confidence scores
                probabilities = torch.softmax(lm_logits, dim=-1)
                confidence_scores, predicted_tokens = probabilities.max(dim=-1)

                # Update the confidence mask for tokens meeting the threshold
                new_confidence_mask = confidence_scores > self.confidence_threshold
                confidence_mask = confidence_mask | new_confidence_mask

                # If all tokens meet the threshold, stop iterating
                if confidence_mask.all():
                    break

                # Update attention mask to "freeze" confident tokens
                if attention_mask is None:
                    attention_mask = ~confidence_mask
                else:
                    attention_mask = attention_mask.masked_fill(confidence_mask, 0.0)

        # If we haven't computed them, which is true when we are not using confidence threshold
        if lm_logits is None:
            assert hidden_states is not None
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)
            # Pass through LM head
            lm_logits = self.lm_head(hidden_states)

        assert lm_logits is not None
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
