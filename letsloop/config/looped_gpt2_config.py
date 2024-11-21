from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from typing import Literal, Optional, Union


class LoopedGPT2Config(GPT2Config):
    """
    Configuration class for a Looped GPT-2 model. Inherits from GPT2Config and adds parameters
    for handling looped processing of embeddings through transformer blocks.

    Attributes:
        positional_embed_type (str): Specifies the type of positional embeddings to use.
            Options: "NoPE" (no positional embeddings), "rotary" (rotary embeddings).
        token_prediction_type (str): Specifies the token prediction type.
            Options:
                - "k-token": Predicts the next `k` tokens, where `k` must be specified.
                - "FAP" (Full Answer Prediction): Predicts the entire answer in one step.
        k (Optional[int]): Number of tokens to predict in "k-token" mode. Must be specified if
            token_prediction_type is "k-token".
        input_injection_type (str): Specifies how additional inputs are injected.
            Options:
                - "None": No input injection.
                - "addition": Input is added to the embeddings.
                - "cross_attn": Input is incorporated using cross-attention.
        stopping_criteria (str): Criteria for stopping the loop.
            Options:
                - "fixed_k": Loop runs for a fixed number of iterations.
                - "confidence": Loop stops based on a confidence threshold or max iterations.
        max_iterations (Optional[int]): Maximum number of iterations for the loop. Required if
            stopping_criteria is "confidence".
        confidence_threshold (Optional[float]): Confidence threshold for stopping. Required if
            stopping_criteria is "confidence".
    """

    positional_embed_type: Literal["NoPE", "rotary"] = "rotary"
    token_prediction_type: Literal["k-token", "FAP"] = "FAP"
    k: Optional[int] = None
    input_injection_type: Literal["None", "addition", "cross_attn"] = "None"
    stopping_criteria: Literal["fixed_n", "confidence"] = "fixed_n"
    max_iterations: Optional[int] = None
    confidence_threshold: Optional[float] = None

    def __init__(
        self,
        positional_embed_type: Literal["NoPE", "rotary"] = "rotary",
        token_prediction_type: Literal["k-token", "FAP"] = "FAP",
        k: Optional[int] = None,
        input_injection_type: Literal["None", "addition", "cross_attn"] = "None",
        stopping_criteria: Literal["fixed_n", "confidence"] = "fixed_n",
        max_iterations: int = 4,
        confidence_threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input validation
        if token_prediction_type == "k-token" and k is None:
            raise ValueError(
                "When token_prediction_type is 'k-token', 'k' must be specified."
            )

        if stopping_criteria == "confidence":
            if max_iterations is None or confidence_threshold is None:
                raise ValueError(
                    "When stopping_criteria is 'confidence', both 'max_iterations' and 'confidence_threshold' must be specified."
                )

        self.positional_embed_type = positional_embed_type
        self.token_prediction_type = token_prediction_type
        self.k = k
        self.input_injection_type = input_injection_type
        self.stopping_criteria = stopping_criteria
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
