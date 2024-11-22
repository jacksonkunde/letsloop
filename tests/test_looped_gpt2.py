import unittest
from letsloop import LoopedGPT2ModelLMHead, LoopedGPT2Config

import torch

class TestLoopedGPT2(unittest.TestCase):
    def test_model_initialization(self):
        config = LoopedGPT2Config(hidden_size=768)
        model = LoopedGPT2ModelLMHead(config)

    def test_looped_gpt2_forward(self):
        config = LoopedGPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            positional_embed_type="NoPE",
            stopping_criteria="confidence",
            confidence_threshold=0.9,
            max_iterations=5,
        )
        model = LoopedGPT2ModelLMHead(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        attention_mask = torch.ones((2, 10))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        assert outputs.logits.shape == (2, 10, config.vocab_size)
        assert outputs.loss is not None