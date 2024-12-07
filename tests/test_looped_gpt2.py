import unittest
from letsloop import LoopedGPT2ModelLMHead, LoopedGPT2Config
import torch


class TestLoopedGPT2(unittest.TestCase):
    def setUp(self):
        # Initialize common configurations and model
        self.config = LoopedGPT2Config(
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
        self.model = LoopedGPT2ModelLMHead(self.config)

    def test_model_initialization(self):
        # Ensure model initializes correctly
        model = LoopedGPT2ModelLMHead(self.config)
        self.assertIsNotNone(model)

    def test_looped_gpt2_forward(self):
        # Test forward pass
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        attention_mask = torch.ones((2, 10))
        labels = input_ids.clone()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        self.assertEqual(outputs.logits.shape, (2, 10, self.config.vocab_size))
        self.assertIsNotNone(outputs.loss)

    def test_looped_gpt2_n_loop_forward(self):
        # Test forward pass with n_loops parameter
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        attention_mask = torch.ones((2, 10))
        labels = input_ids.clone()
        n_loops = torch.tensor([2, 4])

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            n_loops=n_loops,
        )
        self.assertEqual(outputs.logits.shape, (2, 10, self.config.vocab_size))
        self.assertIsNotNone(outputs.loss)
