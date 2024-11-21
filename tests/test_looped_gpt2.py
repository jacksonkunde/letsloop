import unittest
from letsloop import LoopedGPT2ModelLMHead, LoopedGPT2Config


class TestLoopedGPT2(unittest.TestCase):
    def test_model_initialization(self):
        config = LoopedGPT2Config(hidden_size=768)
        model = LoopedGPT2ModelLMHead(config)
