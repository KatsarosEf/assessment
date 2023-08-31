import unittest
import torch
from src.models import AlexNet

class Test_AlexNetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = AlexNet()

    def test_shape(self):
        x = torch.randn(1, 32, 3, 256, 256)
        y = Test_AlexNetModel.model(x)
        self.assertEqual(torch.Size((1, 2)), y.shape)

    def test_probs(self):
        x = torch.randn(1, 32, 3, 256, 256)
        y = Test_AlexNetModel.model.forward(x)
        prob = torch.sigmoid(y).detach().flatten()[-1]
        self.assertTrue(0 <= prob <= 1)