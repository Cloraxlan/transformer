from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

# With this block, we don't need to set device=DEVICE for every tensor.
# But you will still need to avoid accidentally getting int types instead of floating-point types.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.set_default_device(0)
     print("Running on the GPU")
else:
     print("Running on the CPU")

class TestInput(TestCase):
    def setUp(self):
        self.layer = layers.Input((2,1), train=False)
        self.layer.grad = torch.tensor([[10],[5]],dtype=torch.float64)
        self.layer2 = layers.Input((2,1), train=False)

    def test_clear_grad(self):
        self.layer.clear_grad()
        np.testing.assert_allclose(self.layer.grad.cpu().numpy(), np.array([[0],[0]]))
    def test_accumulate(self):
        self.layer2.accumulate_grad(torch.tensor([[2],[-3]],dtype=torch.float64))
        np.testing.assert_allclose(self.layer2.grad.cpu().numpy(), np.array([[2],[-3]]))
        self.layer2.accumulate_grad(torch.tensor([[-1],[0]],dtype=torch.float64))
        np.testing.assert_allclose(self.layer2.grad.cpu().numpy(), np.array([[1],[-3]]))
        self.layer2.accumulate_grad(torch.tensor([[0],[0]],dtype=torch.float64))
        np.testing.assert_allclose(self.layer2.grad.cpu().numpy(), np.array([[1],[-3]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
