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
        self.layer.set(torch.tensor([[10],[5]],dtype=torch.float64))

        self.layer2 = layers.Input((2,1), train=True)
        self.layer2.set(torch.tensor([[10],[5]],dtype=torch.float64))
        self.layer2.grad = torch.tensor([[5],[6]],dtype=torch.float64)

        self.layer3 = layers.Input((2,1), train=False)
        self.layer3.set(torch.tensor([[10],[5]],dtype=torch.float64))
        self.layer3.grad = torch.tensor([[11],[22]],dtype=torch.float64)


    def test_forward(self):
        self.layer.forward()
        np.testing.assert_allclose(self.layer.output.cpu().numpy(), np.array([[10],[5]]))
        np.testing.assert_allclose(self.layer.size, np.array([[2],[1]]).shape)

    def test_step(self):
        self.layer2.step(1)
        self.layer3.step(1)

        np.testing.assert_allclose(self.layer2.output.cpu().numpy(), np.array([[5],[-1]]))
        np.testing.assert_allclose(self.layer3.output.cpu().numpy(), np.array([[10],[5]]))



if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
