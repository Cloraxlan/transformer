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

class TestSum(TestCase):
    """
    Please note: We (the instructors) may have assumed different parameters for my network than you use.
    TODO: Update these tests to work with YOUR definitions of arguments and variables.
    """
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]],dtype=torch.float64))
        self.b = layers.Input((2,1), train=True)
        self.b.set(torch.tensor([[1],[2]],dtype=torch.float64))
        self.sum = layers.Sum(self.a, self.b)

        #Backprop
        self.a2 = layers.Input((3,1), train=True)
        self.a2.set(torch.tensor([[1],[-3],[0]],dtype=torch.float64))
        self.b2 = layers.Input((3,1), train=True)
        self.b2.set(torch.tensor([[2],[4],[5]],dtype=torch.float64))
        self.sum2 = layers.Sum(self.a2, self.b2)
        self.sum2.grad = torch.tensor([[1],[2],[3]],dtype=torch.float64)

    def test_forward(self):
        self.sum.forward()
        np.testing.assert_allclose(self.sum.output.cpu().numpy(), np.array([[4],[7]]))
        np.testing.assert_allclose(self.sum.size, np.array([[4],[7]]).shape)

    def test_backward(self):
        self.sum2.backward()
        np.testing.assert_allclose(self.a2.grad.cpu().numpy(), np.array([[1],[2],[3]]))
        np.testing.assert_allclose(self.b2.grad.cpu().numpy(), np.array([[1],[2],[3]]))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
