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

class TestReLU(TestCase):
    def setUp(self):
        self.input_node = layers.Input((3,1), train=False)
        self.input_node.set(torch.tensor([[7],[-2],[0]],dtype=torch.float64))
        self.relu_node = layers.ReLU(self.input_node)

        #Backprop

        self.input_node2 = layers.Input((3,1), train=False)
        self.input_node2.set(torch.tensor([[-2],[0],[5]],dtype=torch.float64))
        self.relu_node2 = layers.ReLU(self.input_node2)
        self.relu_node2.grad = torch.tensor([[2],[-3],[4]],dtype=torch.float64)
    def test_forward(self):
        self.relu_node.forward()
        np.testing.assert_allclose(self.relu_node.output.cpu().numpy(), np.array([[7],[0],[0]]))
        np.testing.assert_allclose(self.relu_node.size, np.array([[7],[0],[0]]).shape)

    def test_backward(self):
        self.relu_node2.backward()
        np.testing.assert_allclose(self.input_node2.grad.cpu().numpy(), np.array([[0],[-3],[4]]))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
