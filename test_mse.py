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

class TestMSE(TestCase):
    def setUp(self):
        self.input_node1 = layers.Input((1,3), train=False)
        self.input_node1.set(torch.tensor([[5,4,2]],dtype=torch.float64))
        self.input_node2 = layers.Input((1,3), train=False)
        self.input_node2.set(torch.tensor([[5,0,1]],dtype=torch.float64))
        self.mse_node = layers.MSELoss(self.input_node1, self.input_node2)

        # Backprop test
        self.input_node1_2 = layers.Input((1,3), train=False)
        self.input_node1_2.set(torch.tensor([[5,-4,0]],dtype=torch.float64))
        self.input_node2_2 = layers.Input((1,3), train=False)
        self.input_node2_2.set(torch.tensor([[5,-2,1]],dtype=torch.float64))
        self.mse_node_2 = layers.MSELoss(self.input_node2_2, self.input_node1_2)
        self.mse_node_2.grad = torch.tensor([1.5],dtype=torch.float64)

    def test_forward(self):
        self.mse_node.forward()
        np.testing.assert_allclose(self.mse_node.output.cpu().numpy(), np.array([17/3]))
        np.testing.assert_allclose(self.mse_node.size, np.array([17/3]).shape)
    def test_backward(self):
        self.mse_node_2.backward()
        np.testing.assert_allclose(self.input_node1_2.grad.cpu().numpy(), np.array([[0,-2,-1]]))
        np.testing.assert_allclose(self.input_node2_2.grad.cpu().numpy(), np.array([[0,2,1]]))




if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
