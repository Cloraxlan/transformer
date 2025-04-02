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

class TestLinear(TestCase):
    def setUp(self):
        self.weight_node = layers.Input((4,3), train=False)
        self.weight_node.set(torch.tensor([[1,2,3],[-5,10,2],[0,1,8],[-3,2,1]],dtype=torch.float64))

        self.bias_node = layers.Input((4,1), train=False)
        self.bias_node.set(torch.tensor([[2],[3],[2],[-5]],dtype=torch.float64))

        self.input_node = layers.Input((3,2), train=False)
        self.input_node.set(torch.tensor([[10, 1],[-2, 1],[4,1 ]],dtype=torch.float64))
        
        self.linear_node = layers.Linear(self.input_node, self.weight_node, self.bias_node)

        #Backpropagation
        self.weight_node2 = layers.Input((3,2), train=False)
        self.weight_node2.set(torch.tensor([[4,-2],[1,-1],[0,-3]],dtype=torch.float64))

        self.bias_node2 = layers.Input((3,1), train=False)
        self.bias_node2.set(torch.tensor([[2],[-3],[5]],dtype=torch.float64))

        self.input_node2 = layers.Input((2,1), train=False)
        self.input_node2.set(torch.tensor([[1],[-5]],dtype=torch.float64))
        
        self.linear_node2 = layers.Linear(self.input_node2, self.weight_node2, self.bias_node2)
        self.linear_node2.grad = torch.tensor([[-2],[-3],[2]], dtype=torch.float64)
        

    def test_forward(self):
        self.linear_node.forward()
        np.testing.assert_allclose(self.linear_node.output.cpu().numpy(), np.array([[20, 8],[-59, 10],[32, 11],[-35, -5]]))
        np.testing.assert_allclose(self.linear_node.size, np.array([[20, 8],[-59, 10],[32, 11],[-35, -5]]).shape)

    def test_backward(self):
        self.linear_node2.backward()
        np.testing.assert_allclose(self.weight_node2.grad.cpu().numpy(), np.array([[-2, 10],[-3, 15],[2,-10]]))
        np.testing.assert_allclose(self.bias_node2.grad.cpu().numpy(), np.array([[-2],[-3],[2]]))
        np.testing.assert_allclose(self.input_node2.grad.cpu().numpy(), np.array([[-11],[1]]))



if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
