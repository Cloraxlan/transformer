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

class TestRegularization(TestCase):
    def setUp(self):
        self.weight_node = layers.Input((3,2), train=False)
        self.weight_node.set(torch.tensor([[-5,1],[7,4],[0,0]],dtype=torch.float64))
        
        self.reg_node = layers.Regularization(self.weight_node, 0.1)

        #Backprop

        self.weight_node2 = layers.Input((3,2), train=False)
        self.weight_node2.set(torch.tensor([[1,2],[-3,4],[0,5]],dtype=torch.float64))
        
        self.reg_node2 = layers.Regularization(self.weight_node2, 0.1)
        self.reg_node2.grad = 2

    def test_forward(self):
        self.reg_node.forward()
        np.testing.assert_allclose(self.reg_node.output.cpu().numpy(), np.array([4.55]))
        np.testing.assert_allclose(self.reg_node.size, np.array([4.55]).shape)
    def test_backward(self):    
        self.reg_node2.backward()
        np.testing.assert_allclose(self.weight_node2.grad.cpu().numpy(), np.array([[0.4,0.8],[-1.2,1.6],[0,2]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)