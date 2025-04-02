from unittest import TestCase
import layers
import numpy as np
import torch
import unittest
import math

# With this block, we don't need to set device=DEVICE for every tensor.
# But you will still need to avoid accidentally getting int types instead of floating-point types.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.set_default_device(0)
     print("Running on the GPU")
else:
     print("Running on the CPU")

class TestSoftmax(TestCase):
    def setUp(self):
        self.input_node = layers.Input((3,2), train=False)
        self.input_node.set(torch.tensor([[math.log(4), math.log(5)],[math.log(2), math.log(6)],[math.log(5), math.log(7)]],dtype=torch.float64))
        self.true_node = layers.Input((3,2), train=False)
        self.true_node.set(torch.tensor([[1, 0],[0, 0],[0, 1]],dtype=torch.float64))
        self.softmax_node = layers.Softmax(self.input_node, self.true_node)

        #backprop
        self.input_node2 = layers.Input((3,1), train=False)
        self.input_node2.set(torch.tensor([[math.log(2)],[math.log(3)],[math.log(1)]],dtype=torch.float64))
        self.true_node2 = layers.Input((3,1), train=False)
        self.true_node2.set(torch.tensor([[0],[1],[0]],dtype=torch.float64))
        self.softmax_node2 = layers.Softmax(self.input_node2, self.true_node2)
        self.softmax_node2.grad = 0.5



    def test_forward(self):
        self.softmax_node.forward()
        np.testing.assert_allclose(self.softmax_node.classifications.cpu().numpy(), np.array([[4/11, 5/18],[2/11, 6/18],[5/11, 7/18]]))
        np.testing.assert_allclose(self.softmax_node.classifications.cpu().numpy().shape, np.array([[4/11, 5/18],[2/11, 6/18],[5/11, 7/18]]).shape)

        np.testing.assert_allclose(self.softmax_node.output.cpu().numpy(), np.array([-(math.log(4/11)+math.log(7/18))]))
        np.testing.assert_allclose(self.softmax_node.size, np.array([0.02058]).shape)

    def test_backward(self):
        self.softmax_node2.backward()
        np.testing.assert_allclose(self.input_node2.grad.cpu().numpy(), np.array([[1/6],[-1/4],[1/12]]))
        

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
