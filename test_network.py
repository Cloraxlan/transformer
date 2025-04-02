from unittest import TestCase
import layers
import numpy as np
import torch
import unittest
import network
# With this block, we don't need to set device=DEVICE for every tensor.
# But you will still need to avoid accidentally getting int types instead of floating-point types.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.set_default_device(0)
     print("Running on the GPU")
else:
     print("Running on the CPU")

class TestNetwork1(TestCase):
    def setUp(self):
        nn = network.Network()
        self.x = layers.Input((2,1), train=False)
        self.x.set(torch.tensor([[5],[2]],dtype=torch.float32))
        self.b = layers.Input((3,1), train=False)
        self.b.set(torch.tensor([[5],[2],[1]],dtype=torch.float32))
        self.W = layers.Input((3,2), train=False)
        self.W.set(torch.tensor([[10,7],[9,6],[8,5]],dtype=torch.float32))
       
        nn.add(self.x)
        nn.add(self.b)
        nn.add(self.W)
        self.linear1 = layers.Linear(self.x,self.W,self.b)
        nn.add(self.linear1)
        self.relu = layers.ReLU(self.linear1)
        nn.add(self.relu)
       

        self.nn = nn
    def test_forward(self):
        output = self.nn.forward()
        np.testing.assert_allclose(output.cpu().numpy(), np.array([[69],[59],[51]]))

    def test_backward(self):
        #Full network tests not required in lab yet, just testing it runs
        self.nn.forward()
        self.nn.backward()

    def test_step(self):
        #Full network tests not required in lab yet, just testing it runs
        self.nn.forward()
        self.nn.backward()
        self.nn.step(0.1)



if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
