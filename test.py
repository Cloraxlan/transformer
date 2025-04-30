import torch
import layers
import math 
import network

d_model = 512
h = 4
n = 6
d_k = int(d_model / h)
d_v = d_k
attention_scaling = 1/(math.sqrt(d_k))
d_f_f = 4*d_model

W_Q = layers.Input((d_k, d_model), train=True)
W_K = layers.Input((d_k, d_model), train=True)
W_V = layers.Input((d_k, d_model), train=True)

W_Q.randomize()
W_K.randomize()
W_V.randomize()


class ScaledDotProductAttention():
    def __init__(self, Q, K, V, attention_scaling):
        self.Q = Q
        self.K = K
        self.V = V
        self.zeros_1 = layers.Input((K.size[0], 1), train=False)
        self.zeros_1.set(torch.zeros(K.size[0],1))
        self.zeros_2 = layers.Input((V.size[0], 1), train=False)
        self.zeros_2.set(torch.zeros(V.size[0],1))
        self.Q_T = layers.Transpose(Q)
        self.mat_mul_1 = layers.Linear(self.Q_T, K, self.zeros_1)
        self.scale = layers.Scale(self.mat_mul_1, attention_scaling)
        #Mask Layer
        # Todo
        self.softmax = layers.SoftmaxWithoutCrossEnt(self.scale)
        self.mat_mul_2 = layers.Linear(V, self.softmax, self.zeros_2)
    def get_graph(self):
        return [self.zeros_1, self.zeros_2, self.Q_T, self.mat_mul_1, self.scale, self.softmax, self.mat_mul_2]
    def get_out(self):
        return self.mat_mul_2

x = ScaledDotProductAttention(W_Q, W_K, W_V, attention_scaling)

n = network.Network()
n.add(W_Q)
n.add(W_K)
n.add(W_V)

for layer in x.get_graph():
    n.add(layer)
n.forward()