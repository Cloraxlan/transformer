import layers 
import torch
import network
import math


with open("/home/rozpadekk/csc5611/transformer/transformer-from-scratch/bee20script.txt", "r") as file:
    data = file.read()
    
words = set(data.split(" "))
word_dict = {}
unique_words = len(words)
for i, word in enumerate(words):
    one_hot = torch.zeros(unique_words)
    one_hot[i] = 1
    word_dict[word] = one_hot

one_hot_encoded = []
for word in data.split(" "):
    one_hot_encoded.append(word_dict[word])

token_count = len(word_dict)

d_model = 512
h = 4
n = 6
d_k = int(d_model / h)
d_v = d_k
d_f_f = 4*d_model
learning_rate = 0.001
epochs = 1000

class TransformerBlock():    
    def __init__(self, E, d_model, d_k, d_v, d_f_f, h):
        self.E = E
        self.attention_scaling = 1/(math.sqrt(d_k))
        self.h = h
        # Setup input layers for weight of one transformer block
        self.W_O = layers.Input((d_model, h*d_v), train=True)
        self.W_O.randomize()
        #tuples with weights in order Q, K, V for each head
        self.head_weights = []
        self.zero = layers.Input((d_k, 1), train=False)

        for i in range(h):
            W_Q = layers.Input((d_k, d_model), train=True)
            W_K = layers.Input((d_k, d_model), train=True)
            W_V = layers.Input((d_v, d_model), train=True)


            W_Q.randomize()
            W_K.randomize()
            W_V.randomize()

            Q = layers.Linear(E, W_Q, self.zero,  debug_name="Q")
            K = layers.Linear(E, W_K, self.zero,  debug_name="K")
            V = layers.Linear(E, W_V, self.zero,  debug_name="V")

            self.head_weights.append((W_Q, W_K, W_V, Q, K , V))
        
        #Replace with Yoder implementation when given
        self.rms_norm = torch.nn.RMSNorm(d_model)

        self.W_1 = layers.Input((d_f_f, d_model), train=True)
        self.W_1.randomize()
        self.W_2 = layers.Input((d_model, d_f_f), train=True)
        self.W_2.randomize()
        self.b_1 = layers.Input((d_f_f, 1), train=True)
        self.b_1.randomize()
        self.b_2 = layers.Input((d_model, 1), train=True)
        self.b_2.randomize()

    def get_graph(self):
        graph = [self.zero, self.W_O]
        sdpa_out = []
        for weights in self.head_weights:
            for weight in weights:
                graph.append(weight)
            Q = weights[3]
            K = weights[4]
            V = weights[5]
            sdpa = ScaledDotProductAttention(Q, K, V, self.attention_scaling)
            graph += sdpa.get_graph()
            sdpa_out.append(sdpa.get_out())
        concat = layers.Concat(sdpa_out[0], sdpa_out[1])
        graph.append(concat)
        for sdpa in sdpa_out[2:]:
            concat = layers.Concat(concat, sdpa)
            graph.append(concat)
        zero_O = layers.Input((d_model, 1), train=False)
        self.O = layers.Linear(concat, self.W_O, zero_O, debug_name="O")
        graph.append(zero_O)
        graph.append(self.O)
        return graph
    def get_out(self):
        return self.O
        

        
    
class ScaledDotProductAttention():
    def __init__(self, Q, K, V, attention_scaling):
        self.Q = Q
        self.K = K
        self.V = V
        self.zeros_1 = layers.Input((Q.size[0], 1), train=False)
        self.zeros_2 = layers.Input((V.size[0], 1), train=False)
        self.K_T = layers.Transpose(K)
        self.mat_mul_1 = layers.Linear(self.K_T, Q, self.zeros_1, debug_name="mat_mul_1")
        self.scale = layers.Scale(self.mat_mul_1, attention_scaling)
        #Mask Layer
        # Todo
        self.softmax = layers.SoftmaxWithoutCrossEnt(self.scale)
        self.mat_mul_2 = layers.Linear(V, self.softmax, self.zeros_2, debug_name="mat_mul_2")
        print(self.mat_mul_2.size)
    def get_graph(self):
        return [self.zeros_1, self.zeros_2, self.K_T, self.mat_mul_1, self.scale, self.softmax, self.mat_mul_2]
    def get_out(self):
        return self.mat_mul_2

input_ = torch.stack(one_hot_encoded[0:5])
input_ = input_.t()
I = layers.Input(input_.size(), train=False)
I.set(input_)


W_E = layers.Input((d_model, token_count), train=True)
W_E.randomize()
zero = layers.Input((d_model,1), train=False)
E = layers.Linear(I, W_E, zero, debug_name="E")


        
x = TransformerBlock(E, d_model, d_k, d_v, d_f_f, h)
n = network.Network()
n.add(I)
n.add(W_E)
n.add(zero)
n.add(E)
for layer in x.get_graph():
    n.add(layer)
W_E_T = layers.Transpose(W_E)
zero_2 = layers.Input((token_count,1), train=False)
un_E = layers.Linear(x.get_out(), W_E_T, zero_2, debug_name="un_E")
n.add(W_E_T)
n.add(zero_2)
n.add(un_E)
softmax = layers.Softmax(un_E, I)
n.add(softmax)

for _ in range(epochs):
    loss = n.forward()
    print(loss)
    n.backward()
    n.step(learning_rate)