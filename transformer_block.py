import layers 
import torch
import network
import math
from rms_norm import RMSNorm

class NetworkGraphSegment:
    def get_graph(self):
        pass
    def get_out(self):
        pass
   

class TransformerBlock(NetworkGraphSegment):
    def __init__(self, E, d_model, d_k, d_v, d_f_f, h):
        self.E = E
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_f_f = d_f_f
        self.h = h

        # Setting up params
        self.W_1 = layers.Input((d_f_f, d_model), train=True)
        self.W_1.randomize()
        self.W_2 = layers.Input((d_model, d_f_f), train=True)
        self.W_2.randomize()
        self.b_1 = layers.Input((d_f_f, 1), train=True)
        self.b_1.randomize()
        self.b_2 = layers.Input((d_model, 1), train=True)
        self.b_2.randomize()
        self.gamma_1 = layers.Input((1, d_model), train=True)
        self.gamma_1.randomize()
        self.gamma_2 = layers.Input((1, d_model), train=True)
        self.gamma_1.randomize()
        self.pos_encoding = layers.Input(E.size, train = False)
        self.pos_encoding.set(positional_encoding(self.E.size[1], self.d_model))

    def get_graph(self):
        graph = [self.W_1, self.W_2, self.b_1, self.b_2, self.gamma_1, self.gamma_2, self.pos_encoding]
        
        #Positional Encoding
        pos_encoded_E = layers.Sum(self.E, self.pos_encoding)
        graph.append(pos_encoded_E)

        #Masked Multi Head Attention 
        multi_head_attn = MultiHeadAttention(pos_encoded_E, self.d_model, self.d_k, self.d_v, self.d_f_f, self.h)
        graph += multi_head_attn.get_graph()
        multi_head_attn_out = multi_head_attn.get_out()

        #Add and Norm
        add_1 = layers.Sum(multi_head_attn_out, self.E)
        graph.append(add_1)
        add_1_row = layers.Transpose(add_1)
        norm_1_row = RMSNorm(add_1_row, self.gamma_1)
        norm_1 =  layers.Transpose(norm_1_row)
        graph.append(add_1_row)
        graph.append(norm_1_row)
        graph.append(norm_1)

        # Feed foward
        linear_1 = layers.Linear(norm_1, self.W_1, self.b_1)
        relu = layers.ReLU(linear_1)
        linear_2 = layers.Linear(relu, self.W_2, self.b_2)
        graph.append(linear_1)
        graph.append(relu)
        graph.append(linear_2)
        
        #Add and norm
        add_2 =  layers.Sum(linear_2, norm_1)
        graph.append(add_2)
        add_2_row = layers.Transpose(add_2)
        norm_2_row = RMSNorm(add_2_row, self.gamma_2)
        self.norm_2 =  layers.Transpose(norm_2_row)
        graph.append(add_2_row)
        graph.append(norm_2_row)
        graph.append(self.norm_2)

        return graph

    def get_out(self):
        return self.norm_2
    

class MultiHeadAttention(NetworkGraphSegment):    
    def __init__(self, E, d_model, d_k, d_v, d_f_f, h):
        self.E = E
        self.attention_scaling = 1/(math.sqrt(d_k))
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_f_f = d_f_f
        self.h = h
        self.b = layers.Input((self.d_model, 1), train=True)
        self.b.randomize()
        # Setting up params
        self.W_O = layers.Input((d_model, h*d_v), train=True)
        self.W_O.randomize()

    def get_graph(self):
        graph = [self.W_O, self.E, self.b]
        
        #Scaled Dot Product Attention
        sdpa_out = []
        #Paralel Head Setup
        for i in range(self.h):
            sdpa = ScaledDotProductAttention(self.E, self.d_k, self.d_v, self.d_model, self.attention_scaling)
            graph += sdpa.get_graph()
            sdpa_out.append(sdpa.get_out())
        
        #Concat
        concat = layers.Concat(sdpa_out[0], sdpa_out[1])
        graph.append(concat)
        for sdpa in sdpa_out[2:]:
            concat = layers.Concat(concat, sdpa)
            graph.append(concat)
        
        #Linear
        self.O = layers.Linear(concat, self.W_O, self.b, debug_name="O")
        graph.append(self.O)
        
        return graph
    def get_out(self):
        return self.O
        

 
        
    
class ScaledDotProductAttention(NetworkGraphSegment):
    def __init__(self, E, d_k, d_v, d_model, attention_scaling):
        # Setting up params

        self.W_Q = layers.Input((d_k, d_model), train=True)
        self.W_K = layers.Input((d_k, d_model), train=True)
        self.W_V = layers.Input((d_v, d_model), train=True)


        self.W_Q.randomize()
        self.W_K.randomize()
        self.W_V.randomize()

        self.Q_b = layers.Input((d_k, 1), train=False)
        self.K_b = layers.Input((d_k, 1), train=False)
        self.V_b = layers.Input((d_k, 1), train=False)
        self.Q_b.randomize()
        self.K_b.randomize()
        self.V_b.randomize()
        self.Q = layers.Linear(E, self.W_Q, self.Q_b,  debug_name="Q")
        self.K = layers.Linear(E, self.W_K, self.K_b,  debug_name="K")
        self.V = layers.Linear(E, self.W_V, self.V_b,  debug_name="V")
        self.b_1 = layers.Input((self.Q.size[0], 1), train=False)
        self.b_2 = layers.Input((self.V.size[0], 1), train=False)
        self.b_1.randomize()
        self.b_2.randomize()


        
        
        self.K_T = layers.Transpose(self.K)
        self.mat_mul_1 = layers.Linear(self.K_T, self.Q, self.b_1, debug_name="mat_mul_1")
        self.scale = layers.Scale(self.mat_mul_1, attention_scaling)
        self.mask = layers.Mask(self.scale)
        self.softmax = layers.SoftmaxWithoutCrossEnt(self.scale)
        self.mat_mul_2 = layers.Linear(self.V, self.softmax, self.b_2, debug_name="mat_mul_2")
    def get_graph(self):
        return [self.Q_b, self.K_b, self.V_b,  self.W_Q, self.W_K, self.W_V, self.Q, self.K, self.V, self.b_1, self.b_2, self.K_T, self.mat_mul_1, self.scale, self.mask, self.softmax, self.mat_mul_2]
    def get_out(self):
        return self.mat_mul_2

def positional_encoding(num_tokens, d_model):
    encoding = torch.zeros(d_model, num_tokens)
    for pos in range(num_tokens):
        for i in range(0,d_model,2):
            encoding[i , pos] = math.sin(pos/(10000 ** ((2 * i) / d_model)))
            encoding[i + 1, pos] = math.cos(pos/(10000 ** ((2 * i) / d_model)))
    return encoding
