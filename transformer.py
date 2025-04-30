#!/usr/bin/env python
# coding: utf-8

import torch
import math
import json

torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.set_default_device(0)
     print("Running on the GPU")
else:
     print("Running on the CPU")


class TransformerBlock():    
    def __init__(self, d_model, d_k, d_v, d_f_f, h):
        self.attention_scaling = 1/(math.sqrt(d_k))
        self.W_O = torch.randn(h*d_v, d_model, requires_grad=True)
        print(self.W_O.shape)
        #tuples with weights in order Q, K, V for each head
        self.head_weights = []
        for i in range(h):
            W_Q = torch.randn(d_model, d_k, requires_grad=True)
            W_K = torch.randn(d_model, d_k, requires_grad=True)
            W_V = torch.randn(d_model, d_v, requires_grad=True)
            self.head_weights.append((W_Q, W_K, W_V))
        self.rms_norm = torch.nn.RMSNorm(d_model)
        self.W_1 = torch.randn(d_model, d_f_f, requires_grad=True)
        self.W_2 = torch.randn(d_f_f, d_model, requires_grad=True)
        self.b_1 = torch.randn(1, d_f_f, requires_grad=True)
        self.b_2 = torch.randn(1, d_model, requires_grad=True)

        self.rms_norm = torch.nn.RMSNorm(d_model)


    def multi_head_attention(self, E):
        heads = []
        for weights in self.head_weights:
            Q_W = weights[0]
            K_W = weights[1]
            V_W = weights[2]

            Q = E @ Q_W
            K = E @ K_W
            V = E @ V_W
            heads.append(self.attention(Q, K, V))
        print(heads[0].shape)
        print(torch.cat(heads, dim=1).shape)
        return torch.cat(heads, dim=1) @ self.W_O
    def attention(self, Q, K, V):
        y_1 = Q @ K.t()
        
        y_2 = self.attention_scaling * y_1
        
        y_3 = self.attention_mask(y_2)
        
        max_y_3 = torch.max(y_3, 0, keepdim=True)[0]
        exp_softmax = torch.exp(y_3-max_y_3)
        sum_softmax = torch.sum(exp_softmax, 0, keepdim=True)
        y_4 = exp_softmax/sum_softmax
        print(y_4.shape)
        print(V.shape)
        y_5 = y_4 @ V
        print(y_5.shape)
        return y_5
    def feed_foward(self, input_):
        linear_1 = input_ @ self.W_1 + self.b_1
        relu = torch.max(torch.zeros(linear_1.size()), linear_1)
        linear_2 = relu @ self.W_2 + self.b_2
        return linear_2
    def attention_mask(self, input_):
        mask = torch.tril(input_, diagonal=0)
        return mask.masked_fill(mask == 0, float('-inf'))
    def add_and_norm(self, E, transformed_E):
        add = E + transformed_E
        return self.rms_norm(add)
    def foward(self, E):
        transformed_E = self.multi_head_attention(E)
        normed_tran_E = self.add_and_norm(E, transformed_E)
        feed_foward_E = self.feed_foward(normed_tran_E)
        output = self.add_and_norm(normed_tran_E, feed_foward_E)
        return output
    def step(learning_rate):
        #redo with adam
        for weights in self.head_weights:
            Q_W = weights[0]
            K_W = weights[1]
            V_W = weights[2]

            Q_W -= learning_rate * Q_W.grad
            K_W -= learning_rate * K_W.grad
            V_W -= learning_rate * V_W.grad
        self.W_1 -= learning_rate * self.W_1.grad
        self.W_2 -= learning_rate * self.W_2.grad
        self.b_1 -= learning_rate * self.b_1.grad
        self.b_2 -= learning_rate * self.b_2.grad


def positional_encoding(E):
    num_tokens = E.size(0)
    encoding = torch.zeros(num_tokens, d_model)
    for pos in range(num_tokens):
        for i in range(0,d_model,2):
            encoding[pos, i] = math.sin(pos/(10000 ** ((2 * i) / d_model)))
            encoding[pos, i + 1] = math.cos(pos/(10000 ** ((2 * i) / d_model)))
    return encoding


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
attention_scaling = 1/(math.sqrt(d_k))
d_f_f = 4*d_model
num_epochs = 1
input_ = torch.stack(one_hot_encoded[0:5])

for i in range(num_epochs):
    W_E = torch.randn(token_count, d_model, requires_grad=True)
    embedding = input_ @ W_E 

    embedding += positional_encoding(embedding)

    tran_blocks = []
    for i in range(n):
        tran_blocks.append(TransformerBlock(d_model, d_k, d_v, d_f_f, h))

    E_run = embedding
    for tran_block in tran_blocks:
        E_run = tran_block.foward(E_run)

    tran_out = E_run

    final_linear = tran_out @ W_E.T


    max_final = torch.max(final_linear, 0, keepdim=True)[0]
    exp_softmax = torch.exp(final_linear-max_final)
    sum_softmax = torch.sum(exp_softmax, 0, keepdim=True)
    final_output = exp_softmax/sum_softmax
    print(list(word_dict)[torch.argmax(final_output[-1])])



