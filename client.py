import layers 
import torch
import network
import math
from transformer_block import *
import string
"""
with open("/home/rozpadekk/csc5611/transformer/transformer-from-scratch/bee5script.txt", "r") as file:
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
"""

#with open("/home/rozpadekk/csc5611/transformer/transformer-from-scratch/bee20script.txt", "r") as file:
#    data = file.read()

#data = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
#data = "abcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcde"
data = "abcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij"

letters = list(set(data))
token_count = len(letters)
token_dict = {}

for i, letter in enumerate(letters):
    one_hot = torch.zeros(token_count)
    one_hot[i] = 1
    token_dict[letter] = one_hot


one_hot_encoded = []
samples = [data[i:i+5] for i in range(0, len(data), 5)]

samples_one_hot = []

for sentence in samples:
    one_hot = []
    for letter in sentence:
        one_hot.append(token_dict[letter])
    samples_one_hot.append(one_hot)

shift_right = [data[i-1:i+4] for i in range(1, len(data), 5)]
samples_one_hot_right = []

for sentence in shift_right:
    one_hot = []
    for letter in sentence:
        one_hot.append(token_dict[letter])
    samples_one_hot_right.append(one_hot)
    
d_model = 16
h = 4
n = 3
d_k = int(d_model / h)
d_v = d_k
d_f_f = 4*d_model
learning_rate = 0.01
epochs = 10


I = layers.Input((token_count, 5), train=False)



W_E = layers.Input((d_model, token_count), train=True)
W_E.randomize()
zero = layers.Input((d_model,1), train=False)
E = layers.Linear(I, W_E, zero, debug_name="E")


        
net = network.Network()
net.add(I)
net.add(W_E)
net.add(zero)
net.add(E)
tran_input = E
for i in range(n):
    block = TransformerBlock(tran_input, d_model, d_k, d_v, d_f_f, h)
    for layer in block.get_graph():
        net.add(layer)
    tran_input = block.get_out()
W_E_T = layers.Transpose(W_E)
zero_2 = layers.Input((token_count,1), train=False)
un_E = layers.Linear(block.get_out(), W_E_T, zero_2, debug_name="un_E")
net.add(W_E_T)
net.add(zero_2)
net.add(un_E)
y = layers.Input(I.size, train=False)
net.add(y)
softmax = layers.Softmax(un_E, y)
net.add(softmax)


for _ in range(epochs):
    for x_val, y_val in zip(samples_one_hot_right, samples_one_hot):
        y.set(torch.stack(y_val).t())
        I.set(torch.stack(x_val).t())
        loss = net.forward()
        print(loss)
        net.backward()
        net.step(learning_rate)

prompt = []
sentence = "abcde"
#one hot encode prompt
for letter in sentence:
    prompt.append(token_dict[letter])
    
input_ = torch.stack(prompt)
input_ = input_.t()
I.set(input_)
#convert to columns
net.forward()
final_output = softmax.classifications
print(final_output[:, -1])
next_token = list(token_dict)[torch.argmax(final_output[:, -1])]
print(next_token)
