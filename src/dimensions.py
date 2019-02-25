import os
import torch

# TASK 1
h1, w1, c1 = 32, 32, 3
kernel1 = 5
padding1 = 2
stride1 = 1


def forward_dimensions(h, w, kernel_size, padding, stride):
    w_next = (w - kernel_size + 2*padding)/stride + 1
    h_next = (h - kernel_size + 2*padding)/stride + 1
    return w_next, h_next


h2, w2 = forward_dimensions(h1, w1, kernel1, padding1, stride1)

print("h2 = {}, w2 = {}".format(h2, w2))

h3, w3 = forward_dimensions(h1, w1, kernel1, padding1, stride1)
print("h3 = {}, w3 = {}".format(h3, w3))

