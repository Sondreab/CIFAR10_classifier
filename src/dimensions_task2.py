import os
import torch

num_filters = 32
h1, w1, c1 = 32, 32, 3
kernel_conv = 5
stride_conv = 1
padding_conv = 2


kernel_pool = 2
stride_pool = 2

num_layers = 3


def forward_dimensions(h, w, kernel_size, stride, padding=0):
    w_next = (w - (kernel_size-1) + 2*padding - 1)/stride + 1
    h_next = (h - (kernel_size-1) + 2*padding - 1)/stride + 1
    return w_next, h_next

def calc_dims_and_params(parameters=0):
    h, w, c = h1, w1, c1
    kernel_parameters = ((kernel_conv+padding_conv)**2)*c + 1
    for i in range(num_layers):
        h, w = forward_dimensions(h, w, kernel_conv, stride_conv, padding_conv)
        h, w = forward_dimensions(h, w, kernel_pool, stride_pool)
        #parameters += ((kernel_conv+padding_conv)**2 + 1)*num_filters*2**i
        parameters += (kernel_parameters)*num_filters*2**i
        print("After layer ",i)
        print("{} x {} x {}".format(h, w, num_filters*2**i))
    print("Parameters: {}".format(parameters))


calc_dims_and_params()
