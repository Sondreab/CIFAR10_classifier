import os
import torch

# TASK 1
h1, w1, c1 = 32, 32, 3
kernel1 = 4
padding1 = 2
stride1 = 2


def forward_dimensions(h, w, kernel_size, padding, stride):
    w_next = (w - kernel_size + 2*padding)/stride + 1
    h_next = (h - kernel_size + 2*padding)/stride + 1
    return w_next, h_next


h2, w2 = forward_dimensions(h1, w1, kernel1, padding1, stride1)

print("h2 = {}, w2 = {}".format(h2, w2))

h3, w3 = forward_dimensions(17, 17, 5, 1, 2)
print("h3 = {}, w3 = {}".format(h3, w3))

h4, w4 = forward_dimensions(8, 8, 5, 2, 2)
print("h4 = {}, w4 = {}".format(h4, w4))



"""
MED 3x3 kernal 13 epochs
Final training accuracy: 0.9568444444444445
Final validation accuracy: 0.7672
Final test accuracy: 0.7612
Final training loss: 0.13972859129055656
Final validation loss: 0.7748675531224359
Final test loss: 0.8320062149102521

MED 5x5 kernal
Final training accuracy: 0.9669555555555556
Final validation accuracy: 0.755
Final test accuracy: 0.7525
Final training loss: 0.10689857551584613
Final validation loss: 0.8635616091233266
Final test loss: 0.9076087490008895

MED 7x7 kernal 8-9 epochs
Final training accuracy: 0.9391777777777778
Final validation accuracy: 0.7488
Final test accuracy: 0.7472
Final training loss: 0.1914376855976033
Final validation loss: 0.821383334790604
Final test loss: 0.8252700068009128
