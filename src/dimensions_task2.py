import os

num_filters = 32
h1, w1, c1 = 32, 32, 3
kernel_conv = 5
stride_conv = 1
padding_conv = 2

dense = [64, 10]
#print("len(dense) = {}".format(len(dense)))
#print("range(dense) = {}".format(range(len(dense))))


kernel_pool = 2
stride_pool = 2

num_layers = 3


def forward_dimensions(h, w, kernel_size, stride, padding=0):
    w_next = (w - (kernel_size-1) + 2*padding - 1)/stride + 1
    h_next = (h - (kernel_size-1) + 2*padding - 1)/stride + 1
    return w_next, h_next

def calc_dims_and_params(parameters=0):
    h, w, c = h1, w1, c1
    parameters += (((kernel_conv)**2)*c + 1)*num_filters
    print("Parameters in layer: {} \n".format(parameters))
    for i in range(num_layers):
        h, w = forward_dimensions(h, w, kernel_conv, stride_conv, padding_conv)
        h, w = forward_dimensions(h, w, kernel_pool, stride_pool)
        print("### After layer {} ### \n".format(i+1))
        print("{} x {} x {}".format(h, w, num_filters*2**i))
        if i < num_layers - 1:
            layer_parameters = ((kernel_conv**2)*num_filters*(2**i) + 1)*num_filters*(2**(i+1))
            parameters += layer_parameters
            print("Parameters in layer: {} \n".format(layer_parameters))
    #Flatten & dense:
    print("\n### Flatten and dense ###")
    flatten = (h * w * num_filters*(2**(num_layers-1)) + 1)*dense[0]
    parameters += flatten
    print("Parameters in flatten: {}".format(int(flatten)))
    for i in range(len(dense)-1):
        parameters += dense[i]*dense[i+1]
        print("Parameters in denselayer {}: {}".format(i+1, dense[i]*dense[i+1]))
    print("\n\n### Total Parameters: {} ###".format(int(parameters)))

print("\n------------------------------------------\n")

calc_dims_and_params()
