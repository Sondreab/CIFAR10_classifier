import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
import numpy
from torch import nn
from dataloaders import load_cifar10, mean, std
from utils import to_cuda, compute_loss_and_accuracy


def print_image():
    """
    dataloader_train, dataloader_val, dataloader_test = load_cifar10(1)
    image = next(iter(dataloader_test))[0][0]
    
    print(image.size())
    image_tensor = image.view(1,3,32,32)
    image_tensor = nn.functional.interpolate(image_tensor , scale_factor =8)
    print(image_tensor.size()) #[1,256,256,3]
    
    image_np = image_tensor.numpy().transpose(0,2,3,1)
    plt.imshow(image_np[0])
    plt.show()
    """
    image = plt.imread("../docs/test_img.jpg")
    image = to_tensor(image)
    image = normalize(image.data, mean, std)
    image = image.view(1, *image.shape)
    image = nn.functional.interpolate(image, size=(256,256))
    model = torchvision.models.resnet18(pretrained=True)
    first_layer_out = model.conv1(image)

    first_layer_out = first_layer_out[:,0:16,:,:]

    #print(first_layer_out.size())

    to_visualize = first_layer_out.view(first_layer_out.shape[1],
                                        1, *first_layer_out.shape[2:])

    #print(to_visualize.size())

    torchvision.utils.save_image(to_visualize, os.path.join("../docs/plots", "filters_first_layer_test.png"),4)
    print("done")

def print_weights():
    model = torchvision.models.resnet18(pretrained=True)
    weights = model.conv1.weight.data
    weights = nn.functional.interpolate(weights, size=(128,128))

    torchvision.utils.save_image(weights, os.path.join("../docs/plots", "weights.png"))
    print("done")

    


if __name__ == "__main__":
    print_image()
    print_weights()