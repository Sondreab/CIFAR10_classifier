import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
from torch import nn
from dataloaders import load_cifar10, mean, std
from utils import to_cuda, compute_loss_and_accuracy

class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_classes=10):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*16*16
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class LeNet(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_classes=10):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in all conv layers

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # The output of feature_extractor will be [batch_size, num_filters*4, 4, 4]
        self.num_output_features = (num_filters*4) * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class ModelOne(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_classes=10):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        kernal_size = 5
        padding_size = 2

        conv_layers = []

        conv_first = nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=kernal_size,
                stride=1,
                padding=padding_size
            )
        
        nn.init.xavier_uniform_(conv_first.weight)
        conv_layers.append(conv_first)
        
        for i in range(1,6):
            conv_layers.append(
                nn.Conv2d(
                in_channels=num_filters*(2**(i-1)),
                out_channels=num_filters*(2**i),
                kernel_size=kernal_size,
                stride=1,
                padding=padding_size
                )
            )
            nn.init.xavier_uniform_(conv_layers[i].weight)

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            
            conv_layers[0],
            nn.ReLU(),
        
            nn.Dropout2d(p=0.1, inplace=False),
            nn.BatchNorm2d(
                num_filters*(2**0), 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),

            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_layers[1],
            nn.ReLU(),
            conv_layers[2],
            nn.ReLU(),

            nn.BatchNorm2d(
                num_filters*(2**2), 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),

            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_layers[3],
            nn.ReLU(),
            conv_layers[4],
            nn.ReLU(),

            nn.BatchNorm2d(
                num_filters*(2**4), 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
         # The output of feature_extractor will be [batch_size, num_filters*4, 4, 4]
        self.num_output_features = (num_filters*(2**4)) * 4 * 4 # 2⁵ 2⁴ 2² 2² = 2^13
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        
        
        linear1 = nn.Linear(self.num_output_features, 2048)
        nn.init.xavier_uniform_(linear1.weight)

        linear2 = nn.Linear(2048, 64)
        nn.init.xavier_uniform_(linear2.weight)

        linear3 = nn.Linear(64, num_classes)
        nn.init.xavier_uniform_(linear3.weight)

        self.classifier = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2,
            nn.ReLU(),

            nn.BatchNorm1d(
                64, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),

            linear3            
        )

        #nn.init.xavier_uniform(self.feature_extractor.weight)
        #torch.nn.init.xavier_uniform(self)
        #self.feature_extractor.apply(nn.init.xavier_uniform)
        #print(self)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class ModelTwo(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_classes=10):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        kernal_size = 5
        padding_size = 2

        conv1 = nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=kernal_size,
                stride=1,
                padding=padding_size
            )
        
        conv2 = nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=kernal_size,
                stride=1,
                padding=padding_size
            )

        conv3 = nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=kernal_size,
                stride=1,
                padding=padding_size
            )

        nn.init.xavier_uniform_(conv1.weight)
        nn.init.xavier_uniform_(conv2.weight)
        nn.init.xavier_uniform_(conv3.weight)

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(

            conv1,

            torch.nn.Dropout2d(p=0.1, inplace=False),

            torch.nn.BatchNorm2d(
                num_filters, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            conv2,

            torch.nn.BatchNorm2d(
                num_filters*2, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            conv3,

            torch.nn.BatchNorm2d(
                num_filters*4, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
         # The output of feature_extractor will be [batch_size, num_filters*4, 4, 4]
        self.num_output_features = (num_filters*4) * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            torch.nn.BatchNorm1d(
                64, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        #nn.init.xavier_uniform(self.feature_extractor.weight)
        #torch.nn.init.xavier_uniform(self)
        #self.feature_extractor.apply(nn.init.xavier_uniform)
        #print(self)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class ModelStride(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_classes=10):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32 # Set number of filters in all conv layers
        

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            torch.nn.BatchNorm2d(
                image_channels, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=2
            ),
            torch.nn.Dropout2d(p=0.2, inplace=False),
            
            torch.nn.BatchNorm2d(
                num_filters, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=5,
                stride=2,
                padding=1
            ),

            torch.nn.BatchNorm2d(
                num_filters*2, 
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            
        )
        # The output of feature_extractor will be [batch_size, num_filters*4, 4, 4]
        self.num_output_features = (num_filters*4) * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class ResNet_Transfer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18( pretrained = True )
        self.model.fc = nn.Linear( 512 *4 , 10 ) # No need to apply softmax ,
                                            # as this is done in nn. C r o s s E n t r o p y L o s s

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully - connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 c on vo lu ti on al
            param.requires_grad = True # layers

    def forward( self , x):
        x = nn.functional.interpolate(x , scale_factor =8)
        x = self.model(x)
        return x
