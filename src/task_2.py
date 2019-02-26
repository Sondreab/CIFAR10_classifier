import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
from task_1 import LeNet

class ModelTwo(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
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

class ModelOne(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
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


class ModelStride(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
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


class Trainer:

    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture,32*16*16 tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.early_stop_count = 4

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = ModelOne(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)
        self.num_checks_per_epoch = 2
        self.validation_check = len(self.dataloader_train) // self.num_checks_per_epoch

        # Tracking variables
        self.TRAINING_EPOCH = []
        self.TRAINING_STEP = []
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self, training_it=0, training_epoch=0):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.TRAINING_EPOCH.append(training_epoch)
        self.TRAINING_STEP.append(training_it)
        self.model.eval()
        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def should_early_stop_custom(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        inc_count = 0
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss > previous_loss:
                inc_count += 1
        if inc_count >= self.early_stop_count-1:
            return True
        return False

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        training_iteration = 0
        plot_idx = 0
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)
                training_iteration += 1
                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # BackpropagationACC
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all thratasets.
                if batch_it % self.validation_check == 0:
                    plot_idx += 1
                    self.validation_epoch(training_iteration, plot_idx)
                    # Check early stopping criteria.
                    if self.should_early_stop_custom():
                        print("Early stopping.")
                        return


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    for i in range(len(trainer.TRAINING_EPOCH)):
        trainer.TRAINING_EPOCH[i] = trainer.TRAINING_EPOCH[i] / trainer.num_checks_per_epoch
    os.makedirs("plots", exist_ok=True)

    print("Final training accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])

    print("Final training loss:", trainer.TRAIN_LOSS[-trainer.early_stop_count])
    print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])
    print("Final test loss:", trainer.TEST_LOSS[-trainer.early_stop_count])

    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.TRAINING_EPOCH, trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAINING_EPOCH, trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TRAINING_EPOCH, trainer.TEST_LOSS, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(os.path.join("plots", "final_loss_task2.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.TRAINING_EPOCH, trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAINING_EPOCH, trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TRAINING_EPOCH, trainer.TEST_ACC, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy_task2.png"))
    plt.show()

    
    
    

