import os
import matplotlib.pyplot as plt
import torch
import torchvision
import CNN_models
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
from torch import nn
from dataloaders import load_cifar10, mean, std
from utils import to_cuda, compute_loss_and_accuracy

       

class Trainer:

    def __init__(self, networkModel):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture,32*16*16 tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-4

        self.early_stoping = False
        self.early_stop_count = 1
        

        """
        ##### TASK 1 & 2:
        epochs = 100
        batch_size = 64
        learning_rate = 5e-2

        ModelOne should have learning_rate = 1e-3


        ##### TASK 3:
        epochs = 100
        batch_size = 32
        learning_rate = 5e-4

        """

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        #self.model = ResNet_Transfer(image_channels=3, num_classes=10)
        self.model = networkModel
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        """##### TASK 1 USES SGD, NOT ADAM #####"""
        
        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)
        #self.num_checks_per_epoch = 2
        self.validation_check = len(self.dataloader_train) // 2 #self.num_checks_per_epoch

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

    def plot_first_and_last_filter(self):
        image = plt.imread("../docs/test_img.jpg")
        image = to_tensor(image)
        image = normalize(image.data, mean, std)
        image = image.view(1, *image.shape)
        image = nn.functional.interpolate(image, size=(256,256))
        image = to_cuda(image)

        first_layer_out = self.model.model.conv1(image)
        first_layer_out_print = first_layer_out[:,0:16,:,:]
        to_visualize_first = first_layer_out_print.view(first_layer_out_print.shape[1],
                                        1, *first_layer_out_print.shape[2:])

        os.makedirs("../docs/plots", exist_ok=True)
        torchvision.utils.save_image(to_visualize_first, os.path.join("../docs/plots", "filter_activation_first_layer.png"),4)

        model_last_2_removed = nn.Sequential(*list(self.model.model.children())[:-2])
        last_conv_output = model_last_2_removed(image)
        last_conv_output = last_conv_output[:,0:16,:,:]
        last_conv_output = nn.functional.interpolate(last_conv_output, size=(128,128))
        to_visualize_last = last_conv_output.view(last_conv_output.shape[1], 1,
                                                    *last_conv_output.shape[2:])

        torchvision.utils.save_image(to_visualize_last, os.path.join("../docs/plots", "filter_activation_last_layer.png"),4)

    def print_weights(self):
        #model = torchvision.models.resnet18(pretrained=True)
        
        weights = self.model.model.conv1.weight.data
        weights = nn.functional.interpolate(weights, size=(128,128))

        red_weights = weights.clone()
        green_weights = weights.clone()
        blue_weights = weights.clone()
        
        red_weights[:,1,:,:] = torch.zeros([128,128])
        red_weights[:,2,:,:] = torch.zeros([128,128])

        green_weights[:,0,:,:] = torch.zeros([128,128])
        green_weights[:,2,:,:] = torch.zeros([128,128])

        blue_weights[:,0,:,:] = torch.zeros([128,128])
        blue_weights[:,1,:,:] = torch.zeros([128,128])

        print(weights.size())
        print(red_weights.size())
        print(green_weights.size())
        print(blue_weights.size())

        os.makedirs("../docs/plots", exist_ok=True)
        torchvision.utils.save_image(weights, os.path.join("../docs/plots", "weights.png"))
        torchvision.utils.save_image(red_weights, os.path.join("../docs/plots", "weights_red.png"))
        torchvision.utils.save_image(green_weights, os.path.join("../docs/plots", "weights_green.png"))
        torchvision.utils.save_image(blue_weights, os.path.join("../docs/plots", "weights_blue.png"))
        print("done")                                           

    def plot_loss_and_accuracy(self, 
                            filename_loss="loss_plot.png",
                            filename_accuracy="accuracy_plot.png"):

        for i in range(len(self.TRAINING_EPOCH)):
            self.TRAINING_EPOCH[i] = self.TRAINING_EPOCH[i] / 3

        os.makedirs("../docs/plots", exist_ok=True)
        # Save plots and show them
        plt.figure(figsize=(12, 8))
        plt.title("Cross Entropy Loss")
        plt.plot(self.TRAINING_EPOCH, self.VALIDATION_LOSS, label="Validation loss")
        plt.plot(self.TRAINING_EPOCH, self.TRAIN_LOSS, label="Training loss")
        plt.plot(self.TRAINING_EPOCH, self.TEST_LOSS, label="Testing Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.savefig(os.path.join("../docs/plots", filename_loss))
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("Accuracy")
        plt.plot(self.TRAINING_EPOCH, self.VALIDATION_ACC, label="Validation Accuracy")
        plt.plot(self.TRAINING_EPOCH, self.TRAIN_ACC, label="Training Accuracy")
        plt.plot(self.TRAINING_EPOCH, self.TEST_ACC, label="Testing Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join("../docs/plots", filename_accuracy))
        plt.show()
    
    def print_loss_and_accuracy(self):
        print("-------------------------------------")
        print("Final training accuracy:", self.TRAIN_ACC[-self.early_stop_count])
        print("Final validation accuracy:", self.VALIDATION_ACC[-self.early_stop_count])
        print("Final test accuracy:", self.TEST_ACC[-self.early_stop_count])
        print("\n")
        print("Final training loss:", self.TRAIN_LOSS[-self.early_stop_count])
        print("Final validation loss:", self.VALIDATION_LOSS[-self.early_stop_count])
        print("Final test loss:", self.TEST_LOSS[-self.early_stop_count])
        print("-------------------------------------\n")

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        training_iteration = 0
        plot_idx = 0
        # Track initial loss/accuracy
        print("Initial:")
        self.validation_epoch()
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch+1))
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
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
                    
                    if self.early_stoping and self.should_early_stop_custom():
                        print("Early stopping.")
                        return
                    



if __name__ == "__main__":

    ourBestModel = Trainer(CNN_models.ModelOne())
    ourBestModel.learning_rate = 1e-3
    ourBestModel.batch_size = 64
    print("## Training ModelOne ##\n")
    ourBestModel.train()
    print("## Done ##\n")


    resNet = Trainer(CNN_models.ResNet_Transfer())
    print("## Training ResNet ##\n")
    resNet.train()
    print("## Done ##\n")
    

    print("### RESNET RESULTS ###\n")
    resNet.print_loss_and_accuracy()
    resNet.plot_loss_and_accuracy("Loss_ResNet.png", "Accuracy_ResNet.png")
    resNet.plot_first_and_last_filter()

    os.makedirs("../docs/plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(resNet.TRAINING_EPOCH, resNet.VALIDATION_LOSS, label="Val loss ResNet")
    plt.plot(resNet.TRAINING_EPOCH, resNet.TRAIN_LOSS, label="Train loss ResNet")
    plt.plot(ourBestModel.TRAINING_EPOCH, ourBestModel.VALIDATION_LOSS, label="Val loss ModelOne")
    plt.plot(ourBestModel.TRAINING_EPOCH, ourBestModel.TRAIN_LOSS, label="Train loss ModelOne")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(os.path.join("../docs/plots", "Loss_ResNet_vs_ModelOne.png"))
    plt.show()

    
   

