import os
import matplotlib.pyplot as plt
import torch
import CNN_models
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


class Trainer:

    def __init__(self, networkModel):
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
        self.model = networkModel
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
    trainer = Trainer(CNN_models.ModelOne())
    trainer.train()

    for i in range(len(trainer.TRAINING_EPOCH)):
        trainer.TRAINING_EPOCH[i] = trainer.TRAINING_EPOCH[i] / trainer.num_checks_per_epoch
    
    print("Final training accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])

    print("Final training loss:", trainer.TRAIN_LOSS[-trainer.early_stop_count])
    print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])
    print("Final test loss:", trainer.TEST_LOSS[-trainer.early_stop_count])

    os.makedirs("plots", exist_ok=True)
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

    
    
    

