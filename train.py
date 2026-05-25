import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms

#Neural Block class, the building blocks that make up the main Neural Network
class NeuralBlock(nn.Module):

    def __init__(self, inputs, outputs, dropout_rate=0.1):
        super().__init__()

        #Layers defined here in __init__ to be used in the forward()
        self.convolution1 = nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(outputs)
        self.activation1 = nn.ReLU(inplace=True)

        self.convolution2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(outputs)

        #Adjust shortcut channels if input/output sizes differ (skip for the Residual part of the block)
        if inputs != outputs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outputs)
            )
        else:
            self.shortcut = nn.Identity()
        self.activation2 = nn.ReLU(inplace=True)

        #Dropout layer to help prevent overfitting
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):

        shortcut = self.shortcut(x)

        #Convolutional layers for the block
        output = self.convolution1(x)
        output = self.batchNorm1(output)
        output = self.activation1(output)

        output = self.convolution2(output)
        output = self.batchNorm2(output)

        #Residual addition (allows input to skip through layers)
        output += shortcut

        output = self.activation2(output)

        output = self.dropout(output)

        return output

#Neural network class, what the neural blocks come together to form and what the data will run through
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        #Convolutional layers for the network added through the neural blocks (8 neural blocks in total, split into groups of 2)
        self.neuralBlocks = nn.Sequential(
            #Group 1: 3 inputs, 32 outputs
            NeuralBlock(3, 32, dropout_rate=0.0),
            NeuralBlock(32, 32, dropout_rate=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Group 2: 32 inputs, 64 outputs
            NeuralBlock(32, 64, dropout_rate=0.0),
            NeuralBlock(64, 64, dropout_rate=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Group 3: 64 inputs, 128 outputs
            NeuralBlock(64, 128, dropout_rate=0.05),
            NeuralBlock(128, 128, dropout_rate=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Group 4: 128 inputs, 256 outputs
            NeuralBlock(128, 256, dropout_rate=0.1),
            NeuralBlock(256, 256, dropout_rate=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #Reduces each feature map down to 1 value, reducing parameters for the classifer
        self.globalPool = nn.AdaptiveAvgPool2d((1,1))

        #Classifer layers to to reduce input of 256 down to one of 37 categories
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output = self.neuralBlocks(x)

        output = self.globalPool(output)

        output = torch.flatten(output, 1)

        output = self.classifier(output)

        return output
    
#Only runs this code if this is the file being run, to prevent accidental training
if __name__ == "__main__":

    #Sets device to be used for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Hyperparameters
    initial_lr = 0.001
    batch_size = 64
    epochs = 30

    best_accuracy = 0.0

    #Data transforms for both training and validation
    transform_training = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomCrop((144, 144)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_validation = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Loading of datasets for training and validation using the corresponding transforms
    full_training_dataset = datasets.OxfordIIITPet(root="data", split="trainval", transform=transform_training, download=True)
    full_validation_dataset = datasets.OxfordIIITPet(root="data", split="trainval", transform=transform_validation, download=True)

    #Calculates amount of images to split from the trainval dataset to create a training and validation dataset (80/20 split)
    training_size = int(0.8 * len(full_training_dataset))

    #Shuffles indices to prevent any possible accidental bias when splitting datasets, torch.manual allows for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(len(full_training_dataset)).tolist()
    #Splits the shuffled indices based on the training and validation dataset sizes
    training_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    #Creates subsets of the full datasets based on the split indices
    training_dataset = Subset(full_training_dataset, training_indices)
    validation_dataset = Subset(full_validation_dataset, validation_indices)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    #Initialises Neural Network model, and moves it to the device for training
    model = NeuralNetwork()
    model = model.to(device=device)

    #Defines loss function, optimiser, and scheduler to be used during training
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=30)


    #Training function to be used during the training loop
    def training_loop(model, training_loader, loss_function, optimiser, device):
        
        running_loss = 0.0
        correct = 0
        total = 0

        #Switches model to training mode
        model.train()

        for images, labels in training_loader:
            #Moves data to device for training
            images = images.to(device)
            labels = labels.to(device)

            #Resets gradients for current epoch
            optimiser.zero_grad()

            #Forward pass (runs prediction) and computes loss (measures error of prediction)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            #Backwards pass (back propogation) and steps through optimiser, updating model weights with gradients
            loss.backward()
            optimiser.step()

            #Adds the loss of the current batch to total for the epoch
            running_loss += loss.item()

            #Computes predictions (highest scoring category), tracks total images processed, and how many predictions match labels
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()

        #Computes and returns loss and accuracy for the epoch
        epoch_loss = running_loss / len(training_loader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    #Validation function (training function but without changing weights, used to compute loss and accuracy without dropout)
    def validation(model, validation_loader, loss_function, device):

        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)

                running_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                total += labels.size(0)

                correct += predictions.eq(labels).sum().item()

        epoch_loss = running_loss / len(validation_loader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy

    #Training loop over 30 epochs
    for epoch in range(epochs):

        #Performs training and validation functions on the corresponding datasets
        training_loss, training_acc = training_loop(model=model, training_loader=training_dataloader,
                                                    optimiser=optimiser, loss_function=loss_function, device=device)
        validation_loss, validation_acc = validation(model=model, validation_loader=validation_dataloader,
                                                     loss_function=loss_function, device=device)
        
        #Saves best model if the validation accuracy is better than previous
        if validation_acc > best_accuracy:
            best_accuracy = validation_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved to best_model.pth with accuracy {(validation_acc* 100):.2f}%")

        #Keeps track of accuracy and loss of the model (useful for debugging purposes)
        print(f"Training: Epoch {epoch+1}/{epochs}, Accuracy: {(training_acc * 100):.2f}%, Loss: {training_loss:.3f}")
        print(f"Validation: Epoch {epoch+1}/{epochs}, Accuracy: {(validation_acc * 100):.2f}%, Loss: {validation_loss:.3f}")

        current_lr = optimiser.param_groups[0]['lr']

        print(f"Learning Rate: {current_lr:.6f}")

        scheduler.step()
    
    #Training complete, outputs best accuracy achieved
    print(f"Training complete, best validation accuracy: {(best_accuracy * 100):.2f}%")