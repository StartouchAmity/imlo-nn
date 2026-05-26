import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
try:
    from train import NeuralNetwork
except:
    print("Neural Network class not found. Please make sure you are running this program in the same directory as train.py.")

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64

    running_loss = 0.0
    correct = 0
    total = 0

    transform_eval = transforms.Compose([
        transforms.Resize((188, 188)),
        transforms.CenterCrop((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.OxfordIIITPet(root="data", split="test", transform=transform_eval, download=True)
    training_dataset = datasets.OxfordIIITPet(root="data", split="trainval", transform=transform_eval, download=True)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNetwork()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device=device)

    loss_function = nn.CrossEntropyLoss()

    def testing(model, test_loader, loss_function, device):

        running_loss = 0
        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)

                running_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                total += labels.size(0)

                correct += predictions.eq(labels).sum().item()

        test_accuracy = correct / total
        test_loss = running_loss / len(test_loader)

        return test_accuracy, test_loss
    
    accuracy, loss = testing(model=model, test_loader=test_dataloader, loss_function=loss_function, device=device)
    training_accuracy, training_loss = testing(model=model, test_loader=training_dataloader, loss_function=loss_function, device=device)

    print(f"Testing Data: Accuracy of the model on the Oxford IIIT Pet testing dataset is {(accuracy*100):.2f}%, loss is {loss:.3f}")
    print(f"Training Data: Accuracy of the model on the Oxford IIIT Pet training dataset is {(training_accuracy*100):.2f}%, loss is {training_loss:.3f}")