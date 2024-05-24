import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset
from model import LeNet5, CustomMLP

import matplotlib.pyplot as plt

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.train()
    
    total_loss = 0
    correct = 0
    
    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    trn_loss = total_loss / len(trn_loader.dataset)
    acc = 100. * correct / len(trn_loader.dataset)   

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    tst_loss = total_loss / len(tst_loader.dataset)
    acc = 100. * correct / len(tst_loader.dataset)    
    

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    
    train_dataset = dataset.MNIST(data_dir='/home/ishwang/hw/ishwang/mnist-classification/data/train')
    test_dataset = dataset.MNIST(data_dir='/home/ishwang/hw/ishwang/mnist-classification/data/test')

    trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    models = {
        "LeNet-5": LeNet5().to(device),
        "Custom MLP": CustomMLP().to(device)
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for name, model in models.items():
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        for epoch in range(100):
            trn_loss, trn_acc = train(model, trn_loader, device, criterion, optimizer)
            tst_loss, tst_acc = test(model, tst_loader, device, criterion)
            train_loss.append(trn_loss)
            train_acc.append(trn_acc)
            test_loss.append(tst_loss)
            test_acc.append(tst_acc)
            print(f'Epoch {epoch+1}, {name} - Train Loss: {trn_loss:.4f}, Accuracy: {trn_acc:.2f}%, Test Loss: {tst_loss:.4f}, Test Accuracy: {tst_acc:.2f}%')
            
        plot_metrics(train_loss, train_acc, test_loss, test_acc, title=name)
        
def plot_metrics(train_loss, train_acc, test_loss, test_acc, title):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.plot(epochs, test_loss, 'b-', label='Test Loss')
    plt.title(f'{title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Train Accuracy')
    plt.plot(epochs, test_acc, 'b-', label='Test Accuracy')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_metrics.png")
    plt.close()        

if __name__ == '__main__':
    
    main()
