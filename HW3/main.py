import dataset as ds
from model import CharRNN, CharLSTM
from generate import generate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def one_hot_encode(sequence, num_classes):
    batch_size = sequence.shape[0]
    seq_length = sequence.shape[1]
    encoding = np.zeros((batch_size, seq_length, num_classes), dtype=np.float32)
    for i in range(batch_size):
        for j in range(seq_length):
            encoding[i, j, sequence[i, j]] = 1
    return encoding

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = torch.tensor(one_hot_encode(inputs.cpu().numpy(), model.fc.out_features)).to(device)
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = torch.tensor(one_hot_encode(inputs.cpu().numpy(), model.fc.out_features)).to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    input_file = 'shakespeare_train.txt'
    batch_size = 64
    hidden_size = 128
    num_layers = 2
    epochs = 20
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ds.Shakespeare(input_file)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    model_rnn = CharRNN(len(dataset.chars), hidden_size, num_layers).to(device)
    model_lstm = CharLSTM(len(dataset.chars), hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=learning_rate)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learning_rate)

    trn_losses_rnn, val_losses_rnn = [], []
    trn_losses_lstm, val_losses_lstm = [], []

    for epoch in range(epochs):
        trn_loss_rnn = train(model_rnn, train_loader, device, criterion, optimizer_rnn)
        val_loss_rnn = validate(model_rnn, val_loader, device, criterion)
        trn_losses_rnn.append(trn_loss_rnn)
        val_losses_rnn.append(val_loss_rnn)

        trn_loss_lstm = train(model_lstm, train_loader, device, criterion, optimizer_lstm)
        val_loss_lstm = validate(model_lstm, val_loader, device, criterion)
        trn_losses_lstm.append(trn_loss_lstm)
        val_losses_lstm.append(val_loss_lstm)

        print('Epoch {}/{}: RNN Loss: {:.4f}/{:.4f}, LSTM Loss: {:.4f}/{:.4f}'.format(epoch + 1, epochs, trn_loss_rnn, val_loss_rnn, trn_loss_lstm, val_loss_lstm))

    plt.figure(figsize=(12, 6))
    plt.plot(trn_losses_rnn, label='RNN Train Loss')
    plt.plot(val_losses_rnn, label='RNN Validation Loss')
    plt.plot(trn_losses_lstm, label='LSTM Train Loss')
    plt.plot(val_losses_lstm, label='LSTM Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')

    torch.save(model_rnn.state_dict(), 'char_rnn_model.pt')
    torch.save(model_lstm.state_dict(), 'char_lstm_model.pt')

    best_model = model_lstm if min(val_losses_lstm) < min(val_losses_rnn) else model_rnn
    best_model_name = "LSTM" if best_model == model_lstm else "RNN"
    
    seeds = ["A", "B", "C", "D", "E"]
    
    print('Generated Text Examples using the best model {}'.format(best_model_name))
    for seed in seeds:
        generated_text = generate(best_model, seed, 1.0, dataset.char_to_idx, dataset.idx_to_char, length=100)
        print('Seed {} : {}'.format(seed, generated_text))

    best_model_name = "char_lstm_model.pt" if min(val_losses_lstm) < min(val_losses_rnn) else "char_rnn_model.pt"
    with open("best_model.txt", "w") as f:
        f.write(best_model_name)

if __name__ == '__main__':
    main()
