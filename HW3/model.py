import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        # write your codes here
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        # write your codes here
        out, hidden = self.rnn(input, hidden)
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        # write your codes here
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(CharLSTM, self).__init__()
        # write your codes here
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        # write your codes here
        h_0, c_0 = hidden
        out, hidden = self.lstm(input, (h_0, c_0))
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        # write your codes here
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)
