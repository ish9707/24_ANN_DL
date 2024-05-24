import torch
import numpy as np
import dataset as ds
from model import CharRNN, CharLSTM

def one_hot_encode(sequence, num_classes):
    encoding = np.zeros((len(sequence), num_classes), dtype=np.float32)
    for idx, val in enumerate(sequence):
        encoding[idx, val] = 1
    return encoding

def generate(model, seed_characters, temperature, char_to_idx, idx_to_char, length=100):
    model.eval()
    inputs = torch.tensor([[char_to_idx[ch] for ch in seed_characters]]).to(next(model.parameters()).device)
    inputs = torch.tensor(one_hot_encode(inputs.cpu().numpy()[0], len(char_to_idx))).unsqueeze(0).to(next(model.parameters()).device)
    hidden = model.init_hidden(inputs.size(0))
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(next(model.parameters()).device) for h in hidden)
    else:
        hidden = hidden.to(next(model.parameters()).device)

    generated_text = seed_characters
    for _ in range(length):
        output, hidden = model(inputs, hidden)

        if len(output.shape) == 2:
            output = output / temperature
        elif len(output.shape) == 3:
            output = output[:, -1, :] / temperature
        else:
            raise ValueError("Unexpected output shape: {}".format(output.shape))

        probabilities = torch.softmax(output, dim=-1).squeeze().detach().cpu().numpy()

        if probabilities.ndim > 1:
            probabilities = probabilities[-1]

        next_idx = np.random.choice(len(probabilities), p=probabilities)
        next_char = idx_to_char[next_idx]
        generated_text += next_char
        
        inputs = torch.tensor([[next_idx]]).to(next(model.parameters()).device)
        inputs = torch.tensor(one_hot_encode(inputs.cpu().numpy()[0], len(char_to_idx))).unsqueeze(0).to(next(model.parameters()).device)

    return generated_text

if __name__ == '__main__':
    input_file = 'shakespeare_train.txt'
    model_file_rnn = 'char_rnn_model.pt'
    model_file_lstm = 'char_lstm_model.pt'
    hidden_size = 128
    num_layers = 2
    seed_characters = "The future belongs to those who believe in the beauty of their dreams."
    length = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ds.Shakespeare(input_file)

    model_rnn = CharRNN(len(dataset.chars), hidden_size, num_layers).to(device)
    model_rnn.load_state_dict(torch.load(model_file_rnn))
    model_rnn.eval()

    model_lstm = CharLSTM(len(dataset.chars), hidden_size, num_layers).to(device)
    model_lstm.load_state_dict(torch.load(model_file_lstm))
    model_lstm.eval()

    with open("best_model.txt", "r") as f:
        best_model_name = f.read().strip()

    if best_model_name == "char_lstm_model.pt":
        best_model = model_lstm
    else:
        best_model = model_rnn

    temperatures = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
    print('Generated Text Examples using the best model ({}) with different temperatures:'.format(best_model_name))
    for temp in temperatures:
        print('\nTemperature: {}'.format(temp))
        generated_text = generate(best_model, seed_characters, temp, dataset.char_to_idx, dataset.idx_to_char, length)
        print(generated_text)

    # seeds = ["A", "B", "C", "D", "E"]
    # print('Generated Text Examples using the best model ({}) with different seed characters:'.format(best_model_name))
    # for seed in seeds:
    #     generated_text = generate(best_model, seed, 1.0, dataset.char_to_idx, dataset.idx_to_char, length=100)
    #     print('Seed {}: {}'.format(seed, generated_text))
