# import some packages you need here
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):

    def __init__(self, input_file):
        # write your codes here
        with open(input_file, 'r') as file:
            self.text = file.read()

        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = [self.char_to_idx[ch] for ch in self.text]
        self.seq_length = 30
        self.num_samples = len(self.data) - self.seq_length

    def __len__(self):
        # write your codes here
        return self.num_samples

    def __getitem__(self, idx):
        # write your codes here
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

if __name__ == '__main__':
    # write test codes to verify your implementations
    dataset = Shakespeare('shakespeare.txt')
    print(dataset[0])
