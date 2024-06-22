# importing the required libraries 
import torch 
from torch.utils.data import Dataset
import numpy as np
  
# defining the Dataset class 
class data_set(Dataset): 
    def __init__(self, data, sequence_length, neuron_number):  
        self.data = data
        self.sequence_length = sequence_length
        self.neuron_number = neuron_number
  
    def __len__(self):
        length = 0 
        for session in self.data:
            length += (len(session[1]) - self.sequence_length)
        return length
  
    def __getitem__(self, index): 
        i = index
        s_number = 0
        for session in self.data:
            if i >= (len(session[1]) - self.sequence_length):
                i -= (len(session[1]) - self.sequence_length)
                s_number += 1
        tensor = np.array(self.data[s_number][1].iloc[i : i + self.sequence_length, :self.neuron_number])
        return torch.tensor(tensor, dtype=torch.float32).T