import torch
import torch.nn as nn

class DeepGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(DeepGRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=1,
                          batch_first=True, bidirectional=bidirectional, dropout=0.1)
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.linear = nn.Linear(hidden_size*self.num_directions, output_size, )
        #self.act = nn.Tanh()
        self.act = nn.Sigmoid()

    def forward(self, x):
        pred, _ = self.rnn(x, None)
        pred = self.act(self.linear(pred))
        return pred
