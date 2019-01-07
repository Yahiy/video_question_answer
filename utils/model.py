import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TgifModel(nn.Module):
    def __init__(self):
        super(TgifModel, self).__init__()
        self.video_rnn = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.rnn = nn.LSTM(input_size=300, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(2048, 1746))
        self.c0 = torch.zeros((2,32,1024)).cuda()

    def forward(self, features, questions, ql):
        # questions_packed = pack_padded_sequence(questions, ql, batch_first=True)
        # _, (hn, cn) = self.rnn(questions)
        _, (hn, cn) = self.video_rnn(features)
        _, (hn, cn) = self.rnn(questions, (hn, cn))
        f = torch.cat([hn[0], hn[1]], dim=1)
        output = self.fc(f)
        return output