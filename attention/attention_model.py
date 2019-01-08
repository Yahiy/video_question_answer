import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TgifModel(nn.Module):
    def __init__(self):
        super(TgifModel, self).__init__()
        self.video_rnn = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.rnn = nn.LSTM(input_size=300, hidden_size=1024, num_layers=2, batch_first=True)
        # renn cell is for get all hidden states in video
        self.rnn_cell = nn.LSTMCell(input_size=2048, hidden_size=1024)
        self.fc_cls = nn.Sequential(nn.Linear(2048, 1746))
        self.fc_w = nn.Sequential(nn.Linear(2048, 2048))
        self.fc_att = nn.Parameter(torch.randn(2048,1))

    def forward(self, features, questions, ql):
        # _, (hn, cn) = self.video_rnn(features)
        vid_states= []
        for i in range(features.shape[1]):
            input = features[:,i]
            hx1, cx1 = self.rnn_cell(input)
            hx2, cx2 = self.rnn_cell(input, (hx1, cx1))
            vid_states.append(torch.cat([hx1, hx2], dim=1))

        _, (hn, cn) = self.rnn(questions,
                               (torch.stack([hx1, hx2]), torch.stack([cx1, cx2])))

        hv = torch.stack(vid_states, 1)
        hq = torch.cat([hn[0], hn[1]], dim=1)
        hq_ = torch.reshape(hq,(hq.shape[0],1,2048))
        hq_ = hq_.expand(hq.shape[0],35,2048)
        h_add = torch.add(hv, hq_)
        f_att = torch.matmul(torch.tanh(h_add), self.fc_att)
        f_att = F.softmax(f_att, dim=1)
        f = torch.matmul(hv.permute((0,2,1)), f_att)

        f = self.fc_w(f.reshape(hq.shape[0], 2048))
        output = self.fc_cls(torch.add(torch.tanh(f),hq))
        return output