import torch
import torch.nn as nn
from slowfast.model.slow_fast import resnet50


class TgifModel(nn.Module):
    def __init__(self):
        super(TgifModel, self).__init__()
        self.slow_fast = resnet50(class_num=1024)
        # self.video_rnn = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.rnn = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(2048, 1746))

    def forward(self, imgs, questions):
        # _, (hn, cn) = self.video_rnn(features)
        feature = self.slow_fast(imgs)
        _, (hn, cn) = self.rnn(questions)
        f = torch.cat([hn[0], hn[1]], dim=1)
        f = torch.cat([f, feature], dim=1)
        output = self.fc(f)
        return output