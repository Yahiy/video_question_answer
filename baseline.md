1. model_zoo.py
pytorch model_zoo.py 63line Permission deny,need to change path :

```
# cached_file = os.path.join(model_dir, filename) #/home/stage/.torch/models
cached_file = os.path.join('/home/stage/yuan/models', filename)
```

1. text only 
ep 70 lstm 512 output 1024
```
class TgifModel(nn.Module):
    def __init__(self):
        super(TgifModel, self).__init__()
        self.base_resnet = resnet152(pretrained=True)
        self.video_rnn = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.rnn = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(1024, 1746))

    def forward(self, features, questions):
        _, (hn, cn) = self.rnn(questions) #, (hn, cn))
        f = torch.cat([hn[0], hn[1]], dim=1)
        output = self.fc(f)
        return output
```
