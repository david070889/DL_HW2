import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定義 CNN + RNN 的混合模型
class CNN_RNN(nn.Module):
    def __init__(self, num_classes=50):
        super(CNN_RNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.rnn_input_size = 64 * 28 * 28  # 假設輸入圖片大小為 224x224
        self.hidden_size = 128
        self.num_layers = 2
        self.rnn = nn.RNN(self.rnn_input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # 將特徵圖展平成序列
        x = x.view(x.size(0), -1, self.rnn_input_size)
        
        # RNN
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out





