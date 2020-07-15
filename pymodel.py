import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, out_sz):
        super(Network, self).__init__()
        
        self.conv1 =  nn.Conv2d(3, 24, 5, 1, 2) # (160, 320)
        self.conv2 =  nn.Conv2d(24, 36, 5, 1, 2) # (80, 160)
        self.conv3 =  nn.Conv2d(36, 48, 3, 1, 1) # (40, 80)
        self.conv4 =  nn.Conv2d(48, 64, 3, 1, 1) # (20, 40)
        self.fc1 = nn.Linear(10*20*64, 2000) # (10, 20)
        self.fc2 = nn.Linear(2000, 512)
        self.fc3 = nn.Linear(512, out_sz)
        self.maxpool= nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, images):
        batch_size = images.shape[0]
        x = F.relu(self.maxpool(self.conv1(images)))
        x = self.dropout(x)
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.maxpool(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.maxpool(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(batch_size, -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Model = Network(label_sz)
Model.to(device)