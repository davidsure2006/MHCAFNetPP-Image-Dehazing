import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//8),
            nn.ReLU(),
            nn.Linear(c//8, c),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,_,_ = x.shape
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y


class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1 = nn.Conv2d(c,c,3,1,1)
        self.conv2 = nn.Conv2d(c,c,3,1,1)
        self.attn = ChannelAttention(c)
        self.relu = nn.ReLU()

    def forward(self,x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.attn(x)
        x = self.relu(self.conv2(x))
        return x + res


class MHCAFNetPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.e1 = nn.Conv2d(3,64,3,1,1)
        self.e2 = nn.Conv2d(64,128,3,1,1)
        self.e3 = nn.Conv2d(128,256,3,1,1)

        self.b1 = Block(64)
        self.b2 = Block(128)
        self.b3 = Block(256)

        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.d1 = nn.Conv2d(256,128,3,1,1)

        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.d2 = nn.Conv2d(128,64,3,1,1)

        self.final = nn.Conv2d(64,3,1)

    def forward(self,x):
        inp = x

        e1 = self.b1(self.relu(self.e1(x)))
        e2 = self.b2(self.relu(self.e2(self.pool(e1))))
        e3 = self.b3(self.relu(self.e3(self.pool(e2))))

        x = self.up1(e3)
        x = torch.cat([x,e2],1)
        x = self.relu(self.d1(x))

        x = self.up2(x)
        x = torch.cat([x,e1],1)
        x = self.relu(self.d2(x))

        res = self.final(x)
        return torch.clamp(inp - res,0,1)
