import torch.nn as nn
class Generator(nn.Module):
    def __init__(self,ngf,nz):
        super(Generator,self).__init__()
        self.layer1=nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True)
        )
        self.layer2=nn.Sequential(
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True)
        )
        self.layer3=nn.Sequential(
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        self.layer4=nn.Sequential(
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        self.layer5=nn.Sequential(
            nn.ConvTranspose2d(ngf,3,5,3,1,bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        return out

class Discriminator(nn.Module):
    def __init__(self,ndf):
        super(Discriminator,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,ndf,kernel_size=5,stride=3,padding=1,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
        






