import torch
from torch import nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

classifier = torch.nn.Sequential(
          torch.nn.Conv2d(3, 8, 3, padding=1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2),
          torch.nn.Conv2d(8, 16, 3, padding=1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2),
          Flatten(),
          torch.nn.Linear(16*16*16, 1)
        ).to(device)
classifier.load_state_dict(torch.load('./toTransfer/classifier.pt'))

class Generator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

z_dim = 100
generator = Generator(z_dim)
generator.load_state_dict(torch.load('toTransfer/generator.pt')['G'])

classifier = classifier.eval()
generator = generator.eval()

