""" Full assembly of the parts to form the complete network """
import torch
from fvcore.nn import flop_count_str, FlopCountAnalysis

from .qunet_parts import *

class QUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(QUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleQConv(n_channels, 32)
        self.down1 = QDown(32, 64)
        self.down2 = QDown(64, 128)
        self.down3 = QDown(128, 256)
        factor = 2 if bilinear else 1
        # self.down4 = QDown(512, 1024 // factor)
        # self.up1 = QUp(1024, 512 // factor, bilinear)
        self.up2 = QUp(256, 128 // factor, bilinear)
        self.up3 = QUp(128, 64 // factor, bilinear)
        self.up4 = QUp(64, 32, bilinear)
        self.outc = QOutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    model = QUNet(n_channels=3, n_classes=1)
    # # model(torch.rand((1, 3, 961, 640)))
    rand_t = torch.rand((1, 3, 256, 256))
    # print(flop_count_str(FlopCountAnalysis(model, rand_t)))
    for name, p in model.named_parameters():
        print(name)