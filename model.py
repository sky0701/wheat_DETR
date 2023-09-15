import torch
from torch import nn, Tensor
import torchvision.models as models

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.features = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                      nn.Sequential(nn.Conv2d(1024, 2048, (3, 3), padding=(1, 1)),
                                                    nn.BatchNorm2d(2048),
                                                    nn.ReLU()))

    def forward(self, x):
        b, h, w, c = x.shape
        x = self.features(x)

        return x

class DETR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = backbone()
        # 논문에 d가 명시되어있지 않아서 512로 임의로 지정하였음 
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=5)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.features(x)
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        x = self.conv1(x)

        return x



if __name__=='__main__':

    img = torch.rand(5, 3, 1024, 1024)
    model = backbone()
    print(model(img).size())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', params)
