import torch
from torch import nn, Tensor
import torchvision.models as models

import math

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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DETR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = backbone()

        # 논문에 d가 명시되어있지 않아서 512로 임의로 지정하였음 
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)

        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 트랜스포터 디코더
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=2)
        self.transformerDecoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # 포지셔널 인코딩
        self.positionalEncoding = PositionalEncoding(d_model = 512)

        # # FFNs
        # self.ffns = 


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.features(x)
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        x = self.conv1(x).permute(0, 2, 1)

        x = self.positionalEncoding(x)

        mem = self.transformerEncoder(x)
        x = self.transformerDecoder(x, mem)

        return x



if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # input 사이즈 너무 큰데 detection쪽에서는 resizing 안하나?
    img = torch.Tensor(1, 3, 1024, 1024).type(torch.float32).to(device)
    model = DETR().to(device)
    print(model(img).size())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', params)
