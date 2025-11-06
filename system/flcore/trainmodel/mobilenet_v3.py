import torch
from torch import nn
from torch.hub import load_state_dict_from_url

__all__ = ['MobileNetV3UltraLite', 'mobilenet_v3_ultralite']

class h_swish(nn.Module):
    def forward(self, x):
        return x * torch.clamp(x + 3, 0, 6) / 6

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, norm_layer=None, act_layer=None):
        padding = (kernel_size - 1) // 2
        norm_layer = norm_layer or nn.BatchNorm2d
        act_layer = act_layer or nn.ReLU
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_ch),
            act_layer()
        )

class InvertedResidualLite(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNAct(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV3UltraLite(nn.Module):
    def __init__(self, num_classes=1000, width_mult=0.5, dropout=0.1):
        super().__init__()
        # 아주 얕은 구조
        self.cfgs = [
            [1,  8, 1, 1],
            [4, 16, 2, 2],
            [4, 24, 2, 2],
            [4, 40, 2, 2],
            [4, 64, 2, 1],
        ]
        input_channel = 8
        layers = [ConvBNAct(3, input_channel, stride=2, act_layer=h_swish)]
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualLite(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        layers.append(ConvBNAct(input_channel, 128, kernel_size=1, act_layer=h_swish))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

def mobilenet_v3_ultralite(pretrained=False, **kwargs):
    model = MobileNetV3UltraLite(**kwargs)
    if pretrained:
        # 없음 (ImageNet weight 불필요)
        pass
    return model
