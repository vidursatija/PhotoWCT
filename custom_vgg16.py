import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchsummary import summary
import numpy as np

__all__ = [
    'VGG', 'vgg16'
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG16_X_Enc(nn.Module):
    def __init__(self, layers, num_classes=1000, init_weights=True):
        super(VGG16_X_Enc, self).__init__()

        self.features = nn.Sequential(*layers) # layers
        print(self.features)
        # self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        all_maxpools = []
        for l in self.features:
            if isinstance(l, nn.MaxPool2d) == False:
                x = l(x)
            else:
                x, pool_indices = l(x)
                all_maxpools.append(pool_indices)
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x, all_maxpools

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_enc(cfg):
    layers = []
    conv_layers = []
    in_channels = cfg[0]
    cfg = cfg[1:]
    for v in cfg:
        if v == 'M':
            layers += conv_layers # [nn.Sequential(*conv_layers)]
            conv_layers = []
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv_layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if len(conv_layers) > 0:
        layers += conv_layers # [nn.Sequential(*conv_layers)]
    return layers


# cfgs = {
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
# }

VGG_configs_enc = [
    [3, 64],
    [3, 64, 64, 'M', 128],
    [3, 64, 64, 'M', 128, 128, 'M', 256],
    [3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512],
    [3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512]
]

VGG_configs_dec = [
    [64, 3],
    [128, 64, 'M', 64, 3],
    [256, 128, 'M', 128, 64, 'M', 64, 3],
    [512, 256, 'M', 256, 256, 128, 'M', 128, 64, 'M', 64, 3],
    [512, 512, 'M', 512, 512, 256, 'M', 256, 256, 128, 'M', 128, 64, 'M', 64, 3]
]


def _vgg_enc(arch, cfg, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16_X_Enc(make_layers_enc(VGG_configs_enc[cfg]), **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress, strict=False)
        model.load_state_dict(torch.load("vgg16-397923af.pth"), strict=False)
        for p in model.parameters():
            p.requires_grad = False
        model.train(False)
        model.eval()
    return model


def vgg16_enc(x=1, pretrained=False, progress=True, **kwargs):
    return _vgg_enc('vgg16', x-1, pretrained, progress, **kwargs)


class VGG16_X_Dec(nn.Module):
    def __init__(self, layers, num_classes=1000, init_weights=True):
        super(VGG16_X_Dec, self).__init__()

        self.layers = nn.Sequential(*layers)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, all_maxpools):
        ct = -1
        for l in self.layers:
            if isinstance(l, nn.MaxUnpool2d) == False:
                x = l(x)
            else:
                x = l(x, all_maxpools[ct])
                ct -= 1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_dec(cfg):
    layers = []
    conv_layers = []
    in_channels = cfg[0]
    cfg = cfg[1:]
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += conv_layers # [nn.Sequential(*conv_layers)]
            conv_layers = []
            layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
            if i != len(cfg) - 1:
                conv_layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv_layers += [conv2d]
            in_channels = v
    if len(conv_layers) > 0:
        layers += conv_layers # [nn.Sequential(*conv_layers)]
    return layers


def _vgg_dec(arch, cfg, pretrained, pretrained_path=None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16_X_Dec(make_layers_dec(VGG_configs_dec[cfg]), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    return model


def vgg16_dec(x=1, pretrained=False, pretrained_path=None, **kwargs):
    return _vgg_dec('vgg16', x-1, pretrained, pretrained_path, **kwargs)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    encoder = vgg16_enc(x=3, pretrained=True) # .to(device)
    for k in encoder.state_dict():
        print(k)
    summary(encoder, (3, 224, 224), device="cpu")
    z, all_maxpools = encoder(torch.from_numpy(np.zeros([1, 3, 224, 224])).float())

    decoder = vgg16_dec(x=3, pretrained=False) # .to(device)
    for k in decoder.state_dict():
        print(k)
    x_rebuild = decoder(z, all_maxpools)
    # summary(decoder, (256, 56, 56), device="cpu")
