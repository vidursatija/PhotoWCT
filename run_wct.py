import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import custom_vgg16 as cvgg16
import mat_transforms

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--decoders', type=str, default=None,
                    help='Decoder path')
parser.add_argument('--x', type=int, default=2,
                    help='Num layers to transform')
parser.add_argument('--style', type=str, default=None,
                    help='Style image path')
parser.add_argument('--content', type=str, default=None,
                    help='Content image path')
parser.add_argument('--output', type=str, default='stylized.png',
                    help='Output image path')
parser.add_argument('--smooth', help='Whether to apply smoothing')

args = parser.parse_args()


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


transform = transforms.Compose([
#      transforms.RandomResizedCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

reverse_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1./0.229, 1./0.224, 1./0.225])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

decoder_paths = args.decoder.split(",")

encoders = [cvgg16.vgg16_enc(x=j+1, pretrained=True).to(device) for j in range(args.x)]
decoders = [cvgg16.vgg16_dec(x=j+1, pretrained=True, pretrained_path=decoder_paths[j]).to(device) for j in range(args.x)]
# encoder = cvgg16.vgg16_enc(x=args.x, pretrained=True).to(device)
# decoder = cvgg16.vgg16_dec(x=args.x, pretrained=True, pretrained_path=args.decoder).to(device)

content_image = image_loader(transform, args.content).to(device)
style_image = image_loader(transform, args.style).to(device)

for j in range(args.x, 0, -1):
    z_content, maxpool_content = encoders[j-1](content_image) # (1, C, H, W)
    z_style, _ = encoders[j-1](style_image) # (1, C, H, W)

    n_channels = z_content.size()[1] # C
    n_1 = z_content.size()[2] # H
    n_2 = z_content.size()[3] # W

    z_content = z_content.squeeze(0).view([n_channels, -1]) # (C, HW)
    z_style = z_style.squeeze(0).view([n_channels, -1]) # (C, HW)

    white_content = mat_transforms.whitening(z_content) # (C, HW)
    color_content = mat_transforms.colouring(z_style, white_content) # (C, HW)

    alpha = 0.6
    color_content = alpha*color_content + (1.-alpha)*z_content

    color_content = color_content.view([1, n_channels, n_1, n_2]) # (1, C, H, W)
    # color_content = color_content.unsqueeze(0) # (1, C, H, W)

    content_image = decoders[j-1](color_content.to(device), maxpool_content) # (1, C, H, W)

new_image = content_image.squeeze(0) # (C, H, W)
new_image = reverse_normalize(new_image) # (C, H, W)
new_image = torch.transpose(new_image, 0, 1) # (H, C, W)
new_image = torch.transpose(new_image, 1, 2) # (H, W, C)

new_image = np.maximum(np.minimum(new_image.cpu().detach().numpy(), 1.0), 0.0)

result = Image.fromarray((new_image * 255).astype(np.uint8))
result.save(args.output)
