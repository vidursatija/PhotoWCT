import numpy as np
from scipy.signal import convolve2d as c2d
import torch

def whitening(fc):
    # f = [C, H*W]
    mc = torch.mean(fc, dim=-1) # [C]
    mc = mc.unsqueeze(1) # .reshape(mc, [-1, 1])
    fc -= mc

    covar = torch.matmul(fc, torch.transpose(fc, 0, 1)) # [C, C]

    eigenvalues, Ec = torch.symeig(covar, eigenvectors=True) # np.linalg.eigh(covar) # ([C], [C, C])

    eigenvalues = torch.pow(eigenvalues, -0.5)
    Dc = torch.diag(eigenvalues) # [C, C]

    mid = torch.matmul(Ec, torch.matmul(Dc, torch.transpose(Ec, 0, 1)))

    return torch.matmul(mid, fc) # [C, H*W]

def colouring(fs, fc_hat):
    # f = [C, H*W]
    ms = torch.mean(fs, dim=-1) # [C]
    ms = ms.unsqueeze(1) # np.reshape(ms, [-1, 1])
    fs -= ms

    covar = torch.matmul(fs, torch.transpose(fs, 0, 1)) # [C, C]

    eigenvalues, Es = torch.symeig(covar, eigenvectors=True) # ([C], [C, C])

    eigenvalues = torch.pow(eigenvalues, 0.5)
    Dc = torch.diag(eigenvalues) # [C, C]

    mid = torch.matmul(Es, torch.matmul(Dc, torch.transpose(Es, 0, 1)))

    return torch.matmul(mid, fc_hat) + ms

# def toLinear(x, y, w):
#     return y*w + x

# def channel_affmat(image):
#     # image = [H, W]
#     h, w = image.shape
#     padded_image = np.pad(image, [(1, 1), (1, 1)], 'reflect') # [2+H+2, 2+W+2]

#     W = np.zeros((h*w, h*w))
#     for i in range(h):
#         padded_i = i+1
#         for j in range(w):
#             padded_j = j+1
#             center = toLinear(i, j, w)

#             # 4 connected
#             if i != h-1:
#                 W[center, toLinear(i+1, j, w)] = 1
#                 W[toLinear(i+1, j, w), center] = 1

#             if i != 0:
#                 W[center, toLinear(i-1, j, w)] = 1
#                 W[toLinear(i-1, j, w), center] = 1

#             if j != w-1:
#                 W[center, toLinear(i, j+1, w)] = 1
#                 W[toLinear(i, j+1, w), center] = 1

#             if j != 0:
#                 W[center, toLinear(i, j-1, w)] = 1
#                 W[toLinear(i, j-1, w), center] = 1

#             # 8 connected
#             if i != h-1 and j != w-1:
#                 W[center, toLinear(i+1, j+1, w)] = 1
#                 W[toLinear(i+1, j+1, w), center] = 1

#             if i != 0 and j != w-1:
#                 W[center, toLinear(i-1, j+1, w)] = 1
#                 W[toLinear(i-1, j+1, w), center] = 1

#             if i != h-1 and j != 0:
#                 W[center, toLinear(i+1, j-1, w)] = 1
#                 W[toLinear(i+1, j-1, w), center] = 1

#             if i != 0 and j != 0:
#                 W[center, toLinear(i-1, j-1, w)] = 1
#                 W[toLinear(i-1, j-1, w), center] = 1

            # part = padded_image[i: i+3, j: j+3].reshape([-1])
            # center = padded_image[i+1, j+1]
            # sigma = part.stddev()
            # if i > 0 and i < h-1:
            #     if j > 0 and j < w-1:
            #         W[i*w+j, (i-1)*w+j-1] = ((part[0]-center)/sigma)**2
            #         W[i*w+j, (i-1)*w+j] = ((part[1]-center)/sigma)**2
            #         W[i*w+j, (i-1)*w+j+1] = ((part[2]-center)/sigma)**2

            #         W[i*w+j, i*w+j-1] = ((part[3]-center)/sigma)**2
            #         # W[i*w+j, i*w+j] = ((part[4]-center)/sigma)**2
            #         W[i*w+j, i*w+j+1] = ((part[5]-center)/sigma)**2

            #         W[i*w+j, (i+1)*w+j-1] = ((part[6]-center)/sigma)**2
            #         W[i*w+j, (i+1)*w+j] = ((part[7]-center)/sigma)**2
            #         W[i*w+j, (i+1)*w+j+1] = ((part[8]-center)/sigma)**2
            #     else:
            #         # think
            #         pass

    # k = 3
    # mean_filter = np.ones([k, k])/(k*k)

    # mean_image = c2d(padded_image, mean_filter, mode='valid') # [H, W]

    # # 0 1 2
    # # 3 4 5
    # # 6 7 8

    # # ix = [1+H+1, 1+W+1]
    # i0 = np.abs(padded_image - np.pad(mean_image, [(0, 2), (0, 2)], 'constant'))
    # i1 = np.abs(padded_image - np.pad(mean_image, [(0, 2), (1, 1)], 'constant'))
    # i2 = np.abs(padded_image - np.pad(mean_image, [(0, 2), (2, 0)], 'constant'))

    # i3 = np.abs(padded_image - np.pad(mean_image, [(1, 1), (0, 2)], 'constant'))
    # i4 = np.abs(padded_image - np.pad(mean_image, [(1, 1), (1, 1)], 'constant'))
    # i5 = np.abs(padded_image - np.pad(mean_image, [(1, 1), (2, 0)], 'constant'))

    # i6 = np.abs(padded_image - np.pad(mean_image, [(2, 0), (0, 2)], 'constant'))
    # i7 = np.abs(padded_image - np.pad(mean_image, [(2, 0), (1, 1)], 'constant'))
    # i9 = np.abs(padded_image - np.pad(mean_image, [(2, 0), (2, 0)], 'constant'))