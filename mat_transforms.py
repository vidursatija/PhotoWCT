import numpy as np
from scipy.signal import convolve2d as c2d

def whitening(fc):
    # f = [C, H*W]
    mc = np.mean(fc, axis=-1) # [C]
    mc = np.reshape(mc, [-1, 1])
    fc -= mc

    covar = np.matmul(fc, fc.T) # [C, C]

    eigenvalues, Ec = np.linalg.eigh(covar) # ([C], [C, C])

    eigenvalues = np.power(eigenvalues, -0.5)
    Dc = np.diag(eigenvalues) # [C, C]

    mid = np.matmul(Ec, np.matmul(Dc, Ec.T))

    return np.matmul(mid, fc)

def colouring(fs, fc_hat):
    # f = [C, H*W]
    ms = np.mean(fs, axis=-1) # [C]
    ms = np.reshape(ms, [-1, 1])
    fs -= ms

    covar = np.matmul(fs, fs.T) # [C, C]

    eigenvalues, Es = np.linalg.eigh(covar) # ([C], [C, C])

    eigenvalues = np.power(eigenvalues, 0.5)
    Dc = np.diag(eigenvalues) # [C, C]

    mid = np.matmul(Es, np.matmul(Dc, Es.T))

    return np.matmul(mid, fc_hat) + ms

def channel_affmat(image):
    # image = [H, W]
    h, w = image.shape
    padded_image = np.pad(image, [(1, 1), (1, 1)], 'reflect') # [2+H+2, 2+W+2]

    W = np.zeros((h*w, h*w))
    for i in range(h):
        for j in range(w):
            part = padded_image[i: i+3, j: j+3].reshape([-1])
            center = padded_image[i+1, j+1]
            sigma = part.stddev()
            if i > 0 and i < h-1:
                if j > 0 and j < w-1:
                    W[i*w+j, (i-1)*w+j-1] = ((part[0]-center)/sigma)**2
                    W[i*w+j, (i-1)*w+j] = ((part[1]-center)/sigma)**2
                    W[i*w+j, (i-1)*w+j+1] = ((part[2]-center)/sigma)**2

                    W[i*w+j, i*w+j-1] = ((part[3]-center)/sigma)**2
                    # W[i*w+j, i*w+j] = ((part[4]-center)/sigma)**2
                    W[i*w+j, i*w+j+1] = ((part[5]-center)/sigma)**2

                    W[i*w+j, (i+1)*w+j-1] = ((part[6]-center)/sigma)**2
                    W[i*w+j, (i+1)*w+j] = ((part[7]-center)/sigma)**2
                    W[i*w+j, (i+1)*w+j+1] = ((part[8]-center)/sigma)**2
                else:
                    # think
                    pass

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