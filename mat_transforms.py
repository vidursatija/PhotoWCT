import numpy as np
import torch
import scipy
import scipy.misc
import scipy.sparse
import scipy.sparse.linalg
from numpy.lib.stride_tricks import as_strided
from PIL import Image

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

# The implementation of the function is heavily borrowed from
# https://github.com/NVIDIA/FastPhotoStyle/blob/master/photo_smooth.py
"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
def smoothen(content_path, stylized_path, lbda=1e-4):
    beta = 1./(1.+lbda)
    content = scipy.misc.imread(content_path, mode='RGB')

    B = scipy.misc.imread(stylized_path, mode='RGB').astype(np.float64)/255
    h1,w1,k = B.shape
    h = h1 - 4
    w = w1 - 4
    B = B[int((h1-h)/2): int((h1-h)/2+h), int((w1-w)/2): int((w1-w)/2+w), :]
    content = scipy.misc.imresize(content, (h, w))
    B = __replication_padding(B, 2)
    content = __replication_padding(content, 2)
    content = content.astype(np.float64)/255
    B = np.reshape(B, (h1*w1, k))

    # computing the affinity matrix W as mentioned in the paper    
    W = __compute_laplacian(content)
    W = W.tocsc()
    # computing the dd degree matrix of W
    dd = W.sum(0)
    dd = np.sqrt(np.power(dd, -1))
    dd = dd.A.squeeze()
    # the degree matrix D
    D = scipy.sparse.csc_matrix((dd, (np.arange(0, w1*h1), np.arange(0, w1*h1)))) # 0.026
    # S is the normalized laplacian matrix
    S = D.dot(W).dot(D)
    # A = (I - beta*S)
    A = scipy.sparse.identity(w1*h1) - beta*S
    A = A.tocsc()
    # We use this to smartly calculate matmul(A^-1, Y) as mentioned in the paper
    solver = scipy.sparse.linalg.factorized(A)
    V = np.zeros((h1*w1, k))
    V[:,0] = solver(B[:,0])
    V[:,1] = solver(B[:,1])
    V[:,2] = solver(B[:,2])
    V = V*(1-beta)
    V = V.reshape(h1, w1, k)
    V = V[2:2+h,2:2+w,:]
    
    img = Image.fromarray(np.uint8(np.clip(V * 255., 0, 255.)))
    return img

# Returns sparse matting laplacian
# The implementation of the function is heavily borrowed from
# https://github.com/MarcoForte/closed-form-matting/blob/master/closed_form_matting.py
# We thank Marco Forte for sharing his code.
def __compute_laplacian(img, eps=10**(-7), win_rad=1):
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    c_h, c_w = h - 2*win_rad, w - 2*win_rad
    win_diam = win_rad*2+1
    indsM = np.arange(h*w).reshape((h, w))
    ravelImg = img.reshape(h*w, d)
    win_inds = self.__rolling_block(indsM, block=(win_diam, win_diam))
    win_inds = win_inds.reshape(c_h, c_w, win_size)
    winI = ravelImg[win_inds]
    win_mu = np.mean(winI, axis=2, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI)/win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))
    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = (1/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))
    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L

def __replication_padding(arr, pad):
    h,w,c = arr.shape
    ans = np.zeros((h+pad*2,w+pad*2,c))
    for i in range(c):
            ans[:,:,i] = np.pad(arr[:,:,i],pad_width=(pad,pad),mode='edge')
    return ans

def __rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)