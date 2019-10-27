# PhotoWCT with closed form matting
**NOTE: TO BE UPDATED**

Torch implementation of the papers [Universal Style Transfer](https://arxiv.org/pdf/1705.08086.pdf) and [A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/abs/1802.06474)

This is an unofficial implementation.
The original implementation of [Universal Style Transfer](https://github.com/Yijunmaverick/UniversalStyleTransfer) and [A Closed-form Solution to Photorealistic Image Stylization](https://github.com/NVIDIA/FastPhotoStyle) and there <-

## How to get it running
1. Get the 2017 MS COCO train and validation datasets and unzip them
2. Download PyTorch VGG16 model

```wget https://download.pytorch.org/models/vgg16-397923af.pth```

3. For every layer(x = 1 to 5) train the decoder. It is recommended to run training twice with starting lr 0.001 and then 0.0001

```python3 --x <layer number> --batch_size <64> --decoder <saved checkpoint if any> --optimizer <optimized checkpoint if any>```

*Note: all decoders & optimizers are saved in the dir `decoder_<x>`*

4. Run the model on your style and content image

```python3 run_wct.py --x <number of layers to style 1/2/3/4/5> --style <path to style> --content <path to content> --output <output file name> --decoders <comma separated decoder pickle files>```

5. Smooth out the image using affine matting

**NOTE: Affine Matting mentioned in the latter paper is still in progress**

## Results
TO BE UPDATED

## TODO
1. Affine Matting
2. Add results
3. Upload trained files
4. Make training easier
5. Make running easier