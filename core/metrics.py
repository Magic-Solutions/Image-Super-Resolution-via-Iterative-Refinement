import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB or grayscale
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # Clamp values
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # Normalize to [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:  # Batch of images
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    elif n_dim == 3:  # Single image with channels
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    elif n_dim == 2:  # Single grayscale image
        img_np = tensor.numpy()
        img_np = np.expand_dims(img_np, axis=-1)  # Add channel dimension for consistency
    else:
        raise TypeError(f'Unsupported tensor dimension: {n_dim}')
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()  # Scale to [0, 255]
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    '''
    Saves an image to the specified path
    Supports RGB and grayscale images.
    '''
    if img.shape[-1] == 1:  # Grayscale image
        img = img.squeeze(axis=-1)  # Remove channel for OpenCV
        cv2.imwrite(img_path, img)
    else:  # RGB image
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def calculate_psnr(img1, img2):
    '''
    Calculate PSNR between two images.
    Assumes img1 and img2 are in range [0, 255].
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    '''
    Compute SSIM for two images.
    Supports single-channel images.
    '''
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''
    Calculate SSIM between two images.
    Supports RGB and grayscale images.
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:  # Grayscale images
        return ssim(img1, img2)
    elif img1.ndim == 3:  # Color or grayscale images with channel dimension
        if img1.shape[2] == 3:  # RGB
            ssims = [ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]
            return np.mean(ssims)
        elif img1.shape[2] == 1:  # Grayscale
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Unsupported image dimensions.')