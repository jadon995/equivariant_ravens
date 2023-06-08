from math import sqrt

import torch
import torch.nn.functional as F


def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def gen_gaussian_label(map_size, center, radius=1, sigma=1, 
                       normalized=False, dtype=torch.float32, device='cpu'):
    x, y = center
    width, height = map_size
    
    assert 0<=x<width
    assert 0<=y<height
    assert radius==sigma # at this stage


    heatmap = torch.zeros((width+2*radius), (height+2*radius),
                          dtype=torch.float32, device=device)
    gaussian_kernel = gaussian2D(radius, sigma, dtype, device)
    heatmap[x:x+2*radius+1, y:y+2*radius+1] = gaussian_kernel
    heatmap = heatmap[radius:-radius, radius:-radius]
    if normalized: heatmap /= heatmap.sum()
    return heatmap


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                      radius - left:radius + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap

if __name__ == '__main__':
    # gaussian_kernel = gaussian2D(1, sigma=1)
    # print(gaussian_kernel/gaussian_kernel.sum())

    # heatmap = torch.zeros((10,10), dtype=torch.float32)
    # heatmap = gen_gaussian_target(heatmap, (0, 0), 1, k=1)
    # print(heatmap)

    heatmap = gen_gaussian_label((8,8), (3,3), 2, 2, normalized=True, device='cuda')
    print(heatmap.shape)
    print(heatmap)

