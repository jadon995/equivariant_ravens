import numpy as np
import math
import torch

def smooth_label(target, num_tokens, smooth_factor=0.0, device='cpu'):
    one_hot = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    one_hot.fill_(smooth_factor / (num_tokens - 1))
    one_hot[target] = 1 - smooth_factor
    return one_hot

if __name__ == '__main__':
    label = smooth_label(4, 10, 0.2)
    print(label)
    print(label.sum())
    print(label.shape)