import torch
from label.gaussian_label import gen_gaussian_label
from label.smooth_label import get_angle_smooth_label

def get_gaussian_3d_label(map_size, center, radius, sigma=1,
                          dtype=torch.float32, device='cpu'):
    heatmap_2d = gen_gaussian_label(map_size[1:], center[1:], radius=radius, sigma=sigma,
                                    normalized=True, dtype=dtype, device=device)
    angle_omega = int(360/map_size[0])
    target_label = center[0] * angle_omega
    angle_label = get_angle_smooth_label(target_label, angle_range=360, radius=radius,
                                         omega=angle_omega, sig=sigma, normalized=True)
    angle_label = torch.as_tensor(angle_label,dtype=torch.float32,device=device)
    heatmap_2d = heatmap_2d.unsqueeze(dim=-1)
    heatmap_3d = heatmap_2d * angle_label
    heatmap_3d = heatmap_3d.permute(2,0,1)

    return heatmap_3d

if __name__ == '__main__':
    heatmap_3d = get_gaussian_3d_label((360, 10, 10), (90, 4, 4), radius=2, sigma=2, device='cuda')
    print(heatmap_3d.shape)
    print(heatmap_3d[90,4,4])