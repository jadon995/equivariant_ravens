import torch
from label.gaussian_label import gen_gaussian_label
from label.smooth_label import get_angle_smooth_label

# def gen_gaussian_label(map_size, center, radius=1, sigma=1, 
                    #    normalized=False, dtype=torch.float32, device='cpu'):

# get_angle_smooth_label(angle_label, angle_range=180, label_type=0, 
                        #    radius=4, omega=1, normalized=False):
def get_gaussian_3d_label(map_size, center, radius, 
                          dtype=torch.float32, device='cpu'):
    heatmap_2d = gen_gaussian_label(map_size[1:], center[1:], radius=radius, sigma=radius,
                                    normalized=True, dtype=dtype, device=device)
    angle_label = get_angle_smooth_label(center[0], angle_range=map_size[0], radius=radius, normalized=True)
    angle_label = torch.as_tensor(angle_label,dtype=torch.float32,device=device)
    heatmap_2d = heatmap_2d.unsqueeze(dim=-1)
    heatmap_3d = heatmap_2d * angle_label
    heatmap_3d = heatmap_3d.permute(2,0,1)

    return heatmap_3d

if __name__ == '__main__':
    heatmap_3d = get_gaussian_3d_label((360, 10, 10), (90, 4, 4), radius=1, device='cuda')
    print(heatmap_3d.shape)
    print(heatmap_3d[90,4,4])