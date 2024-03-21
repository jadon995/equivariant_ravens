import torch
from networks.so2_equ_res import so2_res
from networks.so2_pick_angle_model import SO2ResNet as so2_pick_angle
from networks.equ_res_3 import dian_res
from networks.pick_angle_model import EquRes as lite_pick_angle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

''' SO2 1,3,4
irrep_kwargs = {'irrep': 3,
                'sample': 16}

model = so2_res(in_dim=6,out_dim=3,middle_dim=(16, 32, 64, 128),
                init=False, init_method='he', **irrep_kwargs)
'''
'''SO(2) 2
angle_irrep_kwargs = {'irrep': 6,
                      'sample': 36}
model = so2_pick_angle(init=False,N=180,**angle_irrep_kwargs)
'''
'''
model = dian_res(in_dim=6,out_dim=3,N=6,middle_dim=(16, 32, 64, 128),init=False)
'''

model = lite_pick_angle(init=False,N=360)

num_params = count_parameters(model)
print(num_params)