import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import numpy as np
import torch
# import e2cnn
# from e2cnn import gspaces
import torch.nn.functional as F
# import e2cnn.nn as enn
from equ_res_3 import dian_res
from pick_angle_model import EquRes as lite_pick_angle
from pick_angle_model_2 import EquRes as pick_angle
from label.smooth_label import get_angle_smooth_label as get_smooth_label
from label.gaussian_label import gen_gaussian_label as get_gaussian_2d_label
from label.label_smoothing import smooth_label

class Attention:
    def __init__(self,in_shape,n_rotations,preprocess,device,
                 network_params={},init=False):
        # TODO BY HAOJIE: add lite model
        self.device = device
        self.preprocess = preprocess
        self.n_rotations = n_rotations
        max_dim = np.max(in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        # self.gspace = gspaces.Rot2dOnR2(4)
        # self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_shape[-1])

        self.pos_label_type = network_params['position']['label_type']
        self.pos_label_radius = network_params['position']['label_radius']
        self.pos_label_sigma = network_params['position']['label_sigma']
        self.pos_label_smooth = network_params['position']['label_smooth']
        self.angle_label_type = network_params['angle']['label_type']
        self.angle_label_radius = network_params['angle']['label_radius']
        self.angle_label_sigma = network_params['angle']['label_sigma']
        self.angle_label_smooth = network_params['angle']['label_smooth']

        pos_Cn = network_params['position']['N']
        
        if network_params['position']['lite']:
          self.model = dian_res(in_dim=6,out_dim=1,N=pos_Cn,middle_dim=(16, 32, 64, 128),init=init).to(self.device)
        else:
          self.model = dian_res(in_dim=6,out_dim=1,N=pos_Cn,middle_dim=(32, 64, 128, 256),init=init).to(self.device)
        if network_params['angle']['lite']:
          self.angle_model = lite_pick_angle(init=init,N=self.n_rotations).to(self.device)
          self.crop_size = 64
        else:
          self.angle_model = pick_angle(init=init,N=self.n_rotations).to(self.device)
          self.crop_size = 96
        
        self.pad_size_2 = int(self.crop_size / 2)
        self.padding_2 = np.zeros((3, 2), dtype=int)
        self.padding_2[:2, :] = self.pad_size_2
        
        #self.parameters = list(self.model.parameters()) + list(self.angle_model.parameters())
        self.optim1 = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.optim2 = torch.optim.Adam(self.angle_model.parameters(),lr=1e-4)

    def forward(self,in_img,softmax=True,train=True):
        in_data = np.pad(in_img, self.padding, mode='constant')
        in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape).transpose(0, 3, 1, 2)
        in_data = torch.from_numpy(in_data).to(self.device)
        #pading image for crop
        img_unprocessed = np.pad(in_img, self.padding_2, mode='constant')
        input_data = self.preprocess(img_unprocessed)
        in_shape_2 = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape_2).transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_data).to(self.device)
        angle_index = None
        if not train:
            self.model.eval()
            with torch.no_grad():
                _,logits = self.model(in_data)
        else:
            _, logits = self.model(in_data)

        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        logits = logits.tensor
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        output = logits.reshape(1,-1)
        
        if softmax:
            output = F.softmax(output,dim=-1)
            output = output.reshape(logits.shape[2:]).cpu().detach().numpy()
            output = output[...,np.newaxis]
        #get the pick_angele if not train
        if not train:
            argmax = np.argmax(output)
            argmax = np.unravel_index(argmax, shape=output.shape)
            p = argmax[:2]
            crop = input_tensor[:,:,p[0]:(p[0] + self.crop_size),p[1]:(p[1] + self.crop_size)]
            #print('crop',crop.size())
            self.angle_model.eval()
            with torch.no_grad():
              angle_index = self.angle_model(crop)
              angle_index = angle_index.tensor.reshape(1,-1)
              angle_index = angle_index.detach().cpu().numpy()
            #print('max angle',angle_index.shape, np.argmax(angle_index.shape))
        
        return output, angle_index, input_tensor

    def train(self,in_img,p,theta,backprop=True):
        self.model.train()
        self.angle_model.train()
        
        output,_,input_tensor = self.forward(in_img,softmax=False)
        crop = input_tensor[:,:,p[0]:(p[0] + self.crop_size),p[1]:(p[1] + self.crop_size)]
        #print('crop',crop.size())
        angle_index = self.angle_model(crop)
        angle_index = angle_index.tensor.reshape(1,-1)
        #print('angle_index',angle_index.shape)
        # Get label
        theta = (theta + 2*np.pi)%(2*np.pi)
        if theta >= np.pi:
          theta = theta -np.pi
        # angle label
        # dgree interval: 10
        # theta_i is in range [0,17]
        # theta_i = theta / (2 * np.pi / self.n_rotations)
        # theta_i = np.int32(np.round(theta_i)) % (self.n_rotations/2)
        # label_theta = torch.as_tensor(theta_i,dtype=torch.long,device=self.device).unsqueeze(dim=0)
        if self.angle_label_type == 2:
          theta_i = theta / (2 * np.pi / self.n_rotations)
          theta_i = np.int32(np.round(theta_i) % (self.n_rotations/2))
          label_theta = smooth_label(theta_i,self.n_rotations//2,self.angle_label_smooth,
                                     device=self.device).unsqueeze(dim=0)
        elif self.angle_label_type == 0:
          theta_i = np.round(theta / (2 * np.pi) * 360.0)
          label_theta = get_smooth_label(angle_label=theta_i, 
                                         angle_range=180,
                                         label_type=self.angle_label_type,
                                         radius=self.angle_label_radius,
                                         omega=int(360/self.n_rotations),
                                         sig=self.angle_label_sigma,
                                         normalized=True)
          label_theta = torch.as_tensor(label_theta,dtype=torch.float32,
                                        device=self.device).unsqueeze(dim=0)
        # print('angle label', theta_i, label_theta)
        # location label
        if self.pos_label_type == 2: # pulse/one-hot label
          label_size = (1,) + in_img.shape[:2]  #(1, 320, 160)
          label = torch.zeros(label_size,dtype=torch.long,device=self.device)
          label[0, p[0], p[1],] = 1
          label = label.reshape(-1)
          label_i = torch.argmax(label).unsqueeze(dim=0)
          label = smooth_label(label_i, label.numel(), self.pos_label_smooth, 
                               device=self.device).unsqueeze(dim=0)
          # print('label_i', label_i)
          # print('pos label', label[0, label_i], label[0, label_i+1])
        elif self.pos_label_type == 0: # gaussian label
          label = get_gaussian_2d_label(in_img.shape[:2], p,
                                        radius=self.pos_label_radius, 
                                        sigma=self.pos_label_sigma, 
                                        normalized=True, device=self.device)
          label = label.reshape(1,-1)
        #print('label size',label.shape)
        #print('out size', output.shape)
        # Get loss
        loss1 = F.cross_entropy(input=output, target=label)
        loss2 = F.cross_entropy(input=angle_index,target=label_theta)

        # Backpropagation
        if backprop:
            self.optim1.zero_grad()
            loss1.backward()
            self.optim1.step()
            self.optim2.zero_grad()
            loss2.backward()
            self.optim2.step()
        return np.float32(loss1.item()),np.float32(loss2.item())

    def load(self,path1,path2):
        # safe operation for e2cnn
        self.model.eval()
        self.model.load_state_dict(torch.load(path1,map_location=self.device))
        
        self.angle_model.eval()
        self.angle_model.load_state_dict(torch.load(path2,map_location=self.device))


    def save(self,filename1,filename2):
        # safe operation for e2cnn
        self.model.eval()
        torch.save(self.model.state_dict(), filename1)
        self.angle_model.eval()
        torch.save(self.angle_model.state_dict(), filename2)

    def clear(self):
        self.model = None
        self.angle_model = None
