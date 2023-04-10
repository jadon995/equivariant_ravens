import torch
from e2cnn import gspaces
import e2cnn.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class EquiResBlock(torch.nn.Module):
    def __init__(self, 
                 input_channels,
                 hidden_dim,
                 kernel_size,
                 N,
                 flip=False, 
                 quotient=False,
                 initialize=True):
        super(EquiResBlock, self).__init__()

        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)
        
        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr
        
        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])
        feat_type_out = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size-1) // 2, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2, initialize=initialize),
        )
        self.relu = nn.ReLU(feat_type_out, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2, initialize=initialize),
            )

    def forward(self, x_in):
        residual = x_in
        out = self.layer1(x_in)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)
        
        return out


class EquiGenerativeResNet(torch.nn.Module):
    def __init__(self, n_input_channel=1,
                       n_output_channel=32,
                       n_middle_channels=(32, 64, 128),
                       kernel_size=3,
                       N=6,
                       flip=False,
                       quotient=False,
                       initialize=False):
        super(EquiGenerativeResNet, self).__init__()
        self.N = N
        self.quotient = quotient        
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 3
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-1', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=9, stride=1, padding=4, initialize=initialize)),
            ('enc-e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), inplace=True))
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-2', nn.R2Conv(nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                      nn.FieldType(self.r2_act, self.l2_c * [self.repr]),
                                      kernel_size=4, stride=2, padding=1, initialize=initialize)),
            ('enc-e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), inplace=True))                                      
        ]))

        self.conv_down_3 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-3', nn.R2Conv(nn.FieldType(self.r2_act, self.l2_c * [self.repr]),
                                      nn.FieldType(self.r2_act, self.l3_c * [self.repr]),
                                      kernel_size=4, stride=2, padding=1, initialize=initialize)),
            ('enc-e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), inplace=True))
        ]))

        self.residual_layer = torch.nn.Sequential(OrderedDict([
            ('enc-e2res-1', EquiResBlock(self.l3_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
            ('enc-e2res-2', EquiResBlock(self.l3_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
            ('enc-e2res-3', EquiResBlock(self.l3_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
            ('enc-e2res-4', EquiResBlock(self.l3_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
            ('enc-e2res-5', EquiResBlock(self.l3_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2upsp-1', nn.R2Upsampling(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2))
            ('dec-e2conv-1', nn.R2ConvTransposed(nn.FieldType(self.r2_act, self.l3_c * [self.repr]),
                                                 nn.FieldType(self.r2_act, self.l2_c * [self.repr]),
                                                 kernel_size=4, stride=2, padding=1, output_padding=1, initialize=initialize)),
            ('dec-e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), inplace=True))                                                 
        ]))

        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2conv-2', nn.R2ConvTransposed(nn.FieldType(self.r2_act, self.l2_c * [self.repr]),
                                                 nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                                 kernel_size=4, stride=2, padding=2, output_padding=1, initialize=initialize)),
            ('dec-e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), inplace=True))                                                 
        ]))

        self.conv_up_3 = torch.nn.Sequential(OrderedDict([
            ('dec-e2conv-3', nn.R2ConvTransposed(nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                                 nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                                 kernel_size=9, stride=1, padding=4, initialize=initialize)),
            ('dec-e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), inplace=True))                                                 
        ]))

    def forward(self, x_in):
        x = nn.GeometricTensor(x_in, nn.FieldType(self.r2_act, x_in.shape[1] * [self.r2_act.trivial_repr]))
        x = self.conv_down_1(x)
        x = self.conv_down_2(x)
        x = self.conv_down_3(x)
        x = self.residual_layer(x)
        x = self.conv_up_1(x)
        x = self.conv_up_2(x)
        x = self.conv_up_3(x)
        return x
    
class equi_gr_res(torch.nn.Module):
    def __init__(self, in_dim,
                       out_dim,
                       N=6,
                       middle_dim=(32, 64, 128),
                       init=False):
        super(equi_gr_res, self).__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.main_block = EquiGenerativeResNet(n_input_channel=in_dim,
                                               n_output_channel=middle_dim[0],
                                               n_middle_channels=middle_dim,
                                               N=N,
                                               initialize=False)
        self.final = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, [self.r2_act.regular_repr]*middle_dim[0]),
                      nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim),
                      kernel_size=4, padding=1, stride=1, initialize=False))

        for name, module in self.named_modules():
            if isinstance(module, (nn.R2Conv, nn.R2ConvTransposed)):
                if init:
                    nn.init.generalized_he_init(module.weights.data, module.basisexpansion)
                    # nn.init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    pass
    
    def forward(self, x):
        out = self.main_block(x)
        out = self.final(out)
        return x, out