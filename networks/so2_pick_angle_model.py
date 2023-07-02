import torch
import escnn
from escnn import gspaces
import escnn.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from escnn_extension.fourier_gapool import FourierGroupAvgPool
from escnn_extension.inverse_fourier_transform import IFTPointwist
from escnn_extension.quotient_ift import QIFTPointwise

class SO2ResBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, N=-1, 
                 irreps=[(f,) for f in range(4)], num_of_samples=16, flip=False, quotient=False, initialize=True, last_act = True):
        super(SO2ResBlock, self).__init__()

        assert N == -1

        if flip:
            self.r2_act = gspaces.flipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=N)
        
        if quotient:
            self.rho = self.r2_act.fibergroup.spectral_quotient_representation(2, *irreps, name=None)
            activation = nn.QuotientFourierELU(self.r2_act, 2, output_channels, irreps, N=num_of_samples, inplace=True)
            last_activation = nn.QuotientFourierELU(self.r2_act, 2, output_channels, irreps, N=num_of_samples, inplace=True)
        else:
            self.rho = self.r2_act.fibergroup.spectral_regular_representation(*irreps, name=None)
            activation = nn.FourierELU(self.r2_act, output_channels, irreps=irreps, N=num_of_samples, inplace=True)
            last_activation = nn.FourierELU(self.r2_act, output_channels, irreps=irreps, N=num_of_samples, inplace=True)

        feat_type_in = nn.FieldType(self.r2_act, [self.rho] * input_channels)
        feat_type_out = nn.FieldType(self.r2_act, [self.rho] * output_channels)
        
        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False, initialize=initialize),
            activation,
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_out, feat_type_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False, initialize=initialize),
        )

        self.last_activation = last_activation if last_act else None

        self.upscale = None
        if input_channels != output_channels:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False, initialize=initialize),
            )
    
    def forward(self, x_in):
        residual = x_in
        out = self.layer1(x_in)
        out = self.layer2(out)

        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        
        if self.last_activation:
            out = self.last_activation(out)

        return out

class SO2ResNet(torch.nn.Module):
    def __init__(self, 
                 n_input_channel=6,
                 n_output_channel=1,
                 n_middle_channels=(16, 32, 16, 8, 4),
                 kernel_size=7,
                 N=36, 
                 flip=False,
                 quotient=False,
                 initialize=False,
                 init=False,
                 **irrep_kwargs):
        super(SO2ResNet, self).__init__()
        max_irrep = irrep_kwargs['irrep']
        num_of_samples = irrep_kwargs['sample']

        N_out = N # discrete group for output
        N = -1 # infinite group

        if flip:
            self.r2_act = gspaces.flipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=N)
        
        irreps = [(f,) for f in range(max_irrep+1)]
        self.irreps = irreps

        if quotient:
            self.rho = self.r2_act.fibergroup.spectral_quotient_representation(2, *(self.irreps), name=None)
        else:
            self.rho = self.r2_act.fibergroup.spectral_regular_representation(*(self.irreps), name=None)

        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]
        self.l5_c = n_middle_channels[4]

        activation_e0 = nn.QuotientFourierELU(self.r2_act, 2, self.l1_c, irreps=self.irreps, N=num_of_samples, inplace=True) if quotient \
              else nn.FourierELU(self.r2_act, self.l1_c, irreps=self.irreps, N=num_of_samples, inplace=True)
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*n_input_channel),
                                       nn.FieldType(self.r2_act, [self.rho]*self.l1_c),
                                       kernel_size=kernel_size,padding=3,bias=False,initialize=initialize)),
            ('enc-e2elu-0', activation_e0),
            ('enc-e2res-1', SO2ResBlock(self.l1_c,self.l1_c,kernel_size=kernel_size,N=N,irreps=self.irreps,num_of_samples=num_of_samples,flip=flip,quotient=quotient,initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act,[self.rho]*self.l1_c),sigma=0.66, stride=2)),
            ('enc-e2res-2', SO2ResBlock(self.l1_c,self.l2_c,kernel_size=kernel_size,N=N,irreps=self.irreps,num_of_samples=num_of_samples,flip=flip,quotient=quotient,initialize=initialize)),
        ]))

        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act,[self.rho]*self.l2_c),sigma=0.66, stride=2)),
            ('enc-e2res-3', SO2ResBlock(self.l2_c,self.l3_c,kernel_size=kernel_size,N=N,irreps=self.irreps,num_of_samples=num_of_samples,flip=flip,quotient=quotient,initialize=initialize)),
        ]))

        # # 16x16x18IR -> 10x10x8IR
        # self.final_0 = torch.nn.Sequential(OrderedDict([
        #     ('enc-final-0', nn.R2Conv(nn.FieldType(self.r2_act, [self.rho]*self.l3_c),
        #                               nn.FieldType(self.r2_act, [self.rho]*self.l4_c),
        #                               kernel_size=kernel_size,padding=0,bias=False,initialize=initialize)),
        #     ('enc-f_elu-0', nn.FourierELU(self.r2_act, self.l4_c, irreps=self.irreps, N=num_of_samples, inplace=True)),
        # ]))

        # # 10x10x8IR -> 10x10x8R -> 4x4x1Q
        # self.r2_act_out = gspaces.rot2dOnR2(N=N_out)
        # self.repr = self.r2_act_out.regular_repr
        # self.final_1 = torch.nn.Sequential(OrderedDict([
        #     ('discrete_map', IFTPointwist(self.r2_act,self.r2_act_out,self.l4_c,irreps=self.irreps,N=self.r2_act_out.regular_repr.size)),
        #     ('enc-final-1', nn.R2Conv(nn.FieldType(self.r2_act_out, [self.r2_act_out.regular_repr]*self.l4_c),
        #                               nn.FieldType(self.r2_act_out, [self.r2_act_out.quotient_repr(2)]*n_output_channel),
        #                               kernel_size=kernel_size,padding=0,bias=False,initialize=initialize))
        # ]))

        # 16x16x18IR -> 4x4x8IR
        # self.final_0 = torch.nn.Sequential(OrderedDict([
        #     ('enc-final-0', nn.R2Conv(nn.FieldType(self.r2_act, [self.rho]*self.l3_c),
        #                               nn.FieldType(self.r2_act, [self.rho]*self.l4_c),
        #                               kernel_size=kernel_size,padding=0,bias=False,initialize=initialize)),
        #     ('enc-f_elu-0', nn.FourierELU(self.r2_act, self.l4_c, irreps=self.irreps, N=num_of_samples, inplace=True)),
        #     ('enc-final-0-1', nn.R2Conv(nn.FieldType(self.r2_act, [self.rho]*self.l4_c),
        #                               nn.FieldType(self.r2_act, [self.rho]*self.l5_c),
        #                               kernel_size=kernel_size,padding=0,bias=False,initialize=initialize)),
        #     ('enc-f_elu-0-1', nn.FourierELU(self.r2_act, self.l5_c, irreps=self.irreps, N=num_of_samples, inplace=True)),
        # ]))

        # # 4x4x8IR -> 4x4x8R -> 2x2x1Q
        # self.r2_act_out = gspaces.rot2dOnR2(N=N_out)
        # self.repr = self.r2_act_out.regular_repr
        # self.final_1 = torch.nn.Sequential(OrderedDict([
        #     ('discrete_map', IFTPointwist(self.r2_act,self.r2_act_out,self.l5_c,irreps=self.irreps,N=self.r2_act_out.regular_repr.size)),
        #     ('enc-final-1', nn.R2Conv(nn.FieldType(self.r2_act_out, [self.r2_act_out.regular_repr]*self.l5_c),
        #                               nn.FieldType(self.r2_act_out, [self.r2_act_out.quotient_repr(2)]*n_output_channel),
        #                               kernel_size=3,padding=0,bias=False,initialize=initialize))
        # ]))
        # self.pool = nn.PointwiseAvgPool(nn.FieldType(self.r2_act_out, [self.r2_act_out.quotient_repr(2)]*n_output_channel),
                                        # kernel_size=2,stride=1,padding=0)

        # 16X16X16IR -> 4X4X1IQ
        activation_f0 = nn.QuotientFourierELU(self.r2_act, 2, self.l4_c, irreps=self.irreps, N=num_of_samples, inplace=True) if quotient \
              else nn.FourierELU(self.r2_act, self.l4_c, irreps=self.irreps, N=num_of_samples, inplace=True)
        self.rho_last= self.r2_act.fibergroup.spectral_quotient_representation(2, *(self.irreps), name=None)
        self.final_0 = torch.nn.Sequential(OrderedDict([
            ('enc-final-0', nn.R2Conv(nn.FieldType(self.r2_act, [self.rho]*self.l3_c),
                                      nn.FieldType(self.r2_act, [self.rho]*self.l4_c),
                                      kernel_size=kernel_size,padding=0,bias=False,initialize=initialize)),
            ('enc-f_elu-0', activation_f0),
            ('enc-final-0-1', nn.R2Conv(nn.FieldType(self.r2_act, [self.rho]*self.l4_c),
                                        nn.FieldType(self.r2_act, [self.rho_last]*n_output_channel),
                                        kernel_size=kernel_size,padding=0,bias=False,initialize=initialize)),         
        ]))

        # 4X4X1IQ -> 1X1X1Q
        self.r2_act_out = gspaces.rot2dOnR2(N=N_out)
        self.final_1 = torch.nn.Sequential(OrderedDict([
                ('discrete_map', QIFTPointwise(self.r2_act,self.r2_act_out,2,n_output_channel,irreps=self.irreps,N=self.r2_act_out.regular_repr.size)),
        ]))
        self.pool = nn.PointwiseAvgPool(nn.FieldType(self.r2_act_out, [self.r2_act_out.quotient_repr(2)]*n_output_channel),
                                        kernel_size=4,stride=1,padding=0)
        
        # initilize weights 
        for name, module in self.named_modules():
            if isinstance(module, nn.R2Conv):
                if init:
                    # print(name)
                    #nn.init.generalized_he_init(module.weights.data, module.basisexpansion)
                    nn.init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    pass
        
    def forward(self,obs):

        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map = self.conv_down_1(obs_gt)
        #print(feature_map.shape)
        feature_map = self.conv_down_2(feature_map)
        #print(feature_map.shape)
        feature_map = self.conv_down_4(feature_map)
        #print(feature_map.shape)
        feature_map = self.final_0(feature_map)
        #print(feature_map.shape)
        feature_map = self.final_1(feature_map)
        #print(feature_map.shape)
        feature_map = self.pool(feature_map)
        # print(feature_map.shape)
        # print('so2 pick angle network')

        return feature_map
