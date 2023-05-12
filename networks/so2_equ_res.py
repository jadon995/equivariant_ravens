import torch
from escnn import gspaces
import escnn.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from escnn_extension.fourier_gapool import FourierGroupAvgPool

class SO2ResBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, N, 
                 irreps=[(f,) for f in range(4)], flip=False, quotient=False, initialize=True, last_act = True):
        super(SO2ResBlock, self).__init__()

        if flip:
            self.r2_act = gspaces.flipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=N)

        # if quotient:
        #     if flip:
        #         rep = r2_act.quotient_repr((None, 2))
        #     else:
        #         rep = r2_act.quotient_repr(2)
        # else:
        #     rep = r2_act.regular_repr
        
        self.rho = self.r2_act.fibergroup.spectral_regular_representation(*irreps, name=None)
        feat_type_in = nn.FieldType(self.r2_act, [self.rho] * input_channels)
        feat_type_out = nn.FieldType(self.r2_act, [self.rho] * output_channels)

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False, initialize=initialize),
            nn.FourierELU(self.r2_act, output_channels, irreps=irreps, N=16, inplace=True),
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_out, feat_type_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False, initialize=initialize),
        )

        self.last_act =last_act
        if self.last_act:
            self.last_activation = nn.FourierELU(self.r2_act, output_channels, irreps=irreps, N=16, inplace=True)

        self.upscale = None
        if input_channels != output_channels:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False, initialize=initialize),
                # nn.R2Conv(feat_type_in, feat_type_out, kernel_size=1, padding=0, bias=False, initialize=initialize),
            )
    
    def forward(self, x_in):
        residual = x_in
        out = self.layer1(x_in)
        out = self.layer2(out)

        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        
        if self.last_act:
            out = self.last_activation(out)

        return out

class SO2ResUnet(torch.nn.Module):
    def __init__(self, 
                 n_input_channel=1, 
                 n_output_channel=16, 
                 n_middle_channels=(16, 32, 64, 128), 
                 kernel_size=3, 
                 N=-1, 
                 flip=False, 
                 quotient=False, 
                 initialize=False):
        super(SO2ResUnet, self).__init__()
        if flip:
            self.r2_act = gspaces.flipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=N)
        
        '''
        # if quotient:
        #     if flip:
        #         self.repr = self.r2_act.quotient_repr((None, 2))
        #     else:
        #         self.repr = self.r2_act.quotient_repr(2)
        # else:
        #     self.repr = self.r2_act.regular_repr
        '''

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.irreps = [(f,) for f in range(5)]
        self.rho = self.r2_act.fibergroup.spectral_regular_representation(*(self.irreps), name=None)
        
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.rho]),
                                       kernel_size=3, padding=1, bias=False, initialize=initialize)),
            ('enc-e2relu-0', nn.FourierELU(self.r2_act, self.l1_c, irreps=self.irreps, N=16, inplace=True)),
            ('enc-e2res-1', SO2ResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),        
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            # ('enc-pool-2', nn.PointwiseMaxPoolAntialiased(nn.FieldType(self.r2_act, self.l1_c * [self.rho]), kernel_size=3, sigma=0.66, stride=2)),
            ('enc-pool-2', nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, self.l1_c * [self.rho]), sigma=0.66, stride=2)),
            ('enc-e2res-2', SO2ResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, self.l2_c * [self.rho]), sigma=0.66, stride=2)),
            ('enc-e2res-3', SO2ResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, self.l3_c * [self.rho]), sigma=0.66, stride=2)),
            ('enc-e2res-4', SO2ResBlock(self.l3_c, self.l4_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, self.l4_c * [self.rho]), sigma=0.66, stride=2)),
            ('enc-e2res-5', SO2ResBlock(self.l4_c, self.l4_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1', SO2ResBlock(2*self.l4_c, self.l3_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', SO2ResBlock(2*self.l3_c, self.l2_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', SO2ResBlock(2*self.l2_c, self.l1_c, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', SO2ResBlock(2*self.l1_c, n_output_channel, kernel_size=kernel_size, N=N, irreps=self.irreps, flip=flip, quotient=quotient, initialize=initialize, last_act=False)),
        ]))
        
        self.upsample_16_8 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l4_c * [self.rho]), 2)
        self.upsample_8_4 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l3_c * [self.rho]), 2)
        self.upsample_4_2 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l2_c * [self.rho]), 2)
        self.upsample_2_1 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l1_c * [self.rho]), 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16
    
    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        # concat_8 = torch.cat((feature_map_8.tensor, self.upsample_16_8(feature_map_16).tensor), dim=1)
        # concat_8 = nn.GeometricTensor(concat_8, nn.FieldType(self.r2_act, 2*self.l4_c * [self.rho]))
        concat_8 = nn.tensor_directsum([feature_map_8, self.upsample_16_8(feature_map_16)])
        feature_map_up_8 = self.conv_up_8(concat_8)

        # concat_4 = torch.cat((feature_map_4.tensor, self.upsample_8_4(feature_map_up_8).tensor), dim=1)
        # concat_4 = nn.GeometricTensor(concat_4, nn.FieldType(self.r2_act, 2*self.l3_c * [self.rho]))
        concat_4 = nn.tensor_directsum([feature_map_4, self.upsample_8_4(feature_map_up_8)])
        feature_map_up_4 = self.conv_up_4(concat_4)

        # concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_up_4).tensor), dim=1)
        # concat_2 = nn.GeometricTensor(concat_2, nn.FieldType(self.r2_act, 2*self.l2_c * [self.rho]))
        concat_2 = nn.tensor_directsum([feature_map_2, self.upsample_4_2(feature_map_up_4)])
        feature_map_up_2 = self.conv_up_2(concat_2)

        # concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        # concat_1 = nn.GeometricTensor(concat_1, nn.FieldType(self.r2_act, 2*self.l1_c * [self.rho]))
        concat_1 = nn.tensor_directsum([feature_map_1, self.upsample_2_1(feature_map_up_2)])
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)
    
class so2_res(torch.nn.Module):
    def __init__(self,in_dim,out_dim,N=-1,middle_dim=(32, 64, 128, 256),init=False):
        super(so2_res, self).__init__()
        N = -1
        self.r2_act = gspaces.rot2dOnR2(N=N)
        self.main_block = SO2ResUnet(n_input_channel=in_dim,
                                     n_output_channel=middle_dim[0],
                                     n_middle_channels=middle_dim,
                                     N=N)
        irreps = [(f,) for f in range(5)]
        rho = self.r2_act.fibergroup.spectral_regular_representation(*irreps, name=None)

        ## Trial One: 16IR -> 1T
        # self.final = torch.nn.Sequential(
            # nn.R2Conv(nn.FieldType(self.r2_act, [rho]*middle_dim[0]),
                        # nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim),
                        # kernel_size=3, padding=1, bias=False, initialize=False))
        
        ## Trial Two: 16IR -> 1IR -> 1T
        # self.final = torch.nn.Sequential(
        #     nn.R2Conv(nn.FieldType(self.r2_act, [rho]*middle_dim[0]),
        #               nn.FieldType(self.r2_act, [rho]*out_dim),
        #               kernel_size=3, padding=1, bias=False, initialize=False),
        #     nn.FourierELU(self.r2_act, out_dim, irreps=irreps, N=16, inplace=True),
        #     nn.R2Conv(nn.FieldType(self.r2_act, [rho]*out_dim),
        #               nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim), 
        #               kernel_size=1, padding=0, bias=False, initialize=False)
        # )

        # Trial Three: 16IR -> 16T -> 1T
        # self.final = torch.nn.Sequential(
        #     nn.R2Conv(nn.FieldType(self.r2_act, [rho]*middle_dim[0]),
        #               nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*middle_dim[0]),
        #               kernel_size=3, padding=1, bias=False, initialize=False),
        #     nn.ELU(nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*middle_dim[0]), inplace=True),
        #     nn.R2Conv(nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*middle_dim[0]),
        #               nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim), 
        #               kernel_size=1, padding=0, bias=False, initialize=False)
        # )

        ## Trial Four: 16IR (Avg Pool)-> 16 -> 1
        # ftgpool = nn.FourierELU(self.r2_act, middle_dim[0], irreps=irreps, 
        #                         out_irreps=self.r2_act.fibergroup.bl_irreps(0), N=16)
        # self.invariant_map = nn.SequentialModule(
        #     nn.R2Conv(nn.FieldType(self.r2_act, [rho]*middle_dim[0]),
        #               ftgpool.in_type,
        #               kernel_size=3, padding=1, bias=False, initialize=False),
        #     ftgpool
        # )
        # self.fcn = torch.nn.Sequential(
        #     torch.nn.Conv2d(ftgpool.out_type.size, middle_dim[1], kernel_size=1, stride=1, padding=0),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(middle_dim[1], out_dim, kernel_size=1, stride=1, padding=0),
        #     # torch.nn.Conv2d(ftgpool.out_type.size, out_dim, kernel_size=1),
        # )
        # self.out_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim)

        ## Trial Five: 16IR --(AvgP)--> 16 --> 1
        ftgpool = nn.FourierELU(self.r2_act, middle_dim[0], irreps=irreps, 
                                out_irreps=self.r2_act.fibergroup.bl_irreps(0), N=16)
        # ftgpool = FourierGroupAvgPool(self.r2_act, middle_dim[0], irreps=irreps,
                                    #   out_irreps=self.r2_act.fibergroup.bl_irreps(0), N=16)
        self.invariant_map = nn.SequentialModule(ftgpool)
        self.out_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim)

        self.fcn = torch.nn.Sequential(
            # torch.nn.Conv2d(ftgpool.out_type.size, out_dim, kernel_size=3, stride=1,padding=1),
            torch.nn.Conv2d(ftgpool.out_type.size, middle_dim[0], kernel_size=3, stride=1,padding=1),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(middle_dim[0], out_dim, kernel_size=1, stride=1, padding=0),
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.R2Conv):
                if init:
                    print(name)
                    #nn.init.generalized_he_init(module.weights.data, module.basisexpansion)
                    nn.init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    pass
    
    def forward(self,x):
        # gconv -> gconv
        # out = self.main_block(x)
        # out = self.final(out)

        # gconv -> conv -> gconv
        out = self.main_block(x)
        out = self.invariant_map(out)
        out = out.tensor
        out = self.fcn(out)
        out = nn.GeometricTensor(out, self.out_type)
        return x,out
    
# model = so2_res(6,3,-1,(16, 32, 64, 128),True).to(device)

# model.eval()
# with torch.no_grad():
#     image_in = torch.rand(1, 6, 320, 320).to(device)
#     _, feat_out = model(image_in)
#     print(feat_out.shape)