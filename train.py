import datetime
import os
import numpy as np
import yaml
import random
from easydict import EasyDict as edict
# import sys
# sys.path.insert(0,'..')
from raven.dataset import Dataset
import argparse
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn

from networks.non_equi_transporter import TransporterAgent as non_equi_agent
# from networks.femi_transporter import TransporterAgent as femi_agent
# from networks.semi_transporter import TransporterAgent as semi_agent
# from networks.equivariant_transporter_tail import TransporterAgent as equ_agent_tail
from networks.equivariant_transporter import TransporterAgent as equ_agent
# from networks.gr_non_equi_transporter import TransporterAgent as gr_agent
# from networks.gr_equi_transporter import TransporterAgent as gr_equ_agent
from networks.so2_equivariant_transporter import TransporterAgent as so2_equ_agent
from networks.so2_align_transporter import TransporterAgent as so2_align_agent
from networks.mix_equ_transporter import TransporterAgent as mix_equ_agent
from networks.mix_align_transporter import TransporterAgent as mix_align_agent

# import faulthandler; faulthandler.enable()

parser = argparse.ArgumentParser(description='ravens')
parser.add_argument('--config_file', type=str, default='train.yaml')
# parser.add_argument('--train_dir', type=str, default='.')
# parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument('--task', type=str, default='block-insertion')
parser.add_argument('--n_demos', type=int, default=10)
parser.add_argument('--n_rotations', type=int, default=36)
parser.add_argument('--agent', type=str, default='so2-align')
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--n_align', type=int, default=12)
parser.add_argument('--n_feat', type=int, default=1)
parser.add_argument('--n_steps', type=int,default=10000) # the total train step n_steps/intervel = epoch
parser.add_argument('--interval', type=int,default=1000) # the training step for one epoch interval/n_demos = the number of resued data
# parser.add_argument('--n_runs', type=int,default=1)# not important
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--logging', action='store_true', default=False)

# parser.add_argument('--lite', action='store_true', default=False)
# parser.add_argument('--load', type=int, default=0)
# parser.add_argument('--angle_lite', action='store_true', default=False)
# parser.add_argument('--equ', action='store_true', default=False)
# parser.add_argument('--femi', action='store_true', default=False)
# parser.add_argument('--semi', action='store_true', default=False)
# parser.add_argument('--non', action='store_true', default=False)
# parser.add_argument('--tail', action='store_true', default=False)
# parser.add_argument('--init', action='store_true', default=False)
# parser.add_argument('--grconv', action='store_true', default=False)
# parser.add_argument('--equ_grconv', action='store_true', default=False)
# parser.add_argument('--equ_so2', action='store_true', default=False)

# parser.add_argument('--off_logger', action='store_true', default=False)
cmd_args = parser.parse_args()

def main():
    # load arguments
    # parser = argparse.ArgumentParser(description='ravens')
    # cmd_args = parser.parse_args()
    config_name = cmd_args.config_file

    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    args = edict(config_data)

    train_dataset = Dataset(os.path.join(args.data_dir, f'{cmd_args.task}-train'))
    print(os.path.join(args.data_dir, f'{cmd_args.task}-train'))
    (obs, act, _, _), _ = train_dataset.sample()
    #test_dataset = Dataset(os.path.join(args.data_dir, f'{cmd_args.task}-test'))
    for train_run in range(args.n_runs):
    #for train_run in range(1):
        #train_run = train_run+1
        name = f'{cmd_args.task}-{cmd_args.n_rotations}-{cmd_args.n_demos}-{cmd_args.seed}'
        
        writer = None
        #set tensorborad logger
        if cmd_args.logging:
            curr_time = datetime.datetime.now().strftime('%Y%m%d')
            log_dir = os.path.join(args.log_dir, 'logs', name,
                                   curr_time+'-{}{}'.format(cmd_args.agent, cmd_args.postfix))
            
            writer = tf.summary.create_file_writer(log_dir)

        # set seed
        # tf.random.set_seed(cmd_args.seed + 1)
        # random.seed(cmd_args.seed + 1)
        np.random.seed(cmd_args.seed + 1)
        torch.set_num_threads(1)
        torch.manual_seed(cmd_args.seed + 1)
        # torch.cuda.manual_seed(cmd_args.seed + 1)
        # torch.cuda.manual_seed_all(cmd_args.seed + 1)
        cudnn.benchmark = False
        cudnn.deterministic = True
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        # Limit random sampling during training to a fixed dataset.
        max_demos = train_dataset.n_episodes
        episodes = np.random.choice(range(max_demos), cmd_args.n_demos, False)
        train_dataset.set(episodes)
        print('use {} demos and train {} steps per epoch'.format(cmd_args.n_demos,cmd_args.interval))
        print('Sample index: ', episodes)
        # train agent and save snapshot
        if cmd_args.agent == 'equ':
            print('equvairant agent')
            network_params = args.equ
            agent = equ_agent(name=name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                              n_rotations=cmd_args.n_rotations,load=args.load,network_params=network_params,
                              init=args.init,postfix=cmd_args.postfix)
        # if args.femi:
        #     print('femi_agent')
        #     agent = femi_agent(name=name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite,load=args.load, angle_lite = args.angle_lite,init = args.init)
        # if args.semi:
        #     print('semi_agent')
        #     agent = semi_agent(name=name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite,load=args.load,init = args.init)
        # if args.non:
        #     print('no equivariant agent')
        #     agent = non_equi_agent(name=name,task=cmd_args.task,root_dir=args.data_dir,load=args.load)
        # if args.tail:
        #     print('equvairant agent with tail network')
        #     agent = equ_agent_tail(name=name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite,load=args.load, angle_lite = args.angle_lite, init = args.init)
        # if args.grconv:
        #     print('non equivariant grconvnet agent')
        #     agent = gr_agent(name=name,task=cmd_args.task,root_dir=args.data_dir,load=args.load)
        # if args.equ_grconv:
        #     print('equvariant grconvnet agent')
        #     agent = gr_equ_agent(name=name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite,load=args.load, angle_lite = args.angle_lite,init = args.init)
        elif cmd_args.agent == 'so2':
            print('so(2) equivariant agent')
            network_params = args.so2
            # print(network_params)
            agent = so2_equ_agent(name=name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                  n_rotations=cmd_args.n_rotations,load=args.load,network_params=network_params,
                                  init=args.init,postfix=cmd_args.postfix)
        elif cmd_args.agent == 'so2-align':
            print('so(2)-align equivariant agent')
            network_params = args.so2
            network_params['transport']['n_ori_align'] = cmd_args.n_align
            network_params['transport']['n_dim_per_ori'] = cmd_args.n_feat
            agent = so2_align_agent(name=name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                  n_rotations=cmd_args.n_rotations,load=args.load,network_params=network_params,
                                  init=args.init,postfix=cmd_args.postfix)
        elif cmd_args.agent == 'non':
            print('non equivariant agent')
            # agent = non_equi_agent(name=name,task=cmd_args.task,root_dir=args.data_dir,load=args.load)
            agent = non_equi_agent(name=name,task=cmd_args.task,root_dir=args.checkpoint_dir, device=cmd_args.gpu_id,
                                   n_rotations=cmd_args.n_rotations,load=args.load,
                                   init=args.init,postfix=cmd_args.postfix)
        elif cmd_args.agent == 'mix':
            print('mix equivariant agent')
            network_params = args.mix
            agent = mix_equ_agent(name=name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                  n_rotations=cmd_args.n_rotations,load=args.load,network_params=network_params,
                                  init=args.init,postfix=cmd_args.postfix)
        elif cmd_args.agent == 'mix-align':
            print('mix align agent')
            network_params = args.mix_align
            network_params['transport']['n_ori_align'] = cmd_args.n_align
            network_params['transport']['n_dim_per_ori'] = cmd_args.n_feat
            agent = mix_align_agent(name=name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                  n_rotations=cmd_args.n_rotations,load=args.load,network_params=network_params,
                                  init=args.init,postfix=cmd_args.postfix)
            
        else:
            raise Exception('Invalid model type')
        while agent.total_steps<cmd_args.n_steps:            
            agent.train(train_dataset,writer)
            if agent.total_steps % cmd_args.interval == 0:
                agent.save()
        
        # clear the model
        agent.clear()

if __name__=="__main__":
    main()