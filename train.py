import datetime
import os
import numpy as np
import yaml
from easydict import EasyDict as edict
# import sys
# sys.path.insert(0,'..')
from raven.dataset import Dataset
import argparse
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn

from networks.non_equi_transporter import TransporterAgent as non_equi_agent
from networks.femi_transporter import TransporterAgent as femi_agent
from networks.semi_transporter import TransporterAgent as semi_agent
from networks.equivariant_transporter_tail import TransporterAgent as equ_agent_tail
from networks.equivariant_transporter import TransporterAgent as equ_agent
from networks.gr_non_equi_transporter import TransporterAgent as gr_agent
from networks.gr_equi_transporter import TransporterAgent as gr_equ_agent
from networks.so2_equivariant_transporter import TransporterAgent as so2_equ_agent

# import faulthandler; faulthandler.enable()

# parser = argparse.ArgumentParser(description='ravens')
# parser.add_argument('--train_dir', type=str, default='.')
# parser.add_argument('--data_dir', type=str, default='.')
# parser.add_argument('--task', type=str, default='block-insertion')
# parser.add_argument('--n_demos', type=int,default=10)
# parser.add_argument('--n_steps', type=int,default=10000) # the total train step n_steps/intervel = epoch
# parser.add_argument('--interval', type=int,default=1000) # the training step for one epoch interval/n_demos = the number of resued data
# parser.add_argument('--n_runs', type=int,default=1)# not important
# parser.add_argument('--gpu', type=int, default=1)
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
# args = parser.parse_args()

def main():
    # load arguments
    parser = argparse.ArgumentParser(description='ravens')
    parser.add_argument('--config_file', type=str, required=True, help='the name of configuration file')
    arguments = parser.parse_args()
    config_name = arguments.config_file

    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    args = edict(config_data)

    train_dataset = Dataset(os.path.join(args.data_dir, f'{args.task}-train'))
    print(os.path.join(args.data_dir, f'{args.task}-train'))
    (obs, act, _, _), _ = train_dataset.sample()
    #test_dataset = Dataset(os.path.join(args.data_dir, f'{args.task}-test'))
    for train_run in range(args.n_runs):
    #for train_run in range(1):
        #train_run = train_run+1
        name = f'{args.task}-{args.n_demos}-{train_run}'
        
        writer = None
        #set tensorborad logger
        if args.logging:
            curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join(args.train_dir, 'logs', args.task,
                                   curr_time, 'train')
            writer = tf.summary.create_file_writer(log_dir)

        # set seed
        np.random.seed(train_run+1)
        torch.set_num_threads(train_run+1)
        torch.manual_seed(train_run+1)
        cudnn.benchmark = False
        cudnn.deterministic = True

        # Limit random sampling during training to a fixed dataset.
        max_demos = train_dataset.n_episodes
        episodes = np.random.choice(range(max_demos), args.n_demos, False)
        train_dataset.set(episodes)
        print('use {} demos and train {} steps per epoch'.format(args.n_demos,args.interval))
        # train agent and save snapshot
        if args.model == 'equ':
            print('equvairant agent')
            agent = equ_agent(name=name,task=args.task,root_dir=args.data_dir,device=args.gpu_id,
                              lite=args.lite,load=args.load, angle_lite=args.angle_lite,init=args.init,
                              postfix=args.data_postfix)
        # if args.femi:
        #     print('femi_agent')
        #     agent = femi_agent(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite,load=args.load, angle_lite = args.angle_lite,init = args.init)
        # if args.semi:
        #     print('semi_agent')
        #     agent = semi_agent(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite,load=args.load,init = args.init)
        # if args.non:
        #     print('no equivariant agent')
        #     agent = non_equi_agent(name=name,task=args.task,root_dir=args.data_dir,load=args.load)
        # if args.tail:
        #     print('equvairant agent with tail network')
        #     agent = equ_agent_tail(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite,load=args.load, angle_lite = args.angle_lite, init = args.init)
        # if args.grconv:
        #     print('non equivariant grconvnet agent')
        #     agent = gr_agent(name=name,task=args.task,root_dir=args.data_dir,load=args.load)
        # if args.equ_grconv:
        #     print('equvariant grconvnet agent')
        #     agent = gr_equ_agent(name=name,task=args.task,root_dir=args.data_dir,lite=args.lite,load=args.load, angle_lite = args.angle_lite,init = args.init)
        if args.model == 'so2':
            print('so(2) equvariant agent')
            irrep_kwargs = {'irrep': args.so2.resunet.irrep, 
                            'sample': args.so2.resunet.sample}
            agent = so2_equ_agent(name=name,task=args.task,root_dir=args.data_dir,device=args.gpu_id,
                                  lite=args.lite,load=args.load,angle_lite=args.angle_lite,init=args.init,
                                  postfix=args.data_postfix,**irrep_kwargs)
        while agent.total_steps<args.n_steps:
            for _ in range(args.interval):
                agent.train(train_dataset,writer)
            agent.save()

if __name__=="__main__":
    main()