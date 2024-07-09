import sys
sys.path.append("./")
import datetime
import os
import yaml
from easydict import EasyDict as edict
import numpy as np
#from equ_transporter import TransporterAgent as equ_agent
# from raven.dataset import Dataset
import argparse
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
# from raven.gripper_env import Environment
#from ravens import tasks
import pickle
import copy

from dataset import Dataset
'''
from raven.gripper_block_insertion import BlockInsertion
from raven.gripper_place_red_in_green import PlaceRedInGreen
from raven.gripper_align_box_corner import AlignBoxCorner
from raven.gripper_stack_block_pyramid import StackBlockPyramid
from raven.gripper_palletizing_boxes import PalletizingBoxes
from raven.gripper_packing_boxes import PackingBoxes
from raven.gripper_assembling_kits import AssemblingKits, AssemblingKitsTool, AssemblingKits3DTool, AssemblingKitsScrewDriver, AssemblingKits3DToolKit
from raven.gripper_assembling_toolkit import AssemblingToolKit
'''
from networks.equivariant_transporter import TransporterAgent as equ_agent
from networks.non_equi_transporter import TransporterAgent as non_equi_agent
# from networks.femi_transporter import TransporterAgent as femi_agent
# from networks.semi_transporter import TransporterAgent as semi_agent
# from networks.equivariant_transporter_tail import TransporterAgent as equ_agent_tail
# from networks.gr_non_equi_transporter import TransporterAgent as gr_agent
# from networks.gr_equi_transporter import TransporterAgent as gr_equ_agent
from networks.so2_equivariant_transporter import TransporterAgent as so2_equ_agent
from networks.so2_align_transporter import TransporterAgent as so2_align_agent
from networks.mix_equ_transporter import TransporterAgent as mix_equ_agent

from robot_so2_align_transporter import TransporterAgent as robot_align_agent

from dataset import Dataset
from environment import Environment as RobotEnv
from task import Task, KitFourTools, KitSixTools

import faulthandler; faulthandler.enable()

# load arguments
parser = argparse.ArgumentParser(description='ravens_test')
parser.add_argument('--config_file', type=str, default='train-robot.yaml')
# parser.add_argument('--root_dir', type=str, default='.')
# parser.add_argument('--data_dir', type=str, default='.')
# parser.add_argument('--assets_root', type=str, default='./raven/assets')
# parser.add_argument('--task', type=str, default='robot-kit-handtools')
parser.add_argument('--task', type=str, default='robot-kit-six-tools')
parser.add_argument('--n_demos', type=int,default=10)# the demo used for testing
parser.add_argument('--n_rotations', type=int, default=180)
parser.add_argument('--agent', type=str, default='robot-align')
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--n_align', type=int, default=12)
parser.add_argument('--n_feat', type=int, default=1)
parser.add_argument('--n_steps', type=int,default=10000)# the testing steps

# parser.add_argument('--n_runs', type=int,default=1)
# parser.add_argument('--interval', type=int,default=1000)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--entire', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=25)
# parser.add_argument('--shared_memory', action='store_true', default=False)
# parser.add_argument('--equ', action='store_true', default=False)
# parser.add_argument('--lite', action='store_true', default=False)
# parser.add_argument('--angle_lite', action='store_true', default=False)
# parser.add_argument('--continuous', action='store_true', default=False)

# parser.add_argument('--femi', action='store_true', default=False)
# parser.add_argument('--semi', action='store_true', default=False)
# parser.add_argument('--non', action='store_true', default=False)
# parser.add_argument('--tail', action='store_true', default=False)
# parser.add_argument('--grconv', action='store_true', default=False)
# parser.add_argument('--equ_grconv', action='store_true', default=False)
# parser.add_argument('--equ_so2', action='store_true', default=False)
cmd_args = parser.parse_args()

with open(os.path.join('configs', cmd_args.config_file), 'r') as file:
    config_data = yaml.load(file, Loader=yaml.FullLoader)
args = edict(config_data)

def main():
    '''
    # Initialize environment and task.
    env = Environment(
        args.assets_root,
        disp=cmd_args.disp,
        shared_memory=args.shared_memory,
        hz=480)
    # TODO FOR NOW JUST TEST on block-insertion
    if cmd_args.task == 'block-insertion':
        task = BlockInsertion(continuous=args.continuous)
    elif cmd_args.task == 'place-red-in-green':
        task = PlaceRedInGreen(continuous=args.continuous)
    elif cmd_args.task == 'align-box-corner':
        task = AlignBoxCorner(continuous=args.continuous)
    elif cmd_args.task == 'stack-block-pyramid':
        task = StackBlockPyramid(continuous=args.continuous)
    elif cmd_args.task == 'palletizing-boxes':
        task = PalletizingBoxes(continuous=args.continuous)
    elif cmd_args.task == 'packing-boxes':
        task = PackingBoxes(continuous=args.continuous)
    elif cmd_args.task == 'assembling-kits':
        task = AssemblingKits(continuous=args.continuous)
    elif cmd_args.task == 'assembling-kits-tool':
        task = AssemblingKitsTool(continuous=args.continuous)
    elif cmd_args.task == 'assembling-kits-screwdriver':
        task = AssemblingKitsScrewDriver(continuous=args.continuous)
    elif cmd_args.task == 'assembling-kits-3dtool':
        task = AssemblingKits3DTool(continuous=args.continuous)
    elif cmd_args.task == 'assembling-kits-3dtoolkit':
        task = AssemblingKits3DToolKit(continuous=args.continuous)
    elif cmd_args.task == 'assembling-single-toolkit':
        task = AssemblingToolKit(continuous=args.continuous)    
    else:
        raise RuntimeError('gripper version no {}'.format(cmd_args.task))
    task.mode = 'test'
    '''

    # robot_env = RobotEnv()
    robot_task = Task()
    robot_task = KitSixTools()

    # robot_task.mode = 'test'
    robot_task.mode = 'train'

    # Load test dataset.
    dataset = Dataset(os.path.join(args.data_dir, f'{cmd_args.task}-{robot_task.mode}'))
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (robot_task.mode == 'test') else -2

    # Determine max steps per episode.
    max_steps = robot_task.max_steps

    #train_run +=1
    name = f'{cmd_args.task}-{cmd_args.n_rotations}-{cmd_args.n_demos}-{cmd_args.seed}'
    # set seed
    np.random.seed(cmd_args.seed + 1)
    # torch.set_num_threads(train_run + 1)
    torch.set_num_threads(1)
    torch.manual_seed(cmd_args.seed + 1)
    cudnn.benchmark = False
    cudnn.deterministic = True   
    # load agent
    # agent = load_agent(cmd_args.agent, name, args)

    if cmd_args.entire == True:
        n_steps = [10000,8000,6000,4000,2000]
    else:
        n_steps = [cmd_args.n_steps] 
    
    for test_step in n_steps:
        # load agent
        agent = None
        agent = load_agent(cmd_args.agent, name)
        agent.load(test_step)

        results = []
        #print(ds.n_episodes,'============')
        '''
        for i in range(ds.n_episodes):
            print(f'Test: {i + 1}/{ds.n_episodes}')
            episode, seed = ds.load(i)
            goal = episode[-1]
            total_reward = 0
            np.random.seed(seed)
            env.seed(seed)
            env.set_task(task)
            obs = env.reset()
            info = None
            reward = 0
            for k in range(task.max_steps):
                extend_secs = 0 if k<task.max_steps-1 else 2
                act = agent.act(obs, info, goal)
                #act = agent.act(obs, info)
                obs, reward, done, info = env.step(act, extend_secs=extend_secs)
                total_reward += reward
                print(f'Total Reward: {total_reward} Done: {done}')
                if done:
                    break
            results.append((total_reward, info))
        '''

        
        # compute mean and std
        num_samples = dataset.n_episodes * robot_task.max_steps
        print("total numble of steps", num_samples)

        color_mean, depth_mean = 0.0, 0.0
        for i in range(dataset.n_episodes):
            print(f"Test: {i + 1}/{dataset.n_episodes}")
            episode, seed = dataset.load(i)
            # goal = episode[-1]
            for k in range(robot_task.max_steps):
                # print("test visualize")
                obs = episode[k][0]
                # color = obs["color"]
                # depth = obs["depth"]
                # print("color mean", np.nanmean(color/255.0))
                # print("depth mean", np.nanmean(depth))

                im_input = agent.get_image(obs)
                color = im_input[:, :, :3]
                depth = im_input[:, :, 3]
                color_mean += np.mean(color/255.0)
                depth_mean += np.mean(depth)

                # print("color mean", color_mean)
                # print("depth mean", depth_mean)
        
        color_mean = color_mean / num_samples
        depth_mean = depth_mean / num_samples
        print("color mean:", color_mean)
        print("depth mean:", depth_mean)

        color_std, depth_std = 0.0, 0.0
        for i in range(dataset.n_episodes):
            print(f"Test: {i + 1}/{dataset.n_episodes}")
            episode, seed = dataset.load(i)
            for k in range(robot_task.max_steps):
                obs = episode[k][0]
                im_input = agent.get_image(obs)
                color = im_input[:, :, :3]
                depth = im_input[:, :, 3]
                color_std += ((color/255.0 - color_mean)**2).sum()/(color.shape[0] * color.shape[1])
                depth_std += ((depth - depth_mean)**2).sum()/(depth.shape[0] * depth.shape[1])

        color_std = np.sqrt(color_std/num_samples)
        depth_std = np.sqrt(depth_std/num_samples)
        print("color std:", color_std)
        print("depth std:", depth_std)

                # color = obs["color"][]
                # act = episode[k][1]
                # agent.test_visualize(obs, act)

        '''
        # visalize test
        for i in range(dataset.n_episodes):
            print(f"Test: {i + 1}/{dataset.n_episodes}")
            episode, seed = dataset.load(i)
            goal = episode[-1]
            for k in range(robot_task.max_steps):
                print("test visualize")
                obs = episode[k][0]
                act = episode[k][1]
                agent.test_visualize(obs, act)
        '''
      
                

def load_agent(agent_name, task_name):
    if agent_name == 'equ':
        print('Cn equivariant agent')
        network_params = args.equact
        agent = equ_agent(name=task_name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                            n_rotations=cmd_args.n_rotations,network_params=network_params,
                            postfix=cmd_args.postfix)
    # if args.femi:
    #     print('femi_agent')
    #     agent = femi_agent(name=task_name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite, angle_lite = args.angle_lite)
    # if args.semi:
    #     print('semi_agent')
    #     agent = semi_agent(name=task_name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite)
    elif agent_name == 'non':
    #     print('no equivariant agent')
        # agent = non_equi_agent(name=task_name,task=cmd_args.task,root_dir=args.data_dir)
        agent = non_equi_agent(name=task_name,task=cmd_args.task,root_dir=args.checkpoint_dir, device=cmd_args.gpu_id,
                                   n_rotations=cmd_args.n_rotations,postfix=cmd_args.postfix)
    # if args.tail:
    #     print('equvairant agent with tail network')
    #     agent = equ_agent_tail(name=task_name,task=cmd_args.task,root_dir=args.data_dir,lite=args.lite, angle_lite = args.angle_lite)
    elif agent_name == 'so2':
        print('so(2) equivariant agent')
        network_params = args.so2
        agent = so2_equ_agent(name=task_name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                n_rotations=cmd_args.n_rotations,network_params=network_params,postfix=cmd_args.postfix)
    elif agent_name == 'so2-align':
        print('so(2)-align equivariant agent')
        network_params = args.so2
        network_params['transport']['n_ori_align'] = cmd_args.n_align
        network_params['transport']['n_dim_per_ori'] = cmd_args.n_feat
        agent = so2_align_agent(name=task_name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                n_rotations=cmd_args.n_rotations,network_params=network_params,postfix=cmd_args.postfix)
    elif agent_name == 'robot-align':
        print('robot-align equivariant agent')
        network_params = args.so2
        network_params['transport']['n_ori_align'] = cmd_args.n_align
        network_params['transport']['n_dim_per_ori'] = cmd_args.n_feat
        agent = robot_align_agent(name=task_name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                n_rotations=cmd_args.n_rotations,network_params=network_params,postfix=cmd_args.postfix)
    elif agent_name == 'mix':
        print('mix equivariant agent')
        network_params = args.mix
        agent = mix_equ_agent(name=task_name,task=cmd_args.task,root_dir=args.checkpoint_dir,device=cmd_args.gpu_id,
                                n_rotations=cmd_args.n_rotations,network_params=network_params,postfix=cmd_args.postfix)
    return agent

if __name__=="__main__":
    main()
