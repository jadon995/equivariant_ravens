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
from task import Task, KitFourTools, KitSixTools, PlaceRedInGreen, StackBlockPyramid

import faulthandler; faulthandler.enable()

# load arguments
parser = argparse.ArgumentParser(description='ravens_test')
parser.add_argument('--config_file', type=str, default='train-robot.yaml')
# parser.add_argument('--root_dir', type=str, default='.')
# parser.add_argument('--data_dir', type=str, default='.')
# parser.add_argument('--assets_root', type=str, default='./raven/assets')
# parser.add_argument('--task', type=str, default='robot-kit-six-tools')
# parser.add_argument('--task', type=str, default='robot-place-red-in-green')
parser.add_argument('--task', type=str, default='robot-stack-block-pyramid')
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

    robot_env = RobotEnv()

    if cmd_args.task == 'robot-kit-four-tools':
        robot_task = KitFourTools()
    elif cmd_args.task == 'robot-kit-six-tools':
        robot_task = KitSixTools()
    elif cmd_args.task == 'robot-stack-block-pyramid':
        robot_task = StackBlockPyramid()
    elif cmd_args.task == 'robot-place-red-in-green':
        robot_task = PlaceRedInGreen()

    robot_task.mode = 'test'

    # Load test dataset.
    dataset = Dataset(os.path.join(args.data_dir, f'{cmd_args.task}-{robot_task.mode}{cmd_args.postfix}'))
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
        # visalize test
        for i in range(dataset.n_episodes):
            print(f"Test: {i + 1}/{dataset.n_episodes}")
            episode, seed = dataset.load(i)
            goal = episode[-1]
            for k in range(robot_task.max_steps):
                print("test visualize")
                obs = episode[k][0]
                act = episode[k][1]
                # act = agent.act(obs, info=None, goal=None)
                agent.test_visualize(obs, act)
        '''
        while dataset.n_episodes < cmd_args.n:
        # for i in range(n_test_runs):
            print(f"Robot test: {dataset.n_episodes + 1} / {cmd_args.n}")
            episode, total_reward = [], 0
            seed += 2

            print(f"===============Run-{dataset.n_episodes + 1}=================")
            print("Please reset environment.")
            robot_env.reset()
            while (input("Enter [y] to confirm setup:") != "y"):
                pass

            reward = robot_task.get_reward()

            for j in range(max_steps):
                print(f"-------Begin Step-{j+1}/{max_steps}-------")
                print("Capture and process image ...")
                # while (input("Enter [y] to capture image:") != "y"):
                    # pass
                obs = robot_env.get_obs()
                act = agent.act(obs, info=None, goal=None)
                # agent.test_visualize(obs, act)

                print("Pick start...")
                key_pick_pose = copy.deepcopy(act["pose0"])
                key_pick_pose[0][2] += 0.1
                robot_env.move_to_gripper_pose(key_pick_pose)
                robot_env.move_to_gripper_pose(act["pose0"])
                robot_env.gripper.close()
                robot_env.move_to_gripper_pose(key_pick_pose)

                print("Place start...")
                key_place_pose = copy.deepcopy(act["pose1"])
                key_place_pose[0][2] += 0.1
                robot_env.move_to_gripper_pose(key_place_pose)
                robot_env.move_to_gripper_pose(act["pose1"])
                robot_env.gripper.open()
                robot_env.move_to_gripper_pose(key_place_pose)

                episode.append((obs, act, reward, None))

                robot_env.reset()
            
            print("Capture last observation")
            # obs = robot_env.get_obs()
            episode.append((obs, None, reward, None))
            
            dataset.add(seed, episode)
            print('Conduct {} trials to collect {} successful {} demonstration'.format(dataset.n_episodes, cmd_args.task, cmd_args.n))

        '''
        # Save results.
        model_dir = '{}{}'.format(cmd_args.agent, cmd_args.postfix)
        if not os.path.exists(os.path.join(args.test_dir, name)):
            os.makedirs(os.path.join(args.test_dir, name))
        if not os.path.exists(os.path.join(args.test_dir, name, model_dir)):
            os.makedirs(os.path.join(args.test_dir, name, model_dir))
        with open(os.path.join(args.test_dir, name, model_dir, f'{test_step}.pkl'),'wb') as f:
            pickle.dump(results, f)

        # clear the model
        agent.clear()
        '''                

def load_agent(agent_name, task_name):
    if agent_name == 'equ':
        print('Cn equivariant agent')
        network_params = args.equ
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
