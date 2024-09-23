import sys
sys.path.append("./")
import numpy as np
import argparse
# from raven.dataset import Dataset
# from raven.gripper_env import Environment
import os
import yaml
from easydict import EasyDict as edict
# from raven.gripper_block_insertion import BlockInsertion
# from raven.gripper_place_red_in_green import PlaceRedInGreen
# from raven.gripper_align_box_corner import AlignBoxCorner
# from raven.gripper_stack_block_pyramid import StackBlockPyramid
# from raven.gripper_palletizing_boxes import PalletizingBoxes
# from raven.gripper_packing_boxes import PackingBoxes
# from raven.gripper_assembling_kits import AssemblingKits, AssemblingKitsTool, AssemblingKitsScrewDriver, AssemblingKits3DTool, AssemblingKits3DToolKit
# from raven.gripper_assembling_toolkit import AssemblingToolKit
from robot_so2_align_transporter import TransporterAgent as robot_align_agent

from dataset import Dataset
from environment import Environment as RobotEnv
from task import KitFourTools, KitSixTools, StackBlockPyramid, PlaceRedInGreen, KitSurgicalTools

parser = argparse.ArgumentParser(description='ravens_demos')

parser.add_argument('--assets_root', type=str, default='./raven/assets/')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--n', type=int,default=20)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--continuous', action='store_true', default=False)
parser.add_argument('--steps_per_seg', type=int, default=3)
# parser.add_argument('--task', type=str, default='robot-kit-six-tools')
# parser.add_argument('--task', type=str, default='robot-place-red-in-green')
# parser.add_argument('--task', type=str, default='robot-stack-block-pyramid')
parser.add_argument('--task', type=str, default='robot-kit-surgical-tools')
parser.add_argument('--n_align', type=int, default=12)
parser.add_argument('--n_feat', type=int, default=1)
parser.add_argument('--config_file', type=str, default='train-robot.yaml')
parser.add_argument('--agent', type=str, default='robot-align')
parser.add_argument('--n_rotations', type=int, default=180)
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

with open(os.path.join('configs', args.config_file), 'r') as file:
    config_data = yaml.load(file, Loader=yaml.FullLoader)
config_args = edict(config_data)

def main():
    # Initialize environment and task.
    robot_env = RobotEnv()

    if args.task == 'robot-kit-four-tools':
        robot_task = KitFourTools()
    elif args.task == 'robot-kit-six-tools':
        robot_task = KitSixTools()
    elif args.task == 'robot-stack-block-pyramid':
        robot_task = StackBlockPyramid()
    elif args.task == 'robot-place-red-in-green':
        robot_task = PlaceRedInGreen()
    elif args.task == 'robot-kit-surgical-tools':
        robot_task = KitSurgicalTools()

    robot_task.mode = args.mode
    agent = None # human demonstrations   
    dataset = Dataset(os.path.join(args.data_dir, f'{args.task}-{robot_task.mode}'))

    # help the act viz
    network_params = config_args.so2
    network_params['transport']['n_ori_align'] = args.n_align
    network_params['transport']['n_dim_per_ori'] = args.n_feat
    agent = robot_align_agent(name="name", task=args.task, root_dir=config_args.checkpoint_dir, device=0,
                                n_rotations=180, network_params=network_params, postfix="")

    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (robot_task.mode == 'test') else -2

    # Determine max steps per episode.
    max_steps = robot_task.max_steps

    trial_n = 0
    while dataset.n_episodes < args.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{args.n}')
        trial_n += 1
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        
        # env.set_task(task)
        print(f"===============Trial-{trial_n}=================")
        print("Please reset environment.")
        robot_env.reset() # Move to initial conditions
        while (input("Enter [y] to confirm setup:") != "y"):
            pass

        info = None
        reward = robot_task.get_reward()
        for i in range(max_steps):
            print(f"-------Begin Demo-{i+1}/{max_steps}-------")
            while (input("Enter [y] to capture image:") != "y"):
                pass
            obs = robot_env.get_obs()
            act = robot_env.get_actions()
            episode.append((obs, act, reward, info))

            # Visulaize the record images and demonstration data
            agent.test_visualize(obs, act)
            
            total_reward += reward
            print(f'Total Reward: {total_reward}')
            input("Enter [y] to move home: ")         
            robot_env.reset()

        episode.append((obs, None, reward, info)) # last episode as a record
    
        if total_reward > 0.99 and input("Enter [y] to save demos: ") == 'y':
            dataset.add(seed, episode)
    print('conduct {} trials to collect {} successful {} demonstration'.format(trial_n, args.task, args.n))
    print('the planner successful rate is {}'.format(trial_n/args.n))

if __name__ == '__main__':
  main()