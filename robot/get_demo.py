import numpy as np
import argparse
# from raven.dataset import Dataset
# from raven.gripper_env import Environment
import os
# from raven.gripper_block_insertion import BlockInsertion
# from raven.gripper_place_red_in_green import PlaceRedInGreen
# from raven.gripper_align_box_corner import AlignBoxCorner
# from raven.gripper_stack_block_pyramid import StackBlockPyramid
# from raven.gripper_palletizing_boxes import PalletizingBoxes
# from raven.gripper_packing_boxes import PackingBoxes
# from raven.gripper_assembling_kits import AssemblingKits, AssemblingKitsTool, AssemblingKitsScrewDriver, AssemblingKits3DTool, AssemblingKits3DToolKit
# from raven.gripper_assembling_toolkit import AssemblingToolKit

from dataset import Dataset
from environment import Environment as RobotEnv
from task import Task, KitFourTools, KitSixTools

parser = argparse.ArgumentParser(description='ravens_demos')

parser.add_argument('--assets_root', type=str, default='./raven/assets/')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--n', type=int,default=10)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--continuous', action='store_true', default=False)
parser.add_argument('--steps_per_seg', type=int, default=3)
#parser.add_argument('--task', type=str, default='align-box-corner')
#parser.add_argument('--task', type=str, default='place-red-in-green')
parser.add_argument('--task', type=str, default='robot-kit-six-tools')
#parser.add_argument('--task', type=str, default='stack-block-pyramid')
#parser.add_argument('--task', type=str, default='palletizing-boxes')
#parser.add_argument('--task', type=str, default='packing-boxes')
#parser.add_argument('--task', type=str, default='assembling-kits')
args = parser.parse_args()

def main():
    '''
    enc_cls = Environment
    env = enc_cls(args.assets_root,
                  disp=args.disp,
                  shared_memory=False,
                  hz=480)
    if args.task == 'block-insertion':
        task = BlockInsertion(continuous=args.continuous)
    elif args.task == 'place-red-in-green':
        task = PlaceRedInGreen(continuous=args.continuous)
    elif args.task == 'align-box-corner':
        task = AlignBoxCorner(continuous=args.continuous)
    elif args.task == 'stack-block-pyramid':
        task = StackBlockPyramid(continuous=args.continuous)
    elif args.task == 'palletizing-boxes':
        task = PalletizingBoxes(continuous=args.continuous)
    elif args.task == 'packing-boxes':
        task = PackingBoxes(continuous=args.continuous)
    elif args.task == 'assembling-kits':
        task = AssemblingKits(continuous=args.continuous)
    elif args.task == 'assembling-kits-tool':
        task = AssemblingKitsTool(continuous=args.continuous)
    elif args.task == 'assembling-kits-screwdriver':
        task = AssemblingKitsScrewDriver(continuous=args.continuous)
    elif args.task == 'assembling-kits-3dtool':
        task = AssemblingKits3DTool(continuous=args.continuous)
    elif args.task == 'assembling-kits-3dtoolkit':
        task = AssemblingKits3DToolKit(continuous=args.continuous)
    elif args.task == 'assembling-single-toolkit':
        task = AssemblingToolKit(continuous=args.continuous)
    else:
        raise RuntimeError('gripper version no {}'.format(args.task))
'''
    robot_env = RobotEnv()

    if args.task == 'robot-kit-four-tools':
        robot_task = KitFourTools()
    elif args.task == 'robot-kit-six-tools':
        robot_task = KitSixTools()
    # robot_task = Task()
    robot_task.mode = args.mode
    agent = None # human demonstrations   
    dataset = Dataset(os.path.join(args.data_dir, f'{args.task}-{robot_task.mode}'))

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
        robot_env.reset()
        robot_env.start_camera_loop()
        while (input("Enter [y] to confirm setup:") != "y"):
            pass
        robot_env.end_camera_loop()

        info = None
        reward = robot_task.get_reward()
        
        for i in range(max_steps):
            print(f"-------Begin Demo-{i+1}/{max_steps}-------")
            while (input("Enter [y] to capture image:") != "y"):
                pass
            obs = robot_env.get_obs()

            act = robot_env.get_actions()
            print(act)

            episode.append((obs, act, reward, info))
            
            total_reward += reward
            print(f'Total Reward: {total_reward}')
            input("Enter [y] to move home: ")
            robot_env.reset()

        episode.append((obs, None, reward, info)) # last episode as a record
    
        if total_reward > 0.99 and input("Enter [y] to save demos: ") == 'y':
            dataset.add(seed, episode)
    print('conduct {} trials to collect {} successful {} demonstration'.format(trial_n, args.task, args.n))
    print('the planner successful rate is {}'.format(trial_n/args.n))


'''
    task.mode = args.mode
    agent = task.oracle(env,steps_per_seg=args.steps_per_seg)
    dataset = Dataset(os.path.join(args.data_dir,f'{args.task}-{task.mode}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == 'test') else -2

        # Determine max steps per episode.
    max_steps = task.max_steps
    if args.continuous:
        max_steps *= (args.steps_per_seg * agent.num_poses)

        # Collect training data from oracle demonstrations.
    trial_n = 0
    while dataset.n_episodes < args.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{args.n}')
        trial_n = trial_n+1
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        env.set_task(task)
        obs = env.reset()

        # wait for setup environment

        info = None
        reward = 0
        for i in range(max_steps):

            # wait for each human demonstration
            # record obs
            # record action
            # save episode

            # wait more time for objects to settle down
            extend_secs = 0 if i<max_steps-1 else 2

            act = agent.act(obs, info)
            # print('obs 0', obs['color'][0].shape, obs['color'][1].shape, obs['color'][2].shape)
            # print('obs 1', obs['depth'][0].shape, obs['depth'][0].shape, obs['depth'][0].shape)
            # print('act',act)
            # print('info',info)
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act, extend_secs)
            # TODO FOR DEBUG CAHNGE DONE TO FALSE
            #done = False
            total_reward += reward
            print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                break
        #print('=====================')
        print('\n')
        episode.append((obs, None, reward, info)) # last episode as a record
        # Only save completed demonstrations.
        if total_reward > 0.99:
            dataset.add(seed, episode)
    print('conduct {} trials to collect {} successful {} demonstration'.format(trial_n, args.task, args.n))
    print('the planner successful rate is {}'.format(trial_n/args.n))

'''

if __name__ == '__main__':
  main()