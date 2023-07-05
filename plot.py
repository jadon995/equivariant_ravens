# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to plot training results."""

import os
import pickle

import argparse
import numpy as np
# from ravens.utils import utils
from sympy import root
import tensorflow as tf


parser = argparse.ArgumentParser(description='ravens')
parser.add_argument('--root_dir', type=str, default='./test')
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--task', type=str, default='block-insertion')
parser.add_argument('--n_rotations', type=int,default=36)# the demo used for testing
parser.add_argument('--n_demos', type=int,default=10)# the demo used for testing
parser.add_argument('--agent', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()



def main(args):
  name = f'{args.task}-{args.n_rotations}-{args.n_demos}-{args.seed}'
  print(name)

  # # Load and print results to console.
  # path = args.root_dir
  # curve = []
  # for fname in tf.io.gfile.listdir(path):
  #   fname = os.path.join(path, fname)
  #   if name in fname and '.pkl' in fname:
  #     n_steps = int(fname[(fname.rfind('-') + 1):-4])
  #     data = pickle.load(open(fname, 'rb'))
  #     rewards = []
  #     for reward, _ in data:
  #       rewards.append(reward)
  #     score = np.mean(rewards)
  #     std = np.std(rewards)
  #     curve.append((n_steps, score, std))
  # curve.sort()
  # for log in curve:
  #   print(f'  {log[0]} steps:\t{log[1]:.4f}%\t± {log[2]:.4f}%')

  path = os.path.join(args.root_dir, name, args.agent)
  curve = []
  for fname in tf.io.gfile.listdir(path):
    fname = os.path.join(path, fname)
    if name in fname and '.pkl' in fname:
      n_steps = int(fname[(fname.rfind('/') + 1):-4])
      data = pickle.load(open(fname, 'rb'))
      rewards = []
      for reward, _ in data:
        rewards.append(reward)
      score = np.mean(rewards)
      std = np.std(rewards)
      curve.append((n_steps, score, std))
  curve.sort()
  for log in curve:
    print(f'  {log[0]} steps:\t{log[1]:.4f}%\t± {log[2]:.4f}%')

  # Plot results over training steps.
  # title = f'{args.agent} on {name}'
  # ylabel = 'Testing Task Success (%)'
  # xlabel = '# of Training Steps'
  # if args.disp:
  #   logs = {}
  #   curve = np.array(curve)
  #   logs[name] = (curve[:, 0], curve[:, 1], curve[:, 2])
  #   fname = os.path.join(path, 'plot.png')
  #   utils.plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
  #   print(f'Done. Plot image saved to: {fname}')

if __name__ == '__main__':
  main(args)
