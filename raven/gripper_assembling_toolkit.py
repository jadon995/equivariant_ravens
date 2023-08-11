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

"""Kitting Tasks."""

from ctypes import util
import os
from re import template
import sys
import json

import numpy as np
from gripper_task import Task
import utils
import random
import matplotlib.pyplot as plt

import trimesh

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


class AssemblingToolKit(Task):
  """Kitting 3D tools"""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 4
    # self.train_set = np.arange(0, 10)
    # self.test_set = np.arange(10, 15)
    # self.homogeneous = False
    self.task_name = 'assembling-kits'
    self.tools = ['hammer', 'plier', 'wrench', 'screwdriver']
    self.kp_idx = [1, 0, 1, 1]
    self.train_set = np.arange(100)
    np.random.seed(0)
    np.random.shuffle(self.train_set)

  def reset(self, env):
    super().reset(env)

    colors = [
        utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]
    # np.random.shuffle(colors)
    symmetry = [
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi,
        2 * np.pi
    ]   

    # build kit
    kit_size = (0.28, 0.26, 0.05)
    kit_pose = self.get_random_pose(env, kit_size)
    pos = (kit_pose[0][0], kit_pose[0][1], 0)
    rot = kit_pose[1]
    kit_pose = (pos, rot)

    # kit_num = np.random.choice(self.train_set, 1)[0]
    kit_num = 25
    kit_shape = os.path.join(self.assets_root, f'tool3d', f'kit_train',
        f'{kit_num:04d}', f'kit.obj')
    kit_shape_coll = os.path.join(self.assets_root, f'tool3d', f'kit_train',
        f'{kit_num:04d}', f'kit_coll.obj')
    replace = {'FNAME': (kit_shape,), 'FNAMECOLL': (kit_shape_coll,),
                  'SCALE': (1, 1, 1), 'COLOR': (0.61176471, 0.45882353, 0.37254902)}
    template = 'tool3d/kit-template.urdf'
    urdf = self.fill_template(template, replace)
    env.add_object(urdf, kit_pose, 'fixed')
    os.remove(urdf)

    # load kit info 
    kit_info = os.path.join(self.assets_root, f'tool3d', f'kit_train',
        f'{kit_num:04d}', f'info.json')
    with open(kit_info, 'r') as json_file:
      kit_data = json.load(json_file)
    
    # TODO targets
    targets = []
    scales = []
    pick_points = []
    for i, obj in enumerate(kit_data):
      fname = os.path.join(self.assets_root, f'tool3d', obj['type'], f"{obj['id']:02d}.json")
      with open(fname, 'r') as json_file:
        kp_data = json.load(json_file)
        
      pick_point = kp_data['keypoints'][self.kp_idx[i]]["xyz"]
      pick_point = np.array([float(value) for value in (pick_point[1:-1]).split(',')]) * obj['scale']
      # pick_point[0] -= 0.001
      
      if i == 0 or i == 2:
        pick_point[1] -= 0.02
      pick_points.append(pick_point)

      pos = utils.apply(kit_pose, obj['pos'])
      rot = utils.quatXYZW_to_eulerXYZ(kit_pose[1])
      rot = utils.eulerXYZ_to_quatXYZW((0, 0, rot[2]))
      targets.append((pos, rot))
      scales.append(obj['scale'])


    # Add objects
    objects = []
    matches = []
    shapes = {}
    sizes =[[0.12, 0.10, 0.02],
            [0.08, 0.10, 0.02],
            [0.05, 0.10, 0.02],
            [0.05, 0.10, 0.02]] 
    template = 'tool3d/object-template.urdf'
    for i, obj in enumerate(kit_data):
      shape = obj['id']
      size = sizes[i]
      pose = self.get_random_pose(env, size)
      fname = os.path.join(self.assets_root, f'tool3d', obj['type'], f"{shape:02d}.obj")
      fname_coll = os.path.join(self.assets_root, f'tool3d', obj['type'], f"{obj['id']:02d}_coll.obj")
      scale = [scales[i]] * 3
      replace = {'FNAME': (fname,), 'FNAMECOLL': (fname_coll,) ,'SCALE': scale, 'COLOR': colors[i]}
      urdf = self.fill_template(template, replace)
      block_id = env.add_object(urdf, pose)
      os.remove(urdf)

      # load meshes
      mesh = trimesh.load_mesh(str(fname_coll))
      mesh = np.asarray(mesh.vertices)

      # grasp keypoint
      pick_point = pick_points[i]

      # objects.append((block_id, (symmetry[shape], None)))
      objects.append((block_id, (symmetry[shape], mesh), pick_point))
      
      shapes[obj['type']] = obj['id']
    print("Kit {:02d}: ".format(kit_num), shapes)
    matches = [[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1]]
    
    matches = np.int32(matches)
    # # Pose metric
    # self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))

    # Using metric of average closest distance
    self.goals.append((objects, matches, targets, True, True, 'adi', None, 1))
    

