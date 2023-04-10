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

import open3d as o3d

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


class AssemblingKits(Task):
  """Kitting Tasks base class."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.ee = 'suction'
    self.max_steps = 10
    # self.metric = 'pose'
    # self.primitive = 'pick_place'
    self.train_set = np.arange(0, 14)
    self.test_set = np.arange(14, 20)
    self.homogeneous = False
    self.task_name = 'assembling-kits'

  def reset(self, env):
    super().reset(env)

    # Add kit.
    kit_size = (0.28, 0.2, 0.005)
    kit_urdf = 'kitting/kit.urdf'
    kit_pose = self.get_random_pose(env, kit_size)
    env.add_object(kit_urdf, kit_pose, 'fixed')

    n_objects = 5
    if self.mode == 'train':
      obj_shapes = np.random.choice(self.train_set, n_objects)
    else:
      if self.homogeneous:
        obj_shapes = [np.random.choice(self.test_set)] * n_objects
      else:
        obj_shapes = np.random.choice(self.test_set, n_objects)

    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]

    symmetry = [
        2 * np.pi, 2 * np.pi, 2 * np.pi / 3, np.pi / 2, np.pi / 2, 2 * np.pi,
        np.pi, 2 * np.pi / 5, np.pi, np.pi / 2, 2 * np.pi / 5, 0, 2 * np.pi,
        2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 2 * np.pi / 6, 2 * np.pi, 2 * np.pi
    ]

    # Build kit.
    targets = []
    targ_pos = [[-0.09, 0.045, 0.0014], [0, 0.045, 0.0014],
                [0.09, 0.045, 0.0014], [-0.045, -0.045, 0.0014],
                [0.045, -0.045, 0.0014]]
    template = 'kitting/object-template.urdf'
    for i in range(n_objects):
      shape = os.path.join(self.assets_root, 'kitting',
                           f'{obj_shapes[i]:02d}.obj')
      scale = [0.003, 0.003, 0.0001]  # .0005
      pos = utils.apply(kit_pose, targ_pos[i])
      theta = np.random.rand() * 2 * np.pi
      rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
      replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)} # 0.61176471, 0.45882353, 0.37254902
      urdf = self.fill_template(template, replace)
      env.add_object(urdf, (pos, rot), 'fixed')
      os.remove(urdf)
      targets.append((pos, rot))

    # Add objects.
    objects = []
    matches = []
    # objects, syms, matcheses = [], [], []
    for i in range(n_objects):
      shape = obj_shapes[i]
      size = (0.08, 0.08, 0.02)
      pose = self.get_random_pose(env, size)
      fname = f'{shape:02d}.obj'
      fname = os.path.join(self.assets_root, 'kitting', fname)
      scale = [0.003, 0.003, 0.001]  # .0005
      replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': colors[i]}
      urdf = self.fill_template(template, replace)
      block_id = env.add_object(urdf, pose)
      os.remove(urdf)
      objects.append((block_id, (symmetry[shape], None)))
      # objects[block_id] = symmetry[shape]
      match = np.zeros(len(targets))
      match[np.argwhere(obj_shapes == shape).reshape(-1)] = 1
      matches.append(match)
      # print(targets)
      # exit()
      # matches.append(list(np.argwhere(obj_shapes == shape).reshape(-1)))
    matches = np.int32(matches)
    # print(matcheses)
    # exit()

    # Add goal.
    # self.goals.append((objects, syms, targets, 'matches', 'pose', 1.))

    # Goal: objects are placed in their respective kit locations.
    # print(objects)
    # print(matches)
    # print(targets)
    # exit()
    self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))
    # goal = Goal(objects, syms, targets)
    # metric = Metric('pose-matches', None, 1.)
    # self.goals.append((goal, metric))

    # # Goal: box is aligned with corner (1 of 4 possible poses).

    # visualize the images 
    # color, depth, segm = env.render_camera(self.oracle_cams[0])
    # plt.imshow(depth, cmap='gray')
    # plt.show()
    # plt.imshow(color)
    # plt.show()

class AssemblingKitsTool(Task):
  """Kitting Task - Plier."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 10
    self.train_set = np.arange(0, 10)
    self.test_set = np.arange(10, 15)
    self.homogeneous = False
    self.task_name = 'assembling-kits'

  def reset(self, env):
    super().reset(env)

    tools = ['hammer', 'plier', 'screwdriver', 'wrench']
    n_objects = [1, 1, 1, 1]

    # Add kit
    kit_size = (0.28, 0.26, 0.005)
    kit_urdf = "tool/kit.urdf"
    kit_pose = self.get_random_pose(env, kit_size)
    env.add_object(kit_urdf, kit_pose, 'fixed')

    obj_shapes = []
    for i, tool in enumerate(tools):
      if self.mode == 'train':
        obj_shape = np.random.choice(self.train_set, n_objects[i])
      else:
        if self.homogeneous:
          obj_shape = [np.random.choice(self.test_set)] * n_objects[i]
        else:
          obj_shape = np.random.choice(self.test_set, n_objects[i])
      obj_shapes.append(obj_shape)

    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]
    random.shuffle(colors)

    symmetry = [
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi,
        2 * np.pi
    ]    

    # # Build kit.
    targets = []
    targ_pos = [[-0.02, 0.03, 0.0014],
                [0.02, -0.03, 0.0014],
                [0.02, 0.09, 0.0014],
                [-0.02, -0.09, 0.0014]]
    template = 'tool/object-template.urdf'
    for j, tool in enumerate(tools):
      for i in range(n_objects[j]):
        shape = os.path.join(self.assets_root, f'tool', f'2d', tool,
                           f'{obj_shapes[j][i]:02d}.obj')
        shape_coll = os.path.join(self.assets_root, f'tool', f'2d', tool,
                           f'{obj_shapes[j][i]:02d}_coll.obj')
        scale = [1, 1, 0.1]
        pos = utils.apply(kit_pose, targ_pos[j]) # need modified

        rot = utils.quatXYZW_to_eulerXYZ(kit_pose[1])
        theta = rot[2] + (-1)** i * np.pi / 2
        # theta = rot[2] + i % 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        replace = {'FNAME': (shape,), 'FNAMECOLL': (shape_coll,),
                  'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
        urdf = self.fill_template(template, replace)
        env.add_object(urdf, (pos, rot), 'fixed')
        os.remove(urdf)
        targets.append((pos, rot))

    # Add objects.
    objects = []
    matches = []
    sizes =[[0.12, 0.15, 0.02],
            [0.08, 0.15, 0.02],
            [0.05, 0.15, 0.02],
            [0.04, 0.15, 0.02]]
    for j, tool in enumerate(tools):
      for i in range(n_objects[j]):
        shape = obj_shapes[j][i]
        # size = (0.12, 0.15, 0.02)
        size = sizes[j]
        pose = self.get_random_pose(env, size)
        fname = os.path.join(self.assets_root, f'tool', f'2d', tool, f'{shape:02d}.obj')
        fname_coll = os.path.join(self.assets_root, f'tool', f'2d', tool, f'{shape:02d}_coll.obj')
        scale = [1, 1, 1]  # .0005
        replace = {'FNAME': (fname,), 'FNAMECOLL': (fname_coll,) ,'SCALE': scale, 'COLOR': colors[j]}
        urdf = self.fill_template(template, replace)
        block_id = env.add_object(urdf, pose)
        # print('block_id', block_id, 'pose')
        os.remove(urdf)
        objects.append((block_id, (symmetry[shape], None)))
        # match = np.zeros(len(targets))
        # match[np.argwhere(obj_shapes[j] == shape).reshape(-1)] = 1
        # matches.append(match)

    matches = [[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1]]
    
    matches = np.int32(matches)
    self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))  

class AssemblingKitsScrewDriver(Task):
  """Kitting Task - Screw Driver."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 10
    self.train_set = np.arange(0, 10)
    self.test_set = np.arange(10, 15)
    self.homogeneous = True
    self.task_name = 'assembling-kits'

  def reset(self, env):
    super().reset(env)

    # Add kit
    kit_size = (0.24, 0.26, 0.005)
    kit_urdf = "tool/kit.urdf"
    kit_pose = self.get_random_pose(env, kit_size)
    env.add_object(kit_urdf, kit_pose, 'fixed')

    n_objects = 6
    if self.mode == 'train':
      obj_shapes = np.random.choice(self.train_set, n_objects)
    else:
      if self.homogeneous:
        obj_shapes = [np.random.choice(self.test_set)] * n_objects
      else:
        obj_shapes = np.random.choice(self.test_set, n_objects)
    
    print(obj_shapes)
    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]
    random.shuffle(colors)

    symmetry = [
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi,
        2 * np.pi
    ]

    # Build kit.
    targets = []
    targ_pos = [[0.00, 0.10, 0.0014],
                [0.00, 0.06, 0.0014],
                [0.00, 0.02, 0.0014],
                [0.00, -0.02, 0.0014],
                [0.00, -0.06, 0.0014],
                [0.00, -0.10, 0.0014]]
    scales = [1.11, 1, 0.89, 0.78, 0.67, 0.56] # different scales [20, 18, ..., 10]
    
    template = 'tool/object-template.urdf'
    for i in range(n_objects):
      shape = os.path.join(self.assets_root, f'tool', f'2d', f'screwdriver',
                           f'{obj_shapes[i]:02d}.obj')
      shape_coll = os.path.join(self.assets_root, f'tool', f'2d', f'screwdriver',
                           f'{obj_shapes[i]:02d}_coll.obj')
      scale = [scales[i], scales[i], 0.1]
      pos = utils.apply(kit_pose, targ_pos[i])

      rot = utils.quatXYZW_to_eulerXYZ(kit_pose[1])
      theta = rot[2] + (-1)** i * np.pi / 2
      rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

      replace = {'FNAME': (shape,), 'FNAMECOLL': (shape_coll,),
                  'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
      urdf = self.fill_template(template, replace)
      env.add_object(urdf, (pos, rot), 'fixed')
      os.remove(urdf)
      targets.append((pos, rot))

    # Add objects.
    objects = []
    matches = []
    # objects, syms, matcheses = [], [], []
    for i in range(n_objects):
      shape = obj_shapes[i]
      size = (0.04, 0.10, 0.02)
      pose = self.get_random_pose(env, size)
      fname = os.path.join(self.assets_root, f'tool', f'2d', 'screwdriver', f'{shape:02d}.obj')
      fname_coll = os.path.join(self.assets_root, f'tool', f'2d', 'screwdriver', f'{shape:02d}_coll.obj')
      scale = [scales[i], scales[i], 1]  # .0005
      replace = {'FNAME': (fname,), 'FNAMECOLL': (fname_coll,),
                  'SCALE': scale, 'COLOR': colors[i % len(colors)]}
      urdf = self.fill_template(template, replace)
      block_id = env.add_object(urdf, pose)
      os.remove(urdf)
      objects.append((block_id, (symmetry[shape], None)))

    matches = [[1,0,0,0,0,0],
               [0,1,0,0,0,0],
               [0,0,1,0,0,0],
               [0,0,0,1,0,0],
               [0,0,0,0,1,0],
               [0,0,0,0,0,1]]

    matches = np.int32(matches)
    self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))

class AssemblingKits3DTool(Task):
  """Kitting Task - Hammer, piler, screwdriver, wrench."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 10
    self.train_set = np.arange(0, 10)
    self.test_set = np.arange(10, 15)
    self.homogeneous = False
    self.task_name = 'assembling-kits'

  def reset(self, env):
    super().reset(env)

    tools = ['hammer', 'plier', 'screwdriver', 'wrench']
    n_objects = [1, 1, 1, 1]

    # Add kit
    kit_size = (0.28, 0.26, 0.005)
    kit_urdf = "tool/kit.urdf"
    kit_pose = self.get_random_pose(env, kit_size)
    env.add_object(kit_urdf, kit_pose, 'fixed')

    obj_shapes = []
    for i, tool in enumerate(tools):
      if self.mode == 'train':
        obj_shape = np.random.choice(self.train_set, n_objects[i])
      else:
        if self.homogeneous:
          obj_shape = [np.random.choice(self.test_set)] * n_objects[i]
        else:
          obj_shape = np.random.choice(self.test_set, n_objects[i])
      obj_shapes.append(obj_shape)

    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]
    random.shuffle(colors)

    symmetry = [
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 
        2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi,
        2 * np.pi
    ]    

    # # Build kit.
    targets = []
    targ_pos = [[-0.02, 0.03, 0.0014],
                [0.02, -0.03, 0.0014],
                [0.02, 0.09, 0.0014],
                [-0.02, -0.09, 0.0014]]
    template = 'tool/object-template.urdf'
    for j, tool in enumerate(tools):
      for i in range(n_objects[j]):
        shape = os.path.join(self.assets_root, f'tool', f'2d', tool,
                           f'{obj_shapes[j][i]:02d}.obj')
        shape_coll = os.path.join(self.assets_root, f'tool', f'2d', tool,
                           f'{obj_shapes[j][i]:02d}_coll.obj')
        scale = [1, 1, 0.1]
        pos = utils.apply(kit_pose, targ_pos[j]) # need modified

        rot = utils.quatXYZW_to_eulerXYZ(kit_pose[1])
        theta = rot[2] + (-1)** i * np.pi / 2
        # theta = rot[2] + i % 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        replace = {'FNAME': (shape,), 'FNAMECOLL': (shape_coll,),
                  'SCALE': scale, 'COLOR': (0.61176471, 0.45882353, 0.37254902)}
        urdf = self.fill_template(template, replace)
        env.add_object(urdf, (pos, rot), 'fixed')
        os.remove(urdf)
        targets.append((pos, rot))

    # Add objects.
    objects = []
    matches = []
    sizes =[[0.12, 0.15, 0.02],
            [0.08, 0.15, 0.02],
            [0.05, 0.15, 0.02],
            [0.04, 0.15, 0.02]]
    for j, tool in enumerate(tools):
      for i in range(n_objects[j]):
        shape = obj_shapes[j][i]
        # size = (0.12, 0.15, 0.02)
        size = sizes[j]
        pose = self.get_random_pose(env, size)
        fname = os.path.join(self.assets_root, f'tool', f'3d', tool, f'{shape:02d}.obj')
        fname_coll = os.path.join(self.assets_root, f'tool', f'3d', tool, f'{shape:02d}_coll.obj')
        scale = [1, 1, 1]  # .0005
        replace = {'FNAME': (fname,), 'FNAMECOLL': (fname_coll,) ,'SCALE': scale, 'COLOR': colors[j]}
        urdf = self.fill_template(template, replace)
        block_id = env.add_object(urdf, pose)
        # print('block_id', block_id, 'pose')
        os.remove(urdf)
        objects.append((block_id, (symmetry[shape], None)))
        # match = np.zeros(len(targets))
        # match[np.argwhere(obj_shapes[j] == shape).reshape(-1)] = 1
        # matches.append(match)

    matches = [[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1]]
    
    matches = np.int32(matches)
    self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))

class AssemblingKits3DToolKit(Task):
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
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]
    random.shuffle(colors)
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

    kit_num = np.random.choice(self.train_set, 1)[0]
    # kit_num = 13
    kit_shape = os.path.join(self.assets_root, f'tool3d', f'kit_{self.mode}',
        f'{kit_num:04d}', f'kit.obj')
    kit_shape_coll = os.path.join(self.assets_root, f'tool3d', f'kit_{self.mode}',
        f'{kit_num:04d}', f'kit_coll.obj')
    replace = {'FNAME': (kit_shape,), 'FNAMECOLL': (kit_shape_coll,),
                  'SCALE': (1, 1, 1), 'COLOR': (0.61176471, 0.45882353, 0.37254902)}
    template = 'tool3d/kit-template.urdf'
    urdf = self.fill_template(template, replace)
    env.add_object(urdf, kit_pose, 'fixed')
    os.remove(urdf)

    # load kit info 
    kit_info = os.path.join(self.assets_root, f'tool3d', f'kit_{self.mode}',
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
      mesh = o3d.io.read_triangle_mesh(str(fname_coll))
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
    

