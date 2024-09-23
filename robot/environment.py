import os
import sys
from math import pi
import copy
import rospy
from zivid_camera.srv import *
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import geometry_msgs.msg
import tf2_ros

from transform import Transform, Rotation
import utils

import matplotlib.pyplot as plt
import numpy as np

from zivid import Zivid
from gripper import Robotiq
from ur import UR

GRIPPER_CONFIG = {
    'position': (0.0, 0.0, 0.2),
    'rotation': (pi, 0.0, pi/2)}

# BOUNDS = np.array([[0.67, 1.07], [-0.3, 0.3], [-0.1, 0.1]])
BOUNDS = np.array([[0.65, 1.07], [-0.28, 0.28], [-0.1, 0.1]]) # (0.42, 0.56, 0.2)


class Environment():
    def __init__(self) -> None:
        rospy.init_node("robot_env_node", anonymous=True)
        rospy.loginfo("Starting robot_env.py")

        # publish tf for debugging 
        self.tf_br = tf2_ros.TransformBroadcaster()

        # camera setup
        self.camera = Zivid()

        # arm setup
        self.arm = UR()

        # gripper setup
        self.gripper = Robotiq()

    def start_camera_loop(self):
        self.camera.start_trigger_loop()

    def end_camera_loop(self):
        self.camera.stop_trigger_loop()
    
    def get_obs(self):
        return self.camera.get_obs()
    
    def get_actions(self):
        while (input("Demo pick and enter [y] to record:") != 'y'):
            pass

        # self.move_to_gripper_pose((np.array([0.80, -0.2, 0.08]),
                                #    np.array([0, 0, 0, 1])))
        self.gripper.close()
        pick_pose = self.get_gripper_pose()
        

        while (input("Demo place and enter [y] to record:") != 'y'):
            pass
        
        # self.move_to_gripper_pose((np.array([0.80, 0.2, 0.1]),
                                #    np.array([0, 0, 0, 1])))
        self.gripper.move(130)
        place_pose = self.get_gripper_pose()
        
        # place angle is a relative value to the pick pose
        return {'pose0': pick_pose, 'pose1': place_pose}
    
    def get_gripper_pose(self):
        arm_pose = self.arm.get_current_pose()
        flange_to_base = (np.array([arm_pose.position.x,
                                    arm_pose.position.y,
                                    arm_pose.position.z]),
                          np.array([arm_pose.orientation.x,
                                    arm_pose.orientation.y,
                                    arm_pose.orientation.z,
                                    arm_pose.orientation.w]))
        gripper_to_flange = (
            np.asarray(GRIPPER_CONFIG['position']),
            utils.eulerXYZ_to_quatXYZW(GRIPPER_CONFIG['rotation']))

        gripper_to_base = utils.multiply(flange_to_base, gripper_to_flange)


        self._broad_transform(gripper_to_base) # debug
        return gripper_to_base
    
    def move_to_gripper_pose(self, pose):
        # assert pose >= -0.0318
        gripper_to_base = pose
        # self._broad_transform(gripper_to_base) # debug

        gripper_to_flange = (
            np.asarray(GRIPPER_CONFIG['position']),
            utils.eulerXYZ_to_quatXYZW(GRIPPER_CONFIG['rotation']))
        flange_to_gripper = utils.invert(gripper_to_flange)
        flange_to_base = utils.multiply(gripper_to_base, flange_to_gripper)
        self._broad_transform(flange_to_base) # debug

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = flange_to_base[0][0]
        pose_goal.position.y = flange_to_base[0][1]
        pose_goal.position.z = flange_to_base[0][2]
        pose_goal.orientation.x = flange_to_base[1][0]
        pose_goal.orientation.y = flange_to_base[1][1]
        pose_goal.orientation.z = flange_to_base[1][2]
        pose_goal.orientation.w = flange_to_base[1][3]
        
        self.arm.move_pose(pose_goal)
        pass

    def reset(self):
        self.arm.move_home()
        # self.gripper.move(100) # kit test
        # self.gripper.move(130) # train
        self.gripper.move(130) # red-in-green test

    
    def _broad_transform(self, pose):
        # visualize tf for debugging purpose
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = "debug_link"
        t.transform.translation.x = pose[0][0]
        t.transform.translation.y = pose[0][1]
        t.transform.translation.z = pose[0][2]
        t.transform.rotation.x = pose[1][0]
        t.transform.rotation.y = pose[1][1]
        t.transform.rotation.z = pose[1][2]
        t.transform.rotation.w = pose[1][3]
        
        self.tf_br.sendTransform(t)

    
if __name__ == '__main__':
    env = Environment()

    rate = rospy.Rate(10)  # 10 Hz

    # rospy.sleep(5)
    # env.gripper.close()
    # rospy.sleep(5)
    # env.gripper.open()

    # joint_goal = env.arm.get_current_joint_values()
    # joint_goal[0] = pi
    # joint_goal[1] = -pi/4
    # joint_goal[2] = 0
    # joint_goal[3] = -pi/2
    # joint_goal[4] = 0
    # joint_goal[5] = pi/3
    # env.arm.move_joint(joint_goal)
        
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.orientation.w = 1.0
    # pose_goal.position.x = 0.4
    # pose_goal.position.y = 0.4
    # pose_goal.position.z = 0.4
    # env.arm.move_pose(pose_goal)

    # waypoints = []
    # scale = 1.0
    # wpose = env.arm.get_current_pose()
    # wpose.position.z -= scale * 0.1  # First move up (z)
    # wpose.position.y += scale * 0.2  # and sideways (y)
    # waypoints.append(copy.deepcopy(wpose))
    # wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
    # waypoints.append(copy.deepcopy(wpose))
    # wpose.position.y -= scale * 0.1  # Third move sideways (y)
    # waypoints.append(copy.deepcopy(wpose))
    # env.arm.move_cartesian_path(waypoints)
    # pose = (np.array([0.673, 0.237, 0.231]),
            # np.array)
    
    # tran = utils.eulerXYZ_to_quatXYZW((pi, 0.0, pi/2))
    # print(tran)
    # env.move_to_gripper_pose((np.array([0.673, -0.234, 0.031]),
                            #  np.array([0, 0, 0, 1])))
    # env.move_to_gripper_pose((np.array([0.673, 0.234, 0.031]),
                            #  np.array([0, 0, 0, 1])))
    env.move_to_gripper_pose((np.array([1.0, 0.15, -0.03]),
                             np.array([0, 0, 0, 1])))
    
    print(env.get_gripper_pose())

    # while not rospy.is_shutdown():
        # rospy.sleep(5)  # Sleep to prevent using 100% CPU
        # env.camera.get_obs()
        # env.get_gripper_pose()

        # print(env.get_gripper_pose())