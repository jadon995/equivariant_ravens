import os
import sys
from math import pi
import copy
import rospy
import ros_numpy
import dynamic_reconfigure.client
from zivid_camera.srv import *
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import numpy as np
import utils
from std_msgs.msg import Bool

ACQUISITION_NUM = 5 # Acquisition number for the camera

CAMERA_CONFIG = [{
    'image_size': (1200, 1920),
    'intrinsics': (2779.074462890625, 0.0, 980.9558715820312, 
                   0.0, 2778.749755859375, 583.1567993164062, 
                   0.0, 0.0, 1.0),
    'position': (0.894348, 0.162832, 0.939701),    # extrinsic calibration
    # 'rotation': utils.eulerXYZ_to_quatXYZW((2.98994, -0.022727, 1.55985)),     # extrinsic calibration
    'rotation': (0.708273, -0.701772, 0.0452119, 0.0618219),     # extrinsic calibration
    'zrange': (0.6, 2.0),
    'noise': False}]


class Zivid():
    def __init__(self) -> None:
        # camera setup
        rospy.wait_for_service("/zivid_camera/capture", 30)
        self.capture_service = rospy.ServiceProxy("/zivid_camera/capture", Capture)
        rospy.Subscriber("/zivid_camera/color/image_color", Image, self.__color_callback)
        rospy.Subscriber("/zivid_camera/depth/image", Image, self.__depth_callback)

        camera_trigger_timer = rospy.Timer(rospy.Duration(1), self._trigger_loop_callback)
        
        self.__config_camera() # set of the camera configuration

        self.color_image, self.depth_image = None, None
        self.camera_config = CAMERA_CONFIG

        self.keep_capturing = False
        pass

    def get_obs(self):
        obs = {"color": [], "depth": []}

        # triger the camera
        self.color_image, self.depth_image = None, None
        self.capture_service()
        # while not (self.color_image and self.depth_image):
        while self.color_image is None or self.depth_image is None:
            rospy.sleep(0.1)

        obs["color"].append(self.color_image)
        obs["depth"].append(self.depth_image)

        return obs
        

    def __color_callback(self, data):
        assert isinstance(data, Image)
        # rospy.loginfo("Color image received")

        # np_image = ros_numpy.numpify(data)
        np_image = ros_numpy.image.image_to_numpy(data)
        # print(f"color image shape {np_image.shape} type {np_image.dtype}")

        self.color_image = np_image[:, :, :3] # remove alpha channel
        
        # plt.imshow(np_image)
        # plt.axis("off")
        # plt.show()


    def __depth_callback(self, data):
        assert isinstance(data, Image)
        # rospy.loginfo("Depth image received")

        depth = ros_numpy.image.image_to_numpy(data)
        # print(f"depth image shape {depth.shape}")
        # print(f"depth max {np.nanmax(depth)}")
        # print(f"depth min {np.nanmin(depth)}")

        # Commented as zivid depth image is in meters
        # (zfar, znear) = CAMERA_CONFIG[0]["zrange"]
        # depth = (zfar + znear - (2. * depth - 1.) * (zfar - znear))
        # depth = (2. * znear * zfar) / depth
        # print(f"depth max {np.nanmax(depth)}")
        # print(f"depth min {np.nanmin(depth)}")

        self.depth_image = depth

        # plt.imshow(zbuffer)
        # plt.axis("off")
        # plt.show()
        return
    
    
    def __config_camera(self):
        rospy.loginfo("Enabling the reflection filter")
        settings_client = dynamic_reconfigure.client.Client("/zivid_camera/settings/")
        settings_config = {"processing_filters_reflection_removal_enabled": True}
        settings_client.update_configuration(settings_config)
        
        for i in range(ACQUISITION_NUM):
            rospy.loginfo(f"Enabling and configure Acquisition {i}")
            acquisition_client = dynamic_reconfigure.client.Client(
               f"/zivid_camera/settings/acquisition_{i}"
            )
            acquisition_config = {
            "enabled": True,
            "aperture": 5.66,
            "exposure_time": 20000,
            }
            acquisition_client.update_configuration(acquisition_config)

    
    def _trigger_loop_callback(self, event):
        while self.keep_capturing:
            self.capture_service()
            rospy.sleep(1)

    def start_trigger_loop(self):
        self.keep_capturing = True

    def stop_trigger_loop(self):
        self.keep_capturing = False


if __name__ == "__main__":
    rospy.init_node("zivid_capture_node", anonymous=True)

    pose = (np.array([-0.276857, 0.872525, 0.942407]),
            np.array([0.708273, -0.701772, 0.0452119, 0.0618219]))
    euler = utils.quatXYZW_to_eulerXYZ(pose[1])
    print(euler) # (0.15161050485454336, -3.1186019216361607, -1.578271003032245)

    # zivid = Zivid()
    # while not rospy.is_shutdown():
    #     rospy.sleep(5)  # Sleep to prevent using 100% CPU
    #     zivid.get_obs()