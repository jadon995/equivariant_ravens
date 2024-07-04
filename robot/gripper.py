import os
import sys
import rospy
from robotiq_2f_gripper_control.srv import *

class Robotiq():
    def __init__(self) -> None:
        rospy.wait_for_service("/robotiq/control_robotiq_2f_gripper", 30)
        self.gripper_service = rospy.ServiceProxy("/robotiq/control_robotiq_2f_gripper", Robotiq2FGripperService)

        self.reset()
        self.activate()
        rospy.loginfo("Robotiq gripper is ready.")

    def execute_commend(self, command):
        gripper_request = Robotiq2FGripperServiceRequest()
        gripper_request.command = command
        self.gripper_service(gripper_request)
        rospy.sleep(0.1)

    def activate(self):
        self.execute_commend('a')
        rospy.sleep(2.0)

    def reset(self):
        self.execute_commend('r')

    def open(self):
        self.execute_commend('o')

    def close(self):
        self.execute_commend('c')

    def move(self, value):
        value = int(value)
        value = max(value, 0)
        value = min(value, 255)
        self.execute_commend(str(value))

    

if __name__ == "__main__":
    rospy.init_node("robotiq_client_test", anonymous=True)
    robotiq = Robotiq()
    
    rospy.sleep(5)
    robotiq.close()

    rospy.sleep(5)
    robotiq.open()







