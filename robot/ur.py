import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

HOME_JOINT = [-0.192222, -1.528962, 2.250302, -2.293070, -1.574837, -0.190362]

class UR():
    def __init__(self) -> None:
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)
        
        # We can get the name of the reference frame for this robot:
        planning_frame = self.move_group.get_planning_frame()
        print("Reference frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = self.move_group.get_end_effector_link()
        print("End effector: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("Robot Groups:", self.robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        # print("Printing robot state")
        # print(self.robot.get_current_state())

        rospy.loginfo("UR arm is ready.")

    def get_current_joint_values(self):
        return self.move_group.get_current_joint_values()
    
    def get_current_pose(self):
        return self.move_group.get_current_pose().pose
    
    def move_joint(self, joint_goal):
        # self.move_group.go(joint_goal, wait=True)
        # self.move_group.stop()
        self.move_group.set_joint_value_target(joint_goal)
        plan_success, plan, planning_time, error_code = self.move_group.plan()
        self.display_plan(plan)

        self.move_group.execute(plan, wait=True)
        self.move_group.stop()

    def move_pose(self, pose_goal):
        self.move_group.set_pose_target(pose_goal)
        # plan = self.move_group.go(wait=True)

        self.move_group.set_pose_target(pose_goal)
        plan_success, plan, planning_time, error_code = self.move_group.plan()
        print("plan result:", plan_success)
        self.display_plan(plan)

        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def move_cartesian_path(self, waypoints):
        (plan, fraction) = self.move_group.compute_cartesian_path(
                            waypoints,   # waypoints to follow
                            0.01,        # eef_step
                            0.0)         # jump_threshold
        print('fraction', fraction)

        self.display_plan(plan)
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()

    def display_plan(self, plan):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)
        
        while(input("Repeat display or enter [y] to execute:") != 'y'): # ask for user input
            self.display_trajectory_publisher.publish(display_trajectory)
    
    def move_home(self):
        self.move_joint(HOME_JOINT)

if __name__ == "__main__":
    # test only with simulation
    rospy.init_node("move_group_python_interface", anonymous=True)
    ur10 = UR()

    joint_goal = ur10.get_current_joint_values()
    ur10.move_home()
    print(joint_goal)
    # joint_goal[0] = pi
    # joint_goal[1] = -pi/4
    # joint_goal[2] = 0
    # joint_goal[3] = -pi/2
    # joint_goal[4] = 0
    # joint_goal[5] = pi/3
    # ur10.move_joint(joint_goal)
        
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.orientation.w = 1.0
    # pose_goal.position.x = 0.4
    # pose_goal.position.y = 0.4
    # pose_goal.position.z = 0.4
    # ur10.move_pose(pose_goal)

    # waypoints = []
    # scale = 1.0
    # wpose = ur10.get_current_pose()
    # wpose.position.z -= scale * 0.1  # First move up (z)
    # wpose.position.y += scale * 0.2  # and sideways (y)
    # waypoints.append(copy.deepcopy(wpose))
    # wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
    # waypoints.append(copy.deepcopy(wpose))
    # wpose.position.y -= scale * 0.1  # Third move sideways (y)
    # waypoints.append(copy.deepcopy(wpose))
    # ur10.move_cartesian_path(waypoints)
