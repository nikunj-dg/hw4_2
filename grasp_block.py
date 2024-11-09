#!/usr/bin/env python

"""
CS 6301 Homework 4 Programming
Robot Control for Grasping
"""

import sys
import time
import rospy
import roslib
import tf
import numpy as np
import moveit_commander
import actionlib
import threading

from transforms3d.quaternions import mat2quat, quat2mat
from geometry_msgs.msg import PoseStamped
from trac_ik_python.trac_ik import IK

roslib.load_manifest('gazebo_msgs')
from gazebo_msgs.srv import GetModelState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

"""
Gripper
"""
import actionlib
import control_msgs.msg
import rospy
import sys, time
import argparse
#import tmc_control_msgs.msg
# HSR uses: tmc_control_msgs.msg.GripperApplyEffortActionGoal (?)

CLOSED_POS = 0.0   # The position for a fully-closed gripper (meters).
OPENED_POS = 0.10  # The position for a fully-open gripper (meters).
ACTION_SERVER = 'gripper_controller/gripper_action'

# Unfortunately none of these work for the HSR :-( incompatible types
#ACTION_SERVER = '/hsrb/gripper_controller/apply_force'
#ACTION_SERVER = '/hsrb/gripper_controller/follow_joint_trajectory'
#ACTION_SERVER = '/hsrb/grasp_state_request_action'
#ACTION_SERVER = '/hsrb/gripper_controller/grasp'

class Gripper(object):
    """Gripper controls the robot's gripper.
    """
    MIN_EFFORT = 35   # Min grasp force, in Newtons
    MAX_EFFORT = 100  # Max grasp force, in Newtons

    def __init__(self):
        self._client = actionlib.SimpleActionClient(ACTION_SERVER, control_msgs.msg.GripperCommandAction)
        self._client.wait_for_server(rospy.Duration(10))

    def open(self):
        """Opens the gripper.
        """
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = OPENED_POS
        self._client.send_goal_and_wait(goal, rospy.Duration(10))

    def close(self, max_effort=MAX_EFFORT):
        """Closes the gripper.

        The `goal` has type:
            <class 'control_msgs.msg._GripperCommandGoal.GripperCommandGoal'>
        with a single attribute, accessed via `goal.command`, which consists of:
            position: 0.0
            max_effort: 0.0
        by default, and is of type:
            <class 'control_msgs.msg._GripperCommand.GripperCommand'>

        Args:
            max_effort: The maximum effort, in Newtons, to use. Note that this
                should not be less than 35N, or else the gripper may not close.
        """
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = CLOSED_POS
        goal.command.max_effort = max_effort
        self._client.send_goal_and_wait(goal, rospy.Duration(10))


def wait_for_time():
    """Wait for simulated time to begin.

    A useful method. Note that rviz will display the ROS Time in the bottom left
    corner. For Gazebo, just click the play button if it's paused to start.
    """
    while rospy.Time().now().to_sec() == 0:
        pass
        
        
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Close or open gripper')
    parser.add_argument('--close', dest='close',
                        help='close gripper',
                        action='store_true')
    parser.add_argument('--open', dest='open',
                        help='open_gripper',
                        action='store_true')                        

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args        


if __name__ == "__main__":

    args = parse_args()
    print('Called with args:')
    print(args)

    # Looks like this works for the Fetch :-)
    # Check if the ROS node is already initialized
    if not rospy.core.is_initialized():
        rospy.init_node('gripper_demo')

    wait_for_time()
    time_delay = 1
    use_delay = True

    print("Now forming the gripper")
    gripper = Gripper()
    
    if args.close:
        gripper.close()
        print("gripper now closed")
        if use_delay:
            time.sleep(time_delay)
    elif args.open:
        gripper.open()
        print("gripper now open")
        if use_delay:
            time.sleep(time_delay)

    '''
    gripper.close(35)
    print("gripper now closed")
    if use_delay:
        time.sleep(time_delay)

    gripper.open()
    print("gripper now open")
    if use_delay:
        time.sleep(time_delay)

    # closes very slowly ...
    gripper.close(1)
    print("gripper now closed")
    if use_delay:
        time.sleep(time_delay)
    '''

"""
Gripper Over
"""


def ros_quat(tf_quat): #wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat
    
    
# rotation matrix about Y axis
def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


# Convert quaternion and translation to a 4x4 tranformation matrix
# See Appendix B.3 in Lynch and Park, Modern Robotics for the definition of quaternion
def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


# Convert a ROS pose message to a 4x4 tranformation matrix
def ros_pose_to_rt(pose):
    qarray = [0, 0, 0, 0]
    qarray[0] = pose.orientation.x
    qarray[1] = pose.orientation.y
    qarray[2] = pose.orientation.z
    qarray[3] = pose.orientation.w

    t = [0, 0, 0]
    t[0] = pose.position.x
    t[1] = pose.position.y
    t[2] = pose.position.z

    return ros_qt_to_rt(qarray, t)

T_bo = 0
# Query pose of frames from the Gazebo environment
def get_pose_gazebo(model_name, relative_entity_name=''):

    def gms_client(model_name, relative_entity_name):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp1 = gms(model_name, relative_entity_name)
            return resp1
        except (rospy.ServiceException, e):
            print("Service call failed: %s" % e)
           
    # query the object pose in Gazebo world T_wo
    res = gms_client(model_name, relative_entity_name) 
    T_wo = ros_pose_to_rt(res.pose)  
    
    # query fetch base link pose in Gazebo world T_wb
    res = gms_client(model_name='fetch', relative_entity_name='base_link')
    T_wb = ros_pose_to_rt(res.pose)
    
    ################ TO DO ##########################
    # compute the object pose in robot base link T_bo
    # use your code from homework 2

    T_bw = np.linalg.inv(T_wb)
    T_bo = np.dot(T_bw, T_wo)
    ################ TO DO ##########################
    
    return T_bo
    

# publish tf for visualization
def publish_tf(trans, qt, model_name):

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        br.sendTransform(trans, qt, rospy.Time.now(), model_name, 'base_link')
        rate.sleep()
    
    
# Send a trajectory to controller
class FollowTrajectoryClient(object):

    def __init__(self, name, joint_names):
        self.client = actionlib.SimpleActionClient("%s/follow_joint_trajectory" % name,
                                                   FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for %s..." % name)
        self.client.wait_for_server()
        self.joint_names = joint_names

    def move_to(self, positions, duration=5.0):
        if len(self.joint_names) != len(positions):
            print("Invalid trajectory position")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = positions
        trajectory.points[0].velocities = [0.0 for _ in positions]
        trajectory.points[0].accelerations = [0.0 for _ in positions]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory

        self.client.send_goal(follow_goal)
        self.client.wait_for_result()    

    
if __name__ == "__main__":
    """
    Main function to run the code
    """
    
    # intialize ros node
    if not rospy.core.is_initialized():
        rospy.init_node('planning_scene_block')
    
    # query the demo cube pose
    model_name = 'demo_cube'
    T = get_pose_gazebo(model_name)
    print('pose of the demo cube')
    print(T)
    
    # translation
    trans = T[:3, 3]
    # quaternion in ros
    qt = ros_quat(mat2quat(T[:3, :3]))
    
    # publish the cube tf for visualization
    x = threading.Thread(target=publish_tf, args=(trans, qt, model_name))
    x.start()
    
    # gripper controller
    gripper = Gripper()
    gripper.open()
    
    # # Setup clients
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    
    # Raise the torso using just a controller
    rospy.loginfo("Raising torso...")
    torso_action.move_to([0.4, ])
    
    # --------- initialize moveit components ------
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander('arm')
    # planning scene
    scene = moveit_commander.PlanningSceneInterface()
    scene.clear()
    robot = moveit_commander.RobotCommander()
    
    # print information about the planner
    planning_frame = group.get_planning_frame()
    print("============ Reference frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print("============ End effector: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Robot Groups:", robot.get_group_names())

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print("============ Printing robot state")
    joint_state = robot.get_current_state().joint_state
    for i in range(len(joint_state.name)):
        print(joint_state.name[i], joint_state.position[i])
    
    # add objects into the planning scene
    rospy.sleep(1.0)
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()    
    # add a box for robot base to avoid hitting the base
    p.pose.position.x = 0
    p.pose.position.y = 0
    p.pose.position.z = 0.18
    scene.add_box("base", p, (0.56, 0.56, 0.4))
    
    # add a big box for the table
    p.pose.position.x = 0.9
    p.pose.position.y = 0
    p.pose.position.z = trans[2] - 0.06 / 2 - 0.51 - 0.2
    scene.add_box("table", p, (1, 5, 1))    
    
    # get the current joints
    joints = group.get_current_joint_values()
    print('current joint state of the robot')
    print(group.get_active_joints())
    print(joints)
    
    # define the IK solver from track_ik
    ik_solver = IK("base_link", "wrist_roll_link")
    
    # change the joint limit of torso_lift_joint in order to fix the torso lift
    lower_bound, upper_bound = ik_solver.get_joint_limits()
    lower_bound = list(lower_bound)
    upper_bound = list(upper_bound)
    lower_bound[0] = 0.4
    upper_bound[0] = 0.4
    ik_solver.set_joint_limits(lower_bound, upper_bound)

    # use initial seed as zeros
    seed_state = [0.0] * ik_solver.number_of_joints
    seed_state[0] = 0.4
    
    ################ TO DO ##########################
    # use the get_ik function from trac_ik to compute the joints of the robot for grasping the cube
    # return the solution to a "sol" variable
    # Refer to https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_python/
    
    x_offset = -0.20
    z_offset = 0.05

    print("x offset values is :",x_offset)
    print("z offset values is :",z_offset)
    
    desired_pose = PoseStamped()
    # set offset value to keep the end grip little behind the table 
    desired_pose.pose.position.x = trans[0] + x_offset # offset on x-axis
    desired_pose.pose.position.y = trans[1] 
    # set the offset value to keep the end grip on the exact position to grab the demo cube
    desired_pose.pose.position.z = trans[2] + z_offset # offset on z-axis

    # get the values for the orientation values
    desired_pose. pose.orientation.x = qt[0]
    desired_pose.pose.orientation.y = qt[1]
    desired_pose. pose.orientation.z = qt[2]
    desired_pose.pose.orientation.w = qt[3]

    print(" end effector value for desired pose")
    print(desired_pose)
    #Use the get_ik function to compute the joint angles
    sol = ik_solver.get_ik(seed_state,
                       desired_pose.pose.position.x,desired_pose.pose.position.y,
                       desired_pose.pose.position.z,
                       desired_pose.pose.orientation.x,desired_pose.pose.orientation.y,
                       desired_pose.pose.orientation.z, desired_pose. pose.orientation.w)
    print(sol)
    while sol is None:
        print("trying to get the sol values")
        sol = ik_solver.get_ik(seed_state,
                       desired_pose.pose.position.x,desired_pose.pose.position.y,
                       desired_pose.pose.position.z,
                       desired_pose.pose.orientation.x,desired_pose.pose.orientation.y,
                       desired_pose.pose.orientation.z, desired_pose. pose.orientation.w)
        if sol is not None:
            break

    print("sol values are:")
    print(sol)
    ################ TO DO ##########################
    
    # move to the joint goal
    joint_goal = sol[1:]
    group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    group.stop()
    
    # close gripper
    gripper.close()
    
    # move back
    group.go(joints, wait=True)
    group.stop()
