cmake_minimum_required(VERSION 3.0.2)
project(hw4)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  control_msgs
  tf
  actionlib
  trajectory_msgs
  moveit_commander
  gazebo_msgs
  trac_ik_python
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs control_msgs tf actionlib trajectory_msgs moveit_commander gazebo_msgs trac_ik_python
)

## Mark scripts for installation
catkin_install_python(PROGRAMS
  grasp_block.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
