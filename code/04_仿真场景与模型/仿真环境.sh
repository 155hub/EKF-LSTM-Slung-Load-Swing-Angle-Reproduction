#!/usr/bin/env bash
# Source from WSL Ubuntu 20.04; deliberately separate from ML package paths.
source /opt/ros/noetic/setup.bash
px4_dir=/home/ubuntu/PX4-Autopilot
repo_dir=/mnt/d/RPi_Deployment/EKF-LSTM_for_Swing_Angle_Estimation-main
export ROS_MASTER_URI=http://127.0.0.1:11321
export ROS_IP=127.0.0.1
export GAZEBO_MASTER_URI=http://127.0.0.1:11351
export ROS_PACKAGE_PATH="$px4_dir:$px4_dir/Tools/simulation/gazebo-classic/sitl_gazebo-classic:${ROS_PACKAGE_PATH:-}"
export GAZEBO_PLUGIN_PATH="$px4_dir/build/px4_sitl_default/build_gazebo-classic:${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_MODEL_PATH="$repo_dir/simulation/models:$px4_dir/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:${GAZEBO_MODEL_PATH:-}"
export LD_LIBRARY_PATH="$px4_dir/build/px4_sitl_default/build_gazebo-classic:${LD_LIBRARY_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=''
export PYTHONUNBUFFERED=1
