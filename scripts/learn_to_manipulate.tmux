#!/bin/bash

SESSION=marc

# Set this variable in order to have a development workspace sourced, surplus/instead of the .bashrc one
DEVELOPMENT_WS=/home/marcrigter/ori_goals_ws/devel/setup.bash
#DEVELOPMENT_WS=/opt/ros/kinetic/setup.bash
_SRC_ENV="tmux send-keys source Space $DEVELOPMENT_WS C-m "

tmux -2 new-session -d -s $SESSION
tmux new-window -t $SESSION:0 -n 'ros'
tmux new-window -t $SESSION:1 -n 'sim'
tmux new-window -t $SESSION:2 -n 'planner'
tmux new-window -t $SESSION:3 -n 'script'


tmux select-window -t $SESSION:1
[ -f $DEVELOPMENT_WS ] && `$_SRC_ENV`
tmux send-keys "roslaunch ur_gazebo ur5_joint_limited.launch" C-m

sleep 15

tmux select-window -t $SESSION:2
[ -f $DEVELOPMENT_WS ] && `$_SRC_ENV`
tmux send-keys "roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true limited:=true " C-m

sleep 10

tmux select-window -t $SESSION:3
[ -f $DEVELOPMENT_WS ] && `$_SRC_ENV`
tmux send-keys "rosrun learn_to_manipulate run_mab.py" C-m


# Set default window
tmux select-window -t $SESSION:0

# Attach to session
tmux -2 attach-session -t $SESSION

