#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy
from learn_to_manipulate.utils import join_demo_files


if __name__ == "__main__" :
    file1 = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_999.pkl'
    file2 = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_1000_1399.pkl'
    controller_type = 'joystick_teleop'
    save_folder = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos'
    join_demo_files(file1, file2, controller_type, save_folder)
