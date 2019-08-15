#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy
from learn_to_manipulate.utils import join_demo_files


if __name__ == "__main__" :
    file1 = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/demonstrations/demo_final_cases_0_499.pkl'
    file2 = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/demonstrations/demo_final_cases_500_999.pkl'
    controller_type = 'joystick_teleop'
    save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/demonstrations'
    join_demo_files(file1, file2, controller_type, save_folder)
