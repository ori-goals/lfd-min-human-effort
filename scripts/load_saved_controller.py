#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/2019-08-08-15-15_keypad_teleop2_learnt0.pkl'
    sim.add_controllers({'learnt':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'keypad_teleop'}})
    case_number = 1
    case_name = 'test_cases'
    sim.run_new_episode(case_name, case_number, controller_type = 'keypad_teleop')
