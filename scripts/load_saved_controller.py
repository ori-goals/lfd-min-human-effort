#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/load_teleop_test.pkl'
    sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
    case_name = 'lfd_rl_aug10'

    for case_number in range(5):
        sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')

    for case_number in range(50, 365):
        sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')

    sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/lfd_then_rl')
