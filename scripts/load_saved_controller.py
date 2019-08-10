#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/lfd_rl_aug10_first50_teleop.pkl'
    sim.add_controllers({'learnt':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'keypad_teleop'}})
    case_name = 'lfd_rl_aug10'

    for case_number in range(50):
        sim.run_new_episode(case_name, case_number, controller_type = 'keypad_teleop')

    for case_number in range(50, 60):
        sim.run_new_episode(case_name, case_number, controller_type = 'learnt')

    sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/lfd_then_rl')
