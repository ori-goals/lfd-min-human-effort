#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')


    for i in range(0, 5):
        sim = Simulation()
        saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/ddpg_bc_testing/demos_0_to_49/rl_attempt_aug11_first50.pkl'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'rl_attempt_aug11'

        for case_number in range(50, 250):
            sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
        sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/ddpg_bc_testing/rl_only')
