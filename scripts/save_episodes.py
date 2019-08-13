#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    sim.add_controllers({'joystick_teleop':{}})

    case_name = 'lfd_rl_aug10'
    for case_number in range(5):
        sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
    sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests')
