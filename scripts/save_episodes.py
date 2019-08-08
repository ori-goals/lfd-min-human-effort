#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    sim.add_controllers({'learnt':{}, 'keypad_teleop':{}})

    case_name = 'test_cases'
    sim.run_new_episode(case_name, 1, controller_type = 'keypad_teleop')
    sim.run_new_episode(case_name, 2, controller_type = 'keypad_teleop')
    sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests')
