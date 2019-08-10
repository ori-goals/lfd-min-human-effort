#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    #saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/similar_cases_teleop.pkl'
    sim.add_controllers({'learnt':{}, 'keypad_teleop':{}})

    case_name = 'similar_cases'
    sim.run_new_episode(case_name, 0, controller_type = 'keypad_teleop')
    sim.run_new_episode(case_name, 0, controller_type = 'learnt')
    sim.run_new_episode(case_name, 0, switching_method = 'contextual_bandit')
