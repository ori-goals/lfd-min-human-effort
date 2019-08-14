#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')


    for i in range(0, 2):
        sim = Simulation()
        baseline_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/baseline_controllers/2019-08-14-14-35_joystick_teleop80_ddpg42.pkl'
        sim.add_controllers({'baseline':{'file':baseline_file}})
        case_name = 'final_cases'

        for case_number in range(400, 500):
            sim.run_new_episode(case_name, case_number, controller_type = 'baseline')
            if case_number == 499:
                sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline')
