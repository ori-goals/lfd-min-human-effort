#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')


    for i in range(0, 1):
        sim = Simulation()
        saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/tuning_problem_domain/final_cases/first_200_demos/2019-08-13-21-01_joystick_teleop182.pkl'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'final_cases'

        for case_number in range(0, 145):
            if case_number < 70:
                sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
            else:
                sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
            if (case_number - 1) % 20 == 0:
                sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/baseline_controllers')
