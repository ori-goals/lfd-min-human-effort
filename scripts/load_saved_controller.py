#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')


    for i in range(0, 3):
        sim = Simulation()
        saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/tuning_problem_domain/high_variation_aug13/demos/2019-08-13-19-39_joystick_teleop242.pkl'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'high_variation_aug13'

        for case_number in range(0, 800):
            if case_number < 240:
                sim.run_new_episode(case_name, case_number/2, controller_type = 'joystick_teleop')
            else:
                sim.run_new_episode(case_name, case_number/2, controller_type = 'ddpg')
        sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/tuning_problem_domain/high_variation_aug13/demos_then_rl')
