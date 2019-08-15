#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    sim.add_controllers({'joystick_teleop':{}})

    case_name = 'final_cases'
    for case_number in range(500, 1030):
        sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
        if (case_number > 10) and (case_number+1) % 20 == 0:
            sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/demonstrations')
