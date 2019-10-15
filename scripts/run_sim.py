#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import matplotlib.pyplot as plt
import rospy
import numpy as np

def run_experiment():
    """ Experiment to first give 200 demonstrations followed by
    reinforcement learning
    """
    saved_controller_file = 'demo_final_cases_0_1399.pkl'
    num_human_episodes = 200
    total_episodes = 1000
    save_folder = ''
    sim = Simulation() # alpha doesn't matter
    sim.add_controllers({'ddpg':{'eps':0.02}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
    case_name = 'final_cases'
    case_count = 0
    for case_number in np.random.choice(total_episodes, total_episodes, replace=False):
        if case_count < num_human_episodes:
            sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
        else:
            sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
        case_count += 1
    sim.save_simulation(save_folder)


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate', log_level=rospy.ERROR)
    run_experiment()
