#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import matplotlib.pyplot as plt
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate', log_level=rospy.ERROR)
    sim = Simulation()
    #saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/similar_cases_teleop.pkl'
    sim.add_controllers({'ddpg':{}})

    case_name = 'lfd_rl_aug10'
    dense_rewards = []
    results = []
    for i in range(500):
        episode, dense_reward = sim.run_new_episode(case_name, 5, controller_type = 'ddpg')
        dense_rewards.append(dense_reward)
        if episode.result:
            results.append(1)
        else:
            results.append(0)
        print(i)
        if i % 20 == 0:

            plt.figure()
            plt.plot(range(len(dense_rewards)), dense_rewards)
            plt.savefig('rewards.png')
            plt.figure()
            plt.plot(range(len(dense_rewards)), results, 'kx')
            plt.savefig('success.png')
