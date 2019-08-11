#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import matplotlib.pyplot as plt
import rospy

def subplot(contr):
    r = list(zip(*contr.plot_average_rewards))
    r2 = list(zip(*contr.plot_reward))
    p = list(zip(*contr.plot_policy))
    q = list(zip(*contr.plot_q))
    s = list(zip(*contr.plot_steps))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'k', linewidth = 2.5) #row=0, col=0
    ax[0, 0].plot(list(r2[1]), list(r2[0]), 'r') #row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b') #row=1, col=0
    ax[0, 1].plot(list(q[1]), list(q[0]), 'g') #row=0, col=1
    ax[1, 1].plot(list(s[1]), list(s[0]), 'k') #row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[0, 1].title.set_text('Q loss')
    ax[1, 1].title.set_text('Max steps')
    plt.savefig('run.png')


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate', log_level=rospy.ERROR)
    sim = Simulation()
    #saved_controller_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/similar_cases_teleop.pkl'
    sim.add_controllers({'ddpg':{}})

    case_name = 'rl_attempt_aug11'
    dense_rewards = []
    results = []
    for i in range(2000):
        episode, dense_reward = sim.run_new_episode(case_name, i, controller_type = 'ddpg')

        print(i)
        if (i-14) % 15 == 0:    # print every print_every episodes
            subplot(sim.controllers[0])
