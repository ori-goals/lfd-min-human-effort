import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np

def plot_human_cost_sliding_window(folders, method_names, max_episodes = 500):
    plt.close("all")
    plt.rcParams.update({'font.size': 14})

    # make sure this is correct!
    human_demo_cost = 1.0
    failure_cost = 3.0
    colors = "rkcm"
    window_half_width = 20

    for folder_ind in range(len(folders)):
        folder_path = folders[folder_ind]
        labelled_flag = False
        files = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
        cost_array = np.zeros([len(files), max_episodes])

        for file_path_index in range(len(files)):
            file_path = files[file_path_index]
            file = open(file_path, 'rb')
            cost = 0.0
            sliding_window_mean_costs = []
            data_list, controller_save_info = pickle.load(file)

            for episode_index in range(len(data_list)):
                window_cost = 0.0
                window_episodes = 0

                # loop through the window and tally success and failure
                for i in range((episode_index - window_half_width), (episode_index + window_half_width + 1)):
                    if (i < 0) or (i >= len(data_list)):
                        continue
                    episode = data_list[i]
                    window_episodes += 1

                    if "teleop" in episode.controller_type:
                        window_cost = window_cost + human_demo_cost
                    elif not episode.result:
                        window_cost = window_cost + failure_cost
                mean_cost = window_cost/window_episodes
                sliding_window_mean_costs.append(mean_cost)
            cost_array[file_path_index, :] = sliding_window_mean_costs[0:max_episodes]
            cost_stddevs = np.std(cost_array, axis=0)
            cost_means = np.mean(cost_array, axis=0)
            file_path_index += 1

        plt.plot(range(len(cost_means)), cost_means, colors[folder_ind], linewidth = 3.0,  label = method_names[folder_ind])
        plt.fill_between(range(len(cost_means)), cost_means-cost_stddevs, cost_means+cost_stddevs, color = colors[folder_ind], alpha=.1)
    plt.xlabel("episodes")
    plt.ylabel("cost per episode")
    plt.ylim([0.0, failure_cost+1.0])
    plt.xlim([0.0, float(max_episodes)])
    plt.legend()
    plt.title("sliding window of mean cost per episode")
    plt.show()

def plot_ddpg_success_rate(folders, method_names, max_episodes = 500):
    plt.close("all")
    plt.rcParams.update({'font.size': 14})

    # make sure this is correct!
    human_demo_cost = 1.0
    failure_cost = 3.0
    colors = "rkcm"
    window_half_width = 20

    for folder_ind in range(len(folders)):
        folder_path = folders[folder_ind]
        labelled_flag = False
        files = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
        success_array = np.zeros([len(files), max_episodes])

        for file_path_index in range(len(files)):
            file_path = files[file_path_index]
            file = open(file_path, 'rb')
            successes = 0.0
            sliding_window_mean_success_rate = []
            data_list, controller_save_info = pickle.load(file)

            for episode_index in range(len(data_list)):
                window_successes = 0.0
                window_episodes = 0.0

                if "teleop" in data_list[episode_index].controller_type:
                    continue

                # loop through the window and tally success and failure
                for i in range((episode_index - window_half_width), (episode_index + window_half_width + 1)):
                    if (i < 0) or (i >= len(data_list)):
                        continue
                    episode = data_list[i]
                    if "teleop" in episode.controller_type:
                        continue
                    else:
                        window_episodes += 1
                        if episode.result:
                            window_successes += 1.0
                mean_success_rate = window_successes/window_episodes
                sliding_window_mean_success_rate.append(mean_success_rate)
            success_array[file_path_index, :] = sliding_window_mean_success_rate[0:max_episodes]
            success_stddevs = np.std(success_array, axis=0)
            success_means = np.mean(success_array, axis=0)
            file_path_index += 1

        plt.plot(range(len(success_means)), success_means, colors[folder_ind], linewidth = 3.0,  label = method_names[folder_ind])
        plt.fill_between(range(len(success_means)), success_means-success_stddevs, success_means+success_stddevs, color = colors[folder_ind], alpha=.1)
    plt.xlabel("episodes")
    plt.ylabel("success rate sliding window")
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, float(max_episodes)])
    plt.legend()
    plt.title("sliding window of success rate for ddpg policy")
    plt.show()
