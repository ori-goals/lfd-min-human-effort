import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import pandas as pd

def plot_human_cost_sliding_window(folders, method_names, num_episodes = 500, cost_failure=5.0):
    plt.close("all")
    plt.rcParams.update({'font.size': 14})

    # make sure this is correct!
    max_episodes = num_episodes
    human_demo_cost = 1.0
    failure_cost = cost_failure
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

def plot_relative_usage(folder_path, num_episodes):
    plt.close("all")
    plt.rcParams.update({'font.size': 18})

    # load a replay buffer
    file_paths = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    # make sure this is correct!
    colors = "rgbk"
    file_index = 0
    window_half_width = 15

    human_usage_array = np.zeros([len(file_paths), num_episodes])
    baseline_usage_array = np.zeros([len(file_paths), num_episodes])
    agent_usage_array = np.zeros([len(file_paths), num_episodes])
    file_path_index = 0

    for file_path in file_paths:
        file = open(file_path, 'rb')

        human_usage_total = []
        baseline_usage_total = []
        agent_usage_total = []
        data_list, _ = pickle.load(file)

        human_count = 0
        baseline_count = 0
        agent_count = 0

        for episode_index in range(num_episodes):

            human_this_window = 0.0
            baseline_this_window = 0.0
            agent_this_window = 0.0
            window_episodes = 0

            episode = data_list[episode_index]

            if 'teleop' in episode.controller_type:
                human_count += 1
            elif episode.controller_type == 'baseline':
                baseline_count += 1
            elif episode.controller_type == 'ddpg':
                agent_count += 1

            # loop through the window and tally success and failure
            for i in range((episode_index - window_half_width), (episode_index + window_half_width + 1)):

                if (i < 0) or (i >= len(data_list)):
                    continue

                episode = data_list[i]
                window_episodes += 1

                if 'teleop' in episode.controller_type:
                    human_this_window += 1.0
                elif episode.controller_type == 'baseline':
                    baseline_this_window += 1.0
                elif episode.controller_type == 'ddpg':
                    agent_this_window += 1.0


            human_usage_total.append(human_this_window/window_episodes)
            baseline_usage_total.append(baseline_this_window/window_episodes)
            agent_usage_total.append(agent_this_window/window_episodes)

        human_usage_array[file_path_index, :] = human_usage_total[0:num_episodes]
        baseline_usage_array[file_path_index, :] = baseline_usage_total[0:num_episodes]
        agent_usage_array[file_path_index, :] = agent_usage_total[0:num_episodes]

        file_path_index += 1
        print('Teleop episodes: %i, Baseline episodes: %i, Agent episodes: %i' % (human_count, baseline_count, agent_count))


    human_usage_stddevs = np.std(human_usage_array, axis=0)
    baseline_usage_stddevs = np.std(baseline_usage_array, axis=0)
    agent_usage_stddevs = np.std(agent_usage_array, axis=0)

    human_usage_means = np.mean(human_usage_array, axis=0)
    baseline_usage_means = np.mean(baseline_usage_array, axis=0)
    agent_usage_means = np.mean(agent_usage_array, axis=0)

    plt.plot(range(len(human_usage_means)), human_usage_means, "k", linewidth = 3.0,  label = "human")
    plt.fill_between(range(len(human_usage_means)), human_usage_means-human_usage_stddevs, human_usage_means+human_usage_stddevs, color = "k", alpha=.1)

    plt.plot(range(len(baseline_usage_means)), baseline_usage_means, "m", linewidth = 3.0,  label = "baseline")
    plt.fill_between(range(len(baseline_usage_means)), baseline_usage_means-baseline_usage_stddevs, baseline_usage_means+baseline_usage_stddevs, color = "m", alpha=.1)

    plt.plot(range(len(agent_usage_means)), agent_usage_means, "g", linewidth = 3.0,  label = "agent")
    plt.fill_between(range(len(agent_usage_means)), agent_usage_means-agent_usage_stddevs, agent_usage_means+agent_usage_stddevs, color = "g", alpha=.1)

    plt.xlabel("episodes")
    plt.ylabel("usage proportion")
    plt.ylim([0.0, 1.0])
    plt.legend(loc = "upper left")
    plt.show()


def all_episodes_to_single_dataframe(all_runs):
    dataframe = pd.DataFrame(columns = all_runs[0].episode_df.columns)
    for episode in all_runs:
        dataframe = pd.concat([dataframe, episode.episode_df], ignore_index=True)
    return dataframe

def plot_length_param_estimation(known_episodes_file_path, unknown_episodes_file_path):

    plt.close("all")
    plt.rcParams.update({'font.size': 20})

    known_episodes_file = open(known_episodes_file_path, 'rb')
    unknown_episodes_file = open(unknown_episodes_file_path, 'rb')

    all_runs_known, controller_save_info  = pickle.load(known_episodes_file)
    all_runs_unknown, controller_save_info  = pickle.load(unknown_episodes_file)
    known_replay_buffer = all_episodes_to_single_dataframe(all_runs_known)

    n_length_param = 50
    length_parameters = np.logspace(-2.0, 2.0, num=n_length_param)
    print(length_parameters)
    log_likelihoods = np.zeros(len(length_parameters))

    for length_param_index in range(len(length_parameters)):
        length_param = length_parameters[length_param_index]
        log_likelihood  = 0.0
        print(length_param_index)

        # loop through each new case
        for episode in all_runs_unknown:
            init_state = np.array(episode.episode_df.loc[0]["state"])

            # initialise arrays to keep track of the stuff. NOTE: using arrays of ones defines the prior
            alpha = 0.2
            beta = 0.2

            # loop through all entries in the replay buffer
            for index, replay_buffer_row in known_replay_buffer.iterrows():
                replay_buffer_state = np.array(replay_buffer_row['state'])
                state_delta = init_state - replay_buffer_state

                # calculate the difference between the two states
                weight = np.exp(-1.0*(np.linalg.norm(state_delta/length_param))**2)

                # increment alpha for success and beta for failure
                if int(round(replay_buffer_row["binary_return"])) == 1:
                    alpha = alpha + weight
                else:
                    beta = beta + weight

            p_current_state_success = alpha/(alpha + beta)

            # nan if too far away from any existing experience. this is equivalent to no knowledge so we shall call it 0.5
            if np.isnan(p_current_state_success):
                p_current_state_success = 0.5

            # prevent silly numbers blowing things up
            if p_current_state_success < 0.000001:
                p_current_state_success = 0.000001
            elif p_current_state_success > 0.999999:
                p_current_state_success = 0.999999

            # if we succeeded the likelihood is the success probability
            if episode.result:
                log_likelihood = log_likelihood + np.log(p_current_state_success)
            else:
                log_likelihood = log_likelihood + np.log(1 - p_current_state_success)

        log_likelihoods[length_param_index] = log_likelihood
    fig, ax = plt.subplots()
    ax.semilogx(length_parameters, log_likelihoods, "k", linewidth = 3.0)
    ax.semilogx(length_parameters, log_likelihoods, "kx", markersize = 6.0)
    ax.grid()
    plt.xlabel("length parameter (m)")
    plt.ylabel("log likelihood")
    plt.show()
