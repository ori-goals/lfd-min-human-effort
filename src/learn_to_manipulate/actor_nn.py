import rospy, os, time, pickle, math, copy, torch
import numpy as np
import pandas as pd

class Weights():
    def __init__(self, init_size):
        self.d_in = 52
        self.h1 = 6
        self.h2 = 4
        self.d_out = 2
        weight_init_size = init_size

        device = torch.device('cpu')
        self.w1 = torch.tensor(np.random.randn(self.d_in, self.h1)*weight_init_size, device=device, requires_grad=True, dtype=torch.float64)
        self.w2 = torch.tensor(np.random.randn(self.h1, self.h2)*weight_init_size, device=device, requires_grad=True, dtype=torch.float64)
        self.w3 = torch.tensor(np.random.randn(self.h2, self.d_out)*weight_init_size, device=device, requires_grad=True, dtype=torch.float64)

class Prior():
    def __init__(self, planner_alpha, planner_beta, agent_alpha, agent_beta):
        self.planner_alpha = planner_alpha
        self.planner_beta = planner_beta
        self.agent_alpha = agent_alpha
        self.agent_beta = agent_beta

class ActorNN():
    def __init__(self, nominal_means, nominal_sigma_exps):
        self.nominal_means = nominal_means
        self.nominal_sigma_exps = nominal_sigma_exps
        self.weights_dx = Weights(0.05)
        self.weights_dy = Weights(0.05)
        self.grad_max = 20.0
        self.prior = Prior(0.261, 0.140, 0.260, 0.141)

    def get_action(self, x_input):
        x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.view(1, len(x_input))

        dx_output = self.forward_pass(x_input, self.weights_dx)
        dy_output = self.forward_pass(x_input, self.weights_dy)

        dx_output = dx_output.data.numpy().flatten()
        dy_output = dy_output.data.numpy().flatten()

        action_output = np.concatenate((dx_output, dy_output)) + np.array([self.nominal_means[0], self.nominal_sigma_exps[0], self.nominal_means[1], self.nominal_sigma_exps[1]])
        means = [dx_output[0]+self.nominal_means[0], dy_output[0]+self.nominal_means[1]]
        sigmas = math.e**np.array([dx_output[1]+self.nominal_sigma_exps[0], dy_output[1]+self.nominal_sigma_exps[1]])
        dx = np.random.normal(means[0], sigmas[0])
        dy = np.random.normal(means[1], sigmas[1])
        print("dx: %.4f (mean: %.4f, sigma: %.4f), dy: %.4f (mean: %.4f, sigma: %.4f)"
             % (dx, means[0], sigmas[0], dy, means[1], sigmas[1]))
        return [dx, dy]

    def forward_pass(self, x_input, weights):
        return x_input.mm(weights.w1).tanh().mm(weights.w2).tanh().mm(weights.w3)

    def get_loss(self, action_taken, nn_output, advantage, nominal_mean, nominal_sigma_exp):
        action1_likelihood = 1/(torch.sqrt(2.0*math.pi*(math.e**(nn_output[0][1] + nominal_sigma_exp))**2))*math.e**(-(action_taken[0][0] - (nn_output[0][0] + nominal_mean))**2/(2.0*(math.e**(nn_output[0][1] + nominal_sigma_exp))**2))
        loss = -1.0*advantage*(torch.log(action1_likelihood))
        return loss

    def ac_onestep_update(self, replay_buffer_rl, experience_df, learning_rates, td_max, length_scale):
        dx_taken = torch.tensor([[experience_df['delta_x']]])
        dy_taken = torch.tensor([[experience_df['delta_y']]])
        current_state = np.array(experience_df["laser_state"])
        next_state = np.array(experience_df["next_laser_state"])

        current_value, sigma, alpha, beta = get_learner_state_value(current_state, replay_buffer_rl, length_scale, self.prior)
        next_value, sigma, alpha, beta = get_learner_state_value(next_state, replay_buffer_rl, length_scale, self.prior)

        # if the reward is not -1 use the reward else use the td error
        if int(round(experience_df['reward'])) != -1:
            onestep_td_error = experience_df['reward'] - current_value
        else:
            onestep_td_error = next_value - current_value

        # enforce limit on maximum error
        if onestep_td_error > td_max:
            onestep_td_error = td_max
        elif onestep_td_error < -1.0*td_max:
            onestep_td_error = -1.0*td_max

        state_input = torch.tensor(np.array(experience_df["laser_state"]), dtype=torch.float64)
        state_input = state_input.view(1,len(current_state))
        nn_dx_output = self.forward_pass(state_input, self.weights_dx)
        nn_dy_output = self.forward_pass(state_input, self.weights_dy)

        print("\nvalue before: %.3f, value after: %.3f" % (current_value, next_value))
        mean_forward = nn_dx_output[0][0] + self.nominal_means[0]
        mean_left = nn_dy_output[0][0] + self.nominal_means[1]
        print("mean forward before: %.5f, mean left before: %.5f" % (mean_forward, mean_left, mean_yaw))
        print("forward taken: %.3f, left taken: %.3f" % (experience_df['delta_x'], experience_df['delta_y']))

        steps = 5
        for i in range(steps):
            nn_dx_output = self.forward_pass(state_input, self.weights_dx)
            nn_dy_output= self.forward_pass(state_input, self.weights_dy)
            nn_dyaw_output= self.forward_pass(state_input, self.weights_dyaw)
            loss_dx = self.get_loss(dx_taken, nn_dx_output, onestep_td_error, self.nominal_means[0], self.nominal_sigma_exps[0])
            loss_dy = self.get_loss(dy_taken, nn_dy_output, onestep_td_error, self.nominal_means[1], self.nominal_sigma_exps[1])
            loss_dx.backward()
            loss_dy.backward()

            with torch.no_grad():
                self.weights_update(self.weights_dx, learning_rates[0])
                self.weights_update(self.weights_dy, learning_rates[1])

        nn_dx_output = self.forward_pass(state_input, self.weights_dx)
        nn_dy_output = self.forward_pass(state_input, self.weights_dy)

        mean_forward = nn_dx_output[0][0] + self.nominal_means[0]
        mean_left = nn_dy_output[0][0] + self.nominal_means[1]
        print("mean forwardafter: %.5f, mean left after: %.5f" % (mean_forward, mean_left))

    def weights_update(self, weights, learning_rate):
        weights.w1.grad[torch.isnan(weights.w1.grad)] = 0.0
        weights.w2.grad[torch.isnan(weights.w2.grad)] = 0.0
        weights.w3.grad[torch.isnan(weights.w3.grad)] = 0.0

        weights.w1.grad = torch.clamp(weights.w1.grad, min=-1.0*self.grad_max, max=self.grad_max)
        weights.w2.grad = torch.clamp(weights.w2.grad, min=-1.0*self.grad_max, max=self.grad_max)
        weights.w3.grad = torch.clamp(weights.w3.grad, min=-1.0*self.grad_max, max=self.grad_max)

        weights.w1 -= learning_rate * weights.w1.grad
        weights.w2 -= learning_rate * weights.w2.grad
        weights.w3 -= learning_rate * weights.w3.grad

        weights.w1.grad.zero_()
        weights.w2.grad.zero_()
        weights.w3.grad.zero_()

    def ac_update(self, replay_buffer_rl, learning_rates, update_steps, td_max, length_scale):

        # make the appropriate number of updates
        for update in range(0, update_steps):

            # randomly sample an experience from the replay buffer with preference for newer experiences
            random_experience_index = int(round(np.random.exponential(100)))
            while random_experience_index >= len(replay_buffer_rl):
                random_experience_index = int(round(np.random.exponential(100)))

            # extract experience
            experience_df = replay_buffer_rl.iloc[random_experience_index]
            dx_taken = torch.tensor([[experience_df['delta_x']]])
            dy_taken = torch.tensor([[experience_df['delta_y']]])
            current_state = np.array(experience_df["laser_state"])
            next_state = np.array(experience_df["next_laser_state"])

            # calcualte one step undiscounted td error
            current_value, sigma, alpha, beta = get_learner_state_value(current_state, replay_buffer_rl, length_scale, self.prior)


            # if the reward is not -1 use the reward else use the td error
            if int(round(experience_df['reward'])) != -1:
                onestep_td_error = experience_df['reward'] - current_value
            else:
                next_value, sigma, alpha, beta = get_learner_state_value(next_state, replay_buffer_rl, length_scale, self.prior)
                onestep_td_error = next_value - current_value

            # enforce limit on maximum error
            if onestep_td_error > td_max:
                onestep_td_error = td_max
            elif onestep_td_error < -1.0*td_max:
                onestep_td_error = -1.0*td_max

            state_input = torch.tensor(np.array(experience_df["laser_state"]), dtype=torch.float64)
            state_input = state_input.view(1, len(current_state))
            nn_dx_output = self.forward_pass(state_input, self.weights_dx)
            nn_dy_output = self.forward_pass(state_input, self.weights_dy)

            # compute the loss (the log of the likelihood times the return in the case of policy gradient)
            loss_dx = self.get_loss(dx_taken, nn_dx_output, onestep_td_error, self.nominal_means[0], self.nominal_sigma_exps[0])
            loss_dy = self.get_loss(dy_taken, nn_dy_output, onestep_td_error, self.nominal_means[1], self.nominal_sigma_exps[1])
            loss_dx.backward()
            loss_dy.backward()

            with torch.no_grad():
                self.weights_update(self.weights_dx, learning_rates[0])
                self.weights_update(self.weights_dy, learning_rates[1])


    def bc_update(self, replay_buffer_demos, learning_rates, update_steps):

        # make the appropriate number of updates
        for update in range(0, update_steps):

            # randomly sample an experience from the replay buffer with preference for newer experiences
            random_experience_index = int(round(np.random.exponential(200)))
            while random_experience_index >= len(replay_buffer_demos):
                random_experience_index = int(round(np.random.exponential(200)))

            # extract experience
            experience_df = replay_buffer_demos.iloc[random_experience_index]
            dx_taken = torch.tensor([[experience_df['delta_x']]])
            dy_taken = torch.tensor([[experience_df['delta_y']]])
            current_state = np.array(experience_df["laser_state"])
            next_state = np.array(experience_df["next_laser_state"])

            # calcualte one step undiscounted td error
            advantage = 1.0

            state_input = torch.tensor(np.array(experience_df["laser_state"]), dtype=torch.float64)
            state_input = state_input.view(1, len(current_state))
            nn_dx_output = self.forward_pass(state_input, self.weights_dx)
            nn_dy_output = self.forward_pass(state_input, self.weights_dy)

            # compute the loss (the log of the likelihood times the return in the case of policy gradient)
            loss_dx = self.get_loss(dx_taken, nn_dx_output, advantage, self.nominal_means[0], self.nominal_sigma_exps[0])
            loss_dy = self.get_loss(dy_taken, nn_dy_output, advantage, self.nominal_means[1], self.nominal_sigma_exps[1])
            loss_dx.backward()
            loss_dy.backward()

            with torch.no_grad():
                self.weights_update(self.weights_dx, learning_rates[0])
                self.weights_update(self.weights_dy, learning_rates[1])
