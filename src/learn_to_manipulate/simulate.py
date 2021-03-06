#!/usr/bin/env python
import rospy
import pickle
import time
import random
import csv
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
import numpy as np
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from learn_to_manipulate.controller import *

class Simulation(object):
    def __init__(self, demo_cost = 0.2, alpha=0.0, delta_tau=0.0, max_demos = float('inf'), file_path = None):
        self.initial_pose_pub = rospy.Publisher('laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)
        self.block_width = 0.04
        self.goal_width_x = 0.001
        self.goal_centre_x = 0.65
        self.all_runs = []
        self.demo_cost = demo_cost
        self.success_reward = 1.0
        self.failure_reward = 0.0
        self.alpha = alpha
        self.episode_number = 0
        self.max_demos = max_demos
        self.block_num = 0
        self.ncb_window = []
        self.baseline_window = []
        self.ncb_window_size = 50
        self.tau = 0
        self.delta_tau = delta_tau

    def run_new_episode(self, case_name, case_number, switching_method = None, controller_type = None):
        """ Choose a controller and run an episode of simulation.
        """
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        # check block has been deleted
        self.controllers[list(self.controllers.keys())[0]].set_arm_initial()
        self.spawn_table()
        self.spawn_block(case_name, case_number)
        controller = self.choose_controller(switching_method, controller_type)
	rospy.sleep(0.2)
        episode, dense_reward = controller.run_episode(case_name, case_number)
        self.all_runs.append(episode)
        self.ncb_window.insert(0, {'controller_type': episode.controller_type, 'result':episode.result})
        if episode.controller_type == 'baseline':
            self.baseline_window.insert(0, {'controller_type': episode.controller_type, 'result':episode.result})
        self.episode_number += 1
        #print('RL buffer size: %g, Demo buffer size: %g' % (self.controllers['ddpg'].replay_buffer.count(), self.controllers['joystick_teleop'].replay_buffer.count()))
        return episode, dense_reward

    @classmethod
    def load_simulation(cls, file_path):
        """ Load controllers from a previous simulation.
        """
        sim =  cls()

        file = open(file_path,"rb")
        all_runs, controller_save_info = pickle.load(file)
        sim.all_runs = all_runs

        controller_list = {}
        for save_info in controller_save_info:
            if save_info['type'] == 'ddpg':
                controller_list[save_info['type']] = DDPGController.from_save_info(sim, save_info)
            elif save_info['type'] == 'keypad_teleop':
                controller_list[save_info['type']] = KeypadController.from_save_info(sim, save_info)
            elif save_info['type'] == 'joystick_teleop':
                controller_list[save_info['type']] = JoystickController.from_save_info(sim, save_info)
            elif save_info['type'] == 'saved_teleop':
                controller_list[save_info['type']] = SavedTeleopController(file_path, save_info['type'])
        sim.controllers = controller_list
        return sim

    @classmethod
    def generate_cases(cls, case_name, number, spec = None):
        """ Randomly generate positions for the block.
        """
        if spec is None:
            spec = {'min_y':-0.1, 'max_y':0.1, 'min_x':0.5, 'max_x':0.65,
                'min_angle_deg':-60, 'max_angle_deg':60,
                'block_names':['block_30']}

        my_path = os.path.abspath(os.path.dirname(__file__))
        save_path = os.path.join(my_path, '../../config/cases/' + case_name + '.csv')
        with open(save_path, mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            for case in range(number):
                x = np.random.uniform(spec['min_x'], spec['max_x'])
                y = np.random.uniform(spec['min_y'], spec['max_y'])
                angle_deg = np.random.uniform(spec['min_angle_deg'], spec['max_angle_deg'])
                block_name = random.choice(spec['block_names'])
                writer.writerow([case, x, y, angle_deg, block_name])

    def save_simulation(self, folder):
        fname = time.strftime("%Y-%m-%d-%H-%M")
        for type, contr in self.controllers.items():
            fname += '_' + contr.type + str(contr.episode_number)
        new_file_path = folder + '/' + fname + '.pkl'

        controller_save_info = []
        for type, contr in self.controllers.items():
            controller_save_info.append(contr.get_save_info())

        with open(new_file_path, 'w') as f:
            pickle.dump([self.all_runs, controller_save_info], f)
            f.close

    def add_controllers(self, type_dict):
        controller_dict = {}
        for type in type_dict:
            mydict = type_dict[type]
            if type == 'learnt':
                controller_dict[type] = LearntController(self)
            elif type == 'keypad_teleop':
                controller_dict[type] = KeypadController(self)
            elif type == 'joystick_teleop':
                controller_dict[type] = JoystickController(self)
            elif type == 'ddpg':
                controller_dict[type] = DDPGController(self)
            elif type == 'saved_teleop':
                  my_path = os.path.abspath(os.path.dirname(__file__))
                  load_path = os.path.join(my_path, '../../config/demos/' + mydict['file'])
                  controller_dict[mydict['type']] = SavedTeleopController(self, load_path, mydict['type'])
            elif type == 'baseline':
                controller_dict['baseline'] = SavedDDPGAgent(self, mydict['file'])
        self.controllers = controller_dict


    def choose_controller(self, switching_method = None, requested_type = None):
        """ Choose controller for the episode. If the controller to use
        is specified, uses that controller. Otherwise chooses controller acccording
        to the specified switching method.
        """

        # if a controller type has been specified
        if requested_type is not None:
            for type, controller in self.controllers.items():
                if controller.type == requested_type:
                    return controller
            sys.exit('Error: requested controller does not exist')

        if switching_method == 'contextual_bandit':
            max_ucb = 0.0
            for type, controller in self.controllers.items():
                confidence, sigma = controller.get_controller_confidence()
                ucb = (confidence + self.alpha*sigma)*self.success_reward
                if 'teleop' in controller.type:
                    ucb -= self.demo_cost
                if ucb > max_ucb:
                    chosen_controller = controller
                    max_ucb = ucb
                #print('%15s: conf %.4f, sigma %.4f, ucb %.4f' %
                    #(controller.type, confidence, sigma, ucb))
            if 'teleop' in chosen_controller.type:
                if chosen_controller.episode_number >= self.max_demos:
                    chosen_controller = self.controllers['ddpg']
            return chosen_controller

        elif switching_method == 'non_contextual_bandit':
            self.ncb_window = self.ncb_window[0:self.ncb_window_size]
            max_ucb = 0.
            for type, controller in self.controllers.items():
                successes = 0.
                failures = 0.
                if type == 'baseline':
                    window = self.baseline_window
                else:
                    window = self.ncb_window
                for entry in window:
                    if entry['controller_type'] == type:
                        if entry['result']:
                            successes += 1.
                        else:
                            failures += 1.
                alpha = controller.experience.prior_alpha + successes
                beta = controller.experience.prior_beta + failures
                mean = alpha/(alpha + beta)
                var = alpha*beta/((alpha+beta)**2*(alpha+beta+1.0))
                sigma = np.sqrt(var)

                if 'teleop' in type:
                    ucb = self.success_reward - self.demo_cost
                else:
                    ucb = (mean + self.alpha*sigma)*self.success_reward
                print('%15s: conf %.4f, sigma %.4f, ucb %.4f' %
                    (type, mean, sigma, ucb))
                if ucb > max_ucb:
                    chosen_controller = controller
                    max_ucb = ucb
            return chosen_controller
        elif switching_method == 'softmax':
            self.ncb_window = self.ncb_window[0:self.ncb_window_size]
            types = []
            vals = []
            for type, controller in self.controllers.items():
                successes = 0.
                failures = 0.
                if type == 'baseline':
                    window = self.baseline_window
                else:
                    window = self.ncb_window
                for entry in window:
                    if entry['controller_type'] == type:
                        if entry['result']:
                            successes += 1.
                        else:
                            failures += 1.
                mean = successes/(successes+failures+1e-6)
                if 'teleop' in type:
                    val = self.success_reward - self.demo_cost
                else:
                    val = (mean)*self.success_reward
                types.append(type)
                vals.append(np.exp(self.tau*val))
            probs = np.array(vals)/np.sum(vals)
            type = np.random.choice(types, p = probs)
            print('Tau = %.4f' % (self.tau))
            print(types)
            print(probs)
            print('\n')
            self.tau += self.delta_tau
            return self.controllers[type]

    def spawn_table(self):
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world = get_world_properties_prox()

        if 'table' not in world.model_names:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "../../models/table1/model.sdf")
            f = open(path,'r')
            table_sdf = f.read()

            table_initial_pose = Pose()
            table_initial_pose.position.x = 0.54
            table_initial_pose.position.y = 0.0
            table_initial_pose.position.z = 0.2

            path = os.path.join(my_path, "../../models/goal/model.sdf")
            f = open(path,'r')
            goal_sdf = f.read()

            self.goal_pose = Pose()
            self.goal_pose.position.x = self.goal_centre_x
            self.goal_pose.position.y = 0.0
            self.goal_pose.position.z = 0.40

            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox("table", table_sdf, "simulation", table_initial_pose, "world")
            spawn_model_prox("goal", goal_sdf, "simulation", self.goal_pose, "world")

    def delete_block(self):
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world = get_world_properties_prox()

        rospy.wait_for_service('/gazebo/delete_model')
        delete_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        for name in world.model_names:
            if 'block' in name:
                succeeded = False
                while not succeeded:
                    delete_model_prox(name)
                    rospy.sleep(0.5)
                    world = get_world_properties_prox()
                    if (name not in world.model_names):
                        succeeded = True
                    else:
                        print('Failed to delete block')


    def spawn_block(self, case_name, case_number):
        self.delete_block()

        # find the correct case from the file
        my_path = os.path.abspath(os.path.dirname(__file__))
        load_path = os.path.join(my_path, '../../config/cases/' + case_name + '.csv')
        file = open(load_path, "r")
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if int(row[0]) == case_number:
                x = float(row[1])
                y = float(row[2])
                angle_deg = float(row[3])
                block_name = row[4]
                break

        self.block_length = float("0." + block_name.split('_')[1])
        model_path = os.path.join(my_path, "../../models/" + block_name + "/model.sdf")
        f = open(model_path,'r')
        sdf = f.read()

        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = 0.43

        initial_pose.orientation.z = np.sin(np.radians(angle_deg)/2)
        initial_pose.orientation.w = np.cos(np.radians(angle_deg)/2)

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        succeeded = False
        while not succeeded:
            spawn_model_prox("block", sdf, "simulation", initial_pose, "world")
            rospy.sleep(0.5)
            resp = get_model_state_prox("block",'')
            if (resp.success):
                succeeded = True
            else:
                print('Failed to spawn block')
