#!/usr/bin/env python
import rospy
import hsrb_interface
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
    def __init__(self, file_path = None):
        self.initial_pose_pub = rospy.Publisher('laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)
        self.block_width = 0.04
        self.goal_width_x = 0.001
        self.goal_centre_x = 0.7
        self.all_runs = []
        self.demo_cost = 0.3
        self.success_reward = 1.0
        self.failure_reward = 0.0
        self.alpha = 0.5

    def run_new_episode(self, case_name, case_number, switching_method = None, controller_type = None):
        self.controllers[0].set_arm_initial()
        self.spawn_table()
        self.spawn_block(case_name, case_number)
        controller = self.choose_controller(switching_method, controller_type)
        episode = controller.run_episode(case_name, case_number)
        self.delete_block()
        self.all_runs.append(episode)

    @classmethod
    def load_simulation(cls, file_path):
        sim =  cls()

        file = open(file_path,"rb")
        all_runs, controller_save_info = pickle.load(file)
        sim.all_runs = all_runs

        controller_list = []
        for save_info in controller_save_info:
            if save_info['type'] == 'learnt':
                controller_list.append(LearntController.from_save_info(sim, save_info))
            elif save_info['type'] == 'keypad_teleop':
                controller_list.append(KeypadController.from_save_info(sim, save_info))
            elif save_info['type'] == 'saved_teleop':
                pass
        sim.controllers = controller_list
        return sim

    @classmethod
    def generate_cases(cls, case_name, number, spec = None):
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
        for contr in self.controllers:
            fname += '_' + contr.type + str(contr.episode_count)
        new_file_path = folder + '/' + fname + '.pkl'

        controller_save_info = []
        for contr in self.controllers:
            controller_save_info.append(contr.get_save_info())

        with open(new_file_path, 'w') as f:
            pickle.dump([self.all_runs, controller_save_info], f)
            f.close

    def add_controllers(self, type_dict):
        controller_list = []
        for type in type_dict:
            if type == 'learnt':
                controller_list.append(LearntController(self))
            elif type == 'keypad_teleop':
                controller_list.append(KeypadController(self))
            elif type == 'saved_teleop':
                mydict = type_dict[type]
                controller_list.append(SavedTeleopController(self, mydict['file'], mydict['type']))
        self.controllers = controller_list


    def choose_controller(self, switching_method = None, requested_type = None):

        # if a controller type has been specified
        if requested_type is not None:
            for controller in self.controllers:
                if controller.type == requested_type:
                    return controller
            sys.exit('Error: requested controller does not exist')

        if switching_method == 'contextual_bandit':
            max_ucb = 0.0
            for controller in self.controllers:
                confidence, sigma = controller.get_controller_confidence()
                ucb = (confidence + self.alpha*sigma)*self.success_reward
                if 'teleop' in controller.type:
                    ucb -= self.demo_cost
                if ucb > max_ucb:
                    chosen_controller = controller
                print('Controller type %s has confidence %.4f with sigma %.4f and ucb %.4f' %
                    (controller.type, confidence, sigma, ucb))
            return chosen_controller

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

        if 'block' in world.model_names:
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model_prox('block')

    def spawn_block(self, case_name, case_number):

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
        spawn_model_prox("block", sdf, "simulation", initial_pose, "world")
        rospy.sleep(0.5)
