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
from hsrb_interface import geometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from learn_to_manipulate.controller import *

class Simulation(object):
    def __init__(self, file_path = None):
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.initial_pose_pub = rospy.Publisher('laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)
        self.block_width = 0.04
        self.goal_width_x = 0.001
        self.goal_centre_x = 0.78
        self.all_runs = []

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
            elif save_info['type'] == 'key_teleop':
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

    def run_new_episode(self, case_number, controller_type = None):
        self.reset_hsrb()
        self.spawn_table()
        controller = self.choose_controller(controller_type)
        controller.set_arm_initial()
        self.spawn_block()
        result, episode = controller.run_episode(case_number)
        self.all_runs.append(episode)

    def add_controllers(self, type_dict):
        controller_list = []
        for type in type_dict:
            if type == 'learnt':
                controller_list.append(LearntController(self))
            elif type == 'key_teleop':
                controller_list.append(KeypadController(self))
            elif type == 'saved_teleop':
                pass
        self.controllers = controller_list


    def choose_controller(self, requested_type):
        if requested_type is not None:
            controller_type = requested_type

        controller_exists = False
        for controller in self.controllers:
            if controller.type == controller_type:
                chosen_controller = controller
                controller_exists =  True
                break

        if not controller_exists:
            print('Error: requested controller does not exist')
        else:
            return chosen_controller

    def reset_hsrb(self):
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world = get_world_properties_prox()

        if 'block' in world.model_names:
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model_prox('block')

        # reset the location of the hsrb
        self.whole_body.move_to_neutral()
        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState(model_name='hsrb', reference_frame='map')
        set_model_state_prox(state)
        self.whole_body.move_to_neutral()

        # reset estimated location
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'map'
        pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        pose.pose.pose.orientation.w = 1.0
        self.initial_pose_pub.publish(pose)

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
            table_initial_pose.position.x = 0.62
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

    def spawn_block(self):
        block_name = 'block_30'
        self.block_length = float("0." + block_name.split('_')[1])
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "../../models/" + block_name + "/model.sdf")
        f = open(path,'r')
        sdf = f.read()

        initial_pose = Pose()
        initial_pose.position.x = 0.55
        initial_pose.position.y = 0.08
        initial_pose.position.z = 0.5

        initial_pose.orientation.z = 0.4226
        initial_pose.orientation.w = 0.9063

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox("block", sdf, "simulation", initial_pose, "world")
        rospy.sleep(0.5)
