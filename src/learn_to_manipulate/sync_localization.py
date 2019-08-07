#!/usr/bin/env python
import hsrb_interface
import rospy
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
from hsrb_interface import geometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *

class Sync(object):
    def __init__(self):
        rospy.init_node('sync_localization')
        self.initial_pose_pub = rospy.Publisher('laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)
        self.update_localization()

    def update_localization(self):
        while not rospy.is_shutdown():
            rospy.wait_for_service('/gazebo/get_model_state')
            get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp1 = get_model_state_prox('hsrb','')

            # reset estimated location
            pose = PoseWithCovarianceStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'map'
            pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
            pose.pose.pose = resp1.pose
            self.initial_pose_pub.publish(pose)
            rospy.sleep(0.02)

if __name__ == "__main__":
    sync = Sync()
