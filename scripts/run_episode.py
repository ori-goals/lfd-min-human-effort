#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulate
from learn_to_manipulate.controller import *


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    controllers = [HandCodedController()]
    sim = Simulate(controllers)
    sim.run_new_episode()
