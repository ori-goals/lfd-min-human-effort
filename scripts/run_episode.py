#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.controller import *


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    controllers = [HandCodedController(sim)]
    sim.controllers = controllers
    sim.run_new_episode()
