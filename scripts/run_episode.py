#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.controller import *


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    controllers = [LearntController(sim), KeypadController(sim)]
    sim.controllers = controllers
    case_number = 10
    sim.run_new_episode(case_number, controller_type = 'learnt')
    sim.run_new_episode(case_number, controller_type = 'learnt')
