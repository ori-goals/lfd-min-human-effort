#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation.load_simulation('/home/marcrigter/2019-08-08-11-49_learnt1_key_teleop0.pkl')
    case_number = 10
    sim.run_new_episode(case_number, controller_type = 'learnt')
