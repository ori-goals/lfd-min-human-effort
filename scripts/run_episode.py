#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import rospy


if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    sim = Simulation()
    sim.add_controllers({'learnt':'', 'key_teleop':''})

    case_number = 10

    sim.run_new_episode(case_number, controller_type = 'learnt')
    sim.save_simulation('/home/marcrigter')
