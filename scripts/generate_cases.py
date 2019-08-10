#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('lfd_rl_aug10', 2000,
    {'min_y':0.05, 'max_y':-0.05, 'min_x':0.45, 'max_x':0.55, 'min_angle_deg':-25, 'max_angle_deg':25, 'block_names':['block_30']})
