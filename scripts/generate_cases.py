#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('don't_verwrite', 3000,
    {'min_y':0.10, 'max_y':-0.10, 'min_x':0.45, 'max_x':0.55, 'min_angle_deg':-20, 'max_angle_deg':20, 'block_names':['block_12', 'block_20', 'block_30']})
