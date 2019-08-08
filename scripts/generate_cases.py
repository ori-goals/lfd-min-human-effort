#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('similar_cases', 10,
    {'min_y':0.05, 'max_y':0.08, 'min_x':0.55, 'max_x':0.57, 'min_angle_deg':50, 'max_angle_deg':55, 'block_names':['block_30']})
