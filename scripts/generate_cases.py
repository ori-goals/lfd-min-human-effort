#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('similar_cases', 20,
    {'min_y':0.05, 'max_y':0.08, 'min_x':0.45, 'max_x':0.48, 'min_angle_deg':35, 'max_angle_deg':40, 'block_names':['block_30']})
