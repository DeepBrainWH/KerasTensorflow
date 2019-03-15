# -*- encoding:utf-8 -*-
import numpy as np

class myRnn:

    def __init__(self):
        self.int2binary = {}
        self.binary_dim = 8
        self.largest_number = pow(2, self.binary_dim)

    def sigmoid(self, x):
        output = 1.0/(1 + np.exp(-x))
        return output

    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

