from __future__ import division
import json
import numpy as np
import argparse
import time
import math
import os.path
from sklearn import datasets
from sklearn import linear_model

def polyak(theta_values, lr, lambdaFac, grad, prev_velocity, mu):
    curr_velocity = mu * prev_velocity - lr * grad
    theta_values = theta_values + curr_velocity
    return (theta_values, curr_velocity)

    # as the difference is only in the weight vector passed to the gradient computation procedure
