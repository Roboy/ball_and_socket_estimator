import magjoint
import sys, random
import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from multiprocessing import Pool, freeze_support, get_context, set_start_method
from itertools import repeat

class MagnetOrientation:
    ball_joint = None
    x_step = 10
    y_step = 10
    z_step = 10
    sensors = None
    joint_positions = []
    def __init__(self,ball_joint):
        self.ball_joint = ball_joint
        self.x_step = x_step
        self.y_step = y_step
        self.z_step = z_step
        self.sensors = ball_joint.gen_sensors()
        
