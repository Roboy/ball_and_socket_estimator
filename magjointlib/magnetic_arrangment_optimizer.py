#!/usr/bin/python3
import magjoint
import sys
import numpy as np

if len(sys.argv) < 3:
    print("\nUSAGE: ./magnetic_sensor_calibration.py ball_joint_config visualize_only, e.g. \n python3 magnetic_sensor_calibration.py two_magnets.yaml 1\n")
    sys.exit()

balljoint_config = sys.argv[1]
visualize_only = sys.argv[2]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets()
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

ball.optimizeMagnetArrangement('posangle')
