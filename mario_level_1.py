#!/usr/bin/env python
__author__ = 'justinarmstrong'

"""
This is an attempt to recreate the first level of
Super Mario Bros for the NES.
"""

import sys
import pygame as pg
from data.main import main
import cProfile
import prepare_network
import cv2
import imutils

if __name__=='__main__':
    vc, net = prepare_network.prepare_network()
    main(net)
    prepare_network.destroy_network()
    pg.quit()
    sys.exit()