import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d as conv2d
import scipy.ndimage
from scipy import signal
from PIL import Image
import time
from numpy import unravel_index
from scipy.stats import chi2
from sympy import *
import random
from scipy.linalg import block_diag

def DisplayResults(F, title):
    print(title+' =')
    print(F/np.linalg.norm(F)*np.sign(F[-1,-1]))
