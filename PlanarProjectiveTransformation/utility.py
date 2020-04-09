import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy import signal
from numpy import unravel_index
from scipy.stats import chi2
import random
from scipy.linalg import block_diag

def DisplayResults(H, title):
    print(title+' =')
    print (H/np.linalg.norm(H)*np.sign(H[-1,-1]))

