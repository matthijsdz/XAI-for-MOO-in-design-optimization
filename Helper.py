import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ScatterPlot(x, y, z = None):
    fig = plt.figure()
    if z.all() != None:
        ax = plt.axes(projection="3d")
        ax.scatter(x , y, z)
    else:
        ax.scatter(x, y)
    plt.show()
