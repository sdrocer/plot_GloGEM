"""
    Useful functions to plot the output of the Global Glacier Evolution Model (GloGEM).

    Code written by: Janosch Beer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_output import Read_GloGEM

# Increase the DPI
plt.rcParams['figure.dpi'] = 200  # Adjust DPI as needed

class plot_firnice_temperature():
    def single_point(dir):
        (temp_data,elevation) = Read_GloGEM.point_firnice_temperature(dir)

        # create timestamp
        temp_data['CalYear'] = 1779 + (temp_data['Month'] - 10) // 12
        temp_data['Timestamp'] = pd.to_datetime(temp_data[['CalYear', 'Month']].assign(DAY=1))


        plt.figure(figsize=(10, 4))
        plt.plot(temp_data['Timestamp'],temp_data['14'])
        # plt.plot([1,2,3,4],[0,-0.5,-0.8,0])

        plt.ylim(-1,0)
        plt.xlabel("Time")
        plt.ylabel("14m ice temperature [°C]")
        plt.title("14m ice temperature for RGI" + dir[-9:-4] + " at " + elevation + " m")
        plt.savefig("/home/jabeer/products/test_run01/plots/14m_icetemp_" + dir[-9:-4])



# Plotting
plot_firnice_temperature.single_point("products/test_run01/CentralEurope/PAST/firnice_temperature/temp_ID1_01450.dat")