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

def glacyear_to_calyear(start_year, end_year):
    # Define the number of repetitions for each year
    num_years = end_year - start_year

    # Create the list using list comprehension
    year_list = [(start_year+1 + i) for i in range(num_years) for _ in range(12)]
    year_list = [start_year,start_year,start_year] + year_list
    return year_list[:-3]

class plot_firnice_temperature:
    def single_point(self, dir):
        (temp_data,elevation) = Read_GloGEM.point_firnice_temperature(dir)

        # create timestamp from glaciological year
        temp_data['CalYear'] = glacyear_to_calyear(1979,2019)
        temp_data['Timestamp'] = pd.to_datetime(temp_data['CalYear'].astype(str) + '-' + temp_data['Month'].astype(str), format='%Y-%m')

        plt.figure(figsize=(10, 4))
        plt.plot(temp_data['Timestamp'],temp_data['14'], color='red')

        plt.ylim(-0.5,0)
        plt.xlabel("Time")
        plt.ylabel("14m ice temperature [°C]")
        plt.title("14m ice temperature for RGI" + dir[-9:-4] + " at " + elevation + " m")
        plt.savefig("/home/jabeer/products/test_run01/plots/14m_icetemp_" + dir[-9:-4])



# Plotting
IceTempPlot = plot_firnice_temperature()

# Example glaciers
IceTempPlot.single_point("products/test_run01/CentralEurope/PAST/firnice_temperature/temp_ID1_01450.dat") # Aletsch
IceTempPlot.single_point("/home/jabeer/products/test_run01/CentralEurope/PAST/firnice_temperature/temp_ID1_02822.dat") # Grenzgletscher