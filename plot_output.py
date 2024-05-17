"""
    Useful functions to plot the output of the Global Glacier Evolution Model (GloGEM).

    Code written by: Janosch Beer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from read_output import Read_GloGEM # make sure to change the working directory to folder where you code -> os.chdir

# Increase the DPI
plt.rcParams['figure.dpi'] = 200                                # Adjust DPI as needed
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})  # Adjust font size & family as needed

def glacyear_to_calyear(start_year, end_year):
    """
        Function to convert the glaciological year time series to the calendar year time series
    """
    # Define the number of repetitions for each year
    num_years = end_year - start_year

    # Create the list using list comprehension
    year_list = [(start_year+1 + i) for i in range(num_years) for _ in range(12)]
    year_list = [start_year,start_year,start_year] + year_list
    return year_list[:-3]

class plot_firnice_temperature:
    """
        Class with various methods to plot the firnice temperature output of GloGEM
    """
    def single_point(self, dir, depths):
        (temp_data,elevation) = Read_GloGEM.point_firnice_temperature(dir)

        # create timestamp from glaciological year
        temp_data['CalYear'] = glacyear_to_calyear(1979,2019)
        temp_data['Timestamp'] = pd.to_datetime(temp_data['CalYear'].astype(str) + '-' + temp_data['Month'].astype(str), format='%Y-%m')

        plt.figure(figsize=(10, 4))

        # plot different depths
        color_values = np.linspace(0, 1, len(depths))
        cmap = cm.get_cmap('afmhot_r')
        for depth, value in zip(depths, color_values):
            color = cmap(value)
            plt.plot(temp_data['Timestamp'],temp_data[depth], color=color)

        # format
        plt.ylim(-10,0)
        plt.xlabel("Time [years]")
        plt.ylabel("14m ice temperature [°C]")
        # plt.grid()
        plt.tight_layout()
        plt.title("14m ice temperature for RGI" + dir[-9:-4] + " at " + elevation + " m")
        plt.savefig("/home/jabeer/products/test_run01/plots/14m_icetemp_" + dir[-9:-4])


# Plotting
IceTempPlot = plot_firnice_temperature()

# Example glaciers
IceTempPlot.single_point("/home/jabeer/products/test_run01/CentralEurope/PAST/firnice_temperature/temp_ID1_01450.dat", ['1','2','3','4']) # Aletsch
IceTempPlot.single_point("/home/jabeer/products/test_run01/CentralEurope/PAST/firnice_temperature/temp_ID1_02822.dat") # Grenzgletscher

# Glacier Candidated (according to Doctoral Plan)
IceTempPlot.single_point()

dir = "/home/jabeer/products/test_run01/CentralEurope/PAST/firnice_temperature/temp_ID1_01450.dat"
depths = ['1','2','3','4','5','6','7']