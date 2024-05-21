"""
    Useful functions to plot the output of the Global Glacier Evolution Model (GloGEM).

    Code written by: Janosch Beer
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import YearLocator, DateFormatter

os.chdir("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM")
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
    def single_point(self, dir, depths, gl_name):
        (temp_data,elevation) = Read_GloGEM.point_firnice_temperature(dir)

        # create timestamp from glaciological year
        temp_data['CalYear'] = glacyear_to_calyear(1979,2019)
        temp_data['Timestamp'] = pd.to_datetime(temp_data['CalYear'].astype(str) + '-' + temp_data['Month'].astype(str), format='%Y-%m')

        plt.figure(figsize=(10, 4))

        # plot different depths
        color_values = np.linspace(0, 1, len(depths))
        cmap = cm.rainbow
        for depth, value in zip(depths, color_values):
            color = cmap(value)
            label = depth + "m"
            plt.plot(temp_data['Timestamp'],temp_data[depth], color=color, label=label)

        # format
        plt.ylim(-8,0)
        plt.xlabel("Time [years]")
        plt.ylabel("Ice temperature [°C]")
        plt.grid(alpha=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Depth")
        plt.title("Ice temperatures at " + gl_name + " (" + elevation + " m)")
        plt.tight_layout()
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/ice_temp_" + gl_name)

    def heatmap(self, dir):
        (temp_data,elevation) = Read_GloGEM.point_firnice_temperature(dir)

        # create timestamp from glaciological year
        temp_data['CalYear'] = glacyear_to_calyear(1979,2019)
        temp_data['Timestamp'] = pd.to_datetime(temp_data['CalYear'].astype(str) + '-' + temp_data['Month'].astype(str), format='%Y-%m')

        # reorganise dataframe to enable visualisation as heatmap
        temp_data.set_index('Timestamp', inplace=True)
        temp_data = temp_data.drop(columns=['CalYear','Year','Month'])
        temp_data.replace(-99.0, np.nan, inplace=True)
        temp_data_t = temp_data.transpose() # transpose to have depths as columns and timestamp on x axis


        # create ice temperature heatmap
        plt.figure(figsize=(15, 10))
        plt.imshow(temp_data_t)
        # sns.heatmap(temp_data_t, cmap='viridis', cbar_kws={'label': 'Ice Temperature (°C)'}, yticklabels=True, vmax=30)

        # Format x-axis to show only the year
        ax = plt.gca()  
        ax.xaxis.set_major_locator(YearLocator(1)) ## calling the locator for the x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%Y")) ## calling the formatter for the x-axis

        plt.xticks(rotation=45, ha='right')
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/heatmap_01450.png")        


## Plotting

# set dir & depths
dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/GloGEM/firnice_temperature/temp_ID1_01450.dat"
depths = ['3','5','9','14','24','34']

IceTempPlot = plot_firnice_temperature()

# Example glaciers
IceTempPlot.single_point(dir, ['3','5','9','14','24','34'],"Aletsch glacier") # Aletsch

# Glacier Candidates (according to Doctoral Plan)
IceTempPlot.single_point() # Chessjengletscher
IceTempPlot.single_point() # Hohlaubgletscher at Saas Grund
IceTempPlot.single_point() # Sex Rouge
IceTempPlot.single_point() # Glacier de Tortin
IceTempPlot.single_point() # Milibachgletscher
IceTempPlot.single_point() # Corvatsch
IceTempPlot.single_point() # Triftjigletscher at Gornergrat
IceTempPlot.single_point() # Alphubel South
