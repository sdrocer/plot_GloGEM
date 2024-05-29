"""
    Useful functions to plot the output of the Global Glacier Evolution Model (GloGEM).

    Code written by: Janosch Beer
"""

## import libraries ##
import os
import numpy as np
import pandas as pd

# import geospatial libraries
import shapefile as shp
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray as rxr
import rasterio
import xarray as xr

# import plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import YearLocator, DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable # to create a colorbar

# change working directory
os.chdir("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM")
from read_output import Read_GloGEM # make sure to change the working directory to folder where you code -> os.chdir

# plot settings
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
        # read GloGEM data
        (temp_data,elevation) = Read_GloGEM.point_firnice_temperature(dir)

        # create timestamp from glaciological year
        temp_data['CalYear'] = glacyear_to_calyear(1979,2019)
        temp_data['Timestamp'] = pd.to_datetime(temp_data['CalYear'].astype(str) + '-' + temp_data['Month'].astype(str), format='%Y-%m')

        start_date = '1990-01-01'
        temp_data = temp_data[(temp_data['Timestamp'] >= start_date)]

        plt.figure(figsize=(10, 4))

        # plot different depths
        color_values = np.linspace(0, 1, len(depths))
        cmap = cm.rainbow
        for depth, value in zip(depths, color_values):
            color = cmap(value)
            label = depth + "m"
            plt.plot(temp_data['Timestamp'],temp_data[depth], color=color, label=label)

        # format
        plt.ylim(-15,0)
        plt.xlabel("Time [years]")
        plt.ylabel("Ice temperature [°C]")

        # format xticks
        ax = plt.gca()  
        ax.xaxis.set_major_locator(YearLocator(5)) ## calling the locator for the x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%Y")) ## calling the formatter for the x-axis

        plt.grid(alpha=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Depth")
        plt.title("Ice temperatures at " + gl_name + " (" + elevation + " m)")
        plt.tight_layout()
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/ice_temp_single_point" + gl_name)

    def heatmap(self, dir):
        # read GloGEM data
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

        # format x-axis to show only the year
        ax = plt.gca()  
        ax.xaxis.set_major_locator(YearLocator(1)) ## calling the locator for the x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%Y")) ## calling the formatter for the x-axis
        plt.xticks(rotation=45, ha='right')

        # savefig
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/heatmap_01450.png")        

    def map10m(self, rgiid_to_find, year, gl_names):
        """ 
            idea: function that takes the RGI as a shape file as well as a digital elevation model and computes the elevation per pixel.
            The pixel will be colored acoording to it's ice temperature (taken from the respective elevation band temperature).
            This will ultimately give a 2-dimensional snapshot in time (per year) of the ice temperature (at a certain depth) of a particular glacier 
        """
        # set directions
        main_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/GloGEM/firnice_temperature/glacier_candidates/"
        dir_bands = main_dir + "temp_10m_" + rgiid_to_find + ".dat"
        rgi_dir = "/Users/janoschbeer/iCloud/PhD/data/RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp"
        dem_dir = "/Users/janoschbeer/iCloud/PhD/data/swissALTI3D/r2023/swissALTI3D_2023_10m_LV95_LN02.tif"
        ela_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/GloGEM/main_output/" + rgiid_to_find + "_ELA_r1.dat"

        # read glaciers ice temperature data per elevation band
        temp_data = Read_GloGEM.elevation_band_firnice_temperature(dir_bands)

        # check glacier name
        gl_name = gl_names[rgiid_to_find]

        # read RGI60 shapefile
        rgi_gdf = gpd.read_file(rgi_dir)

        # read DEM
        dem = rxr.open_rasterio(dem_dir, masked=True).squeeze()

        # find the glacier of interest
        glacier_geom = rgi_gdf[rgi_gdf['RGIId'] == "RGI60-11." + rgiid_to_find]

        # clip the DEM to the glacier
        glacier_dem = dem.rio.clip(glacier_geom.geometry.apply(mapping),
                                    # This is needed if your GDF is in a diff CRS than the raster data
                                    glacier_geom.crs)

        # save the clipped DEM
        glacier_dem.rio.to_raster("glacier_dem.tif")
        
        # Initialize temperature_dem with NaN values
        temperature_dem = xr.full_like(glacier_dem, np.nan)

        # Iterate over the rows in temp_data
        for elev, row in temp_data.iterrows():
            # Get the temperature
            temp = row[year]

            # Find all pixels in glacier_dem that fall within the elevation range
            in_range = ((glacier_dem >= elev) & (glacier_dem < elev+10))

            # Assign the temperature to those pixels in temperature_dem
            temperature_dem = xr.where(in_range, temp, temperature_dem)

        # Plot the temperature_dem
        fig, ax = plt.subplots(figsize=(15, 12))
        plt.rcParams.update({'font.size': 20})  # Adjust font size as needed
        im = plt.imshow(temperature_dem, cmap='coolwarm')
        plt.title(f"{dir_bands[-13:-10]} ice temperatures for {gl_name} in {year}", fontsize=22, fontweight='bold')  # Add fontweight='bold' to make the title bold
        plt.grid()
        
        # Add colorbar that is the same height as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, label='Ice Temperature [°C]')

        # Find the equilibrium line altitude (ELA) for the glacier
        elas = pd.read_csv(ela_dir, delim_whitespace=True, header=0)
        ela = elas[year]  # set ELA

        # Draw contour lines & ELA
        all_contours = ax.contour(glacier_dem, levels=range(0, int(glacier_dem.max()), 50), colors='black', linewidths=0.5)
        ela_contour  = ax.contour(glacier_dem, levels=[ela], colors='black', linewidths=3, linestyles='dashed')
        el1,_ = all_contours.legend_elements()
        el2,_ = ela_contour.legend_elements()
        ax.legend(el1 + el2, ['Elevation Contour', 'ELA'], loc='upper left', handles=[el1[0], el2[0]], labels=['50m contours', 'ELA'])

        # Set x and y axis labels to CRS coordinates without decimals
        x_ticks = np.arange(0, temperature_dem.shape[1], 50)
        y_ticks = np.arange(0, temperature_dem.shape[0], 50)
        x_labels = [f"{int(coord):.0f}" for coord in glacier_dem.x.values[x_ticks]]
        y_labels = [f"{int(coord):.0f}" for coord in glacier_dem.y.values[y_ticks]]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # set x and y axis labels
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")

        # Adjust vmin and vmax based on the data range
        vmin = np.nanmin(temperature_dem)
        vmax = np.nanmax(temperature_dem)
        im.set_clim(vmin, vmax)
        plt.tight_layout()

        # Save the plot
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/10m_ice_temp_map_" + gl_name + "_" + year + ".png")

        # Write temperature_dem to a new raster
        temperature_dem.rio.to_raster("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/rasters/10m_ice_temp_map_" + gl_name + "_" + year +".tif")

## Plotting ##

# create instance of firnice temperature plot
IceTempPlot = plot_firnice_temperature()

# Example glaciers
IceTempPlot.single_point(dir, ['3','5','9','14','24','34'],"Aletsch glacier") # Aletsch

## Glacier Candidates (according to Doctoral Plan)

# create dictionary containing rgiids and glacier names
gl_names = {"02624": "Chessjengletscher",
            "02526": "Hohlaubgletscher",
            "02244": "Sex Rouge",
            "02600": "Glacier de Tortin",
            "01931": "Milibachgletscher",
            "01962": "Vadret dal Corvatsch",
            "02803": "Triftjigletscher at Gornergrat",
            "02692": "Alphubel South"
            }

# single point, different depths
depths = ['3','5','9','14','24','34']
for rgiid, gl_name in gl_names.items():
    IceTempPlot.single_point(dir + rgiid + ".dat", depths, gl_name)

# map of ice temperatures
for rgiid, gl_name in gl_names.items():
    # IceTempPlot.map10m(rgiid, "2019", gl_names) # 2019
    IceTempPlot.map10m(rgiid, "1980", gl_names) # 1980