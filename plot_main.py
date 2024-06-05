"""
    Useful functions to plot the output of the Global Glacier Evolution Model (GloGEM).

    Code written by: Janosch Beer
"""

## import libraries ##
import os
import numpy as np
import pandas as pd
import glob

# import geospatial libraries
import shapefile as shp
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from rasterio.merge import merge
from shapely.geometry import mapping

# import plotting libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.dates import YearLocator, DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable # to create a colorbar

# change working directory
os.chdir("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM")
from read_output import Read_GloGEM # make sure to change the working directory to folder where you code -> os.chdir

# plot settings
plt.rcParams['figure.dpi'] = 200                                # Adjust DPI as needed
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})  # Adjust font size & family as needed

# create custom colormap
coolwarm = plt.cm.get_cmap('coolwarm')
red = coolwarm(0.99)  # Get the red color
blue_colors = [coolwarm(i) for i in np.linspace(0.0, 0.5, 100)]  # Get the blue colors
colors = blue_colors + [red] # Combine the blue colors with the red color
icetemp = mcolors.LinearSegmentedColormap.from_list('new_cmap', colors)

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
            This will ultimately give a 2-dimensional snapshot in time (per year) of the ice temperature (at a certain depth) of a particular glacier. 
        """
        # check glacier name
        gl_name = gl_names[rgiid_to_find]

        # set directions
        main_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/GloGEM/firnice_temperature/glacier_candidates_no_permeability/"
        bands_dir = main_dir + "temp_10m_" + rgiid_to_find + ".dat"
        rgi_dir = "/Users/janoschbeer/iCloud/PhD/data/RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp"
        ela_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/GloGEM/main_output/" + rgiid_to_find + "_ELA_r1.dat"
        
        # set dem directory -> find the folder that contains the gl_name
        dem_dir = ""
        swissALTI3D_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/swissALTI3D/"
        folders = glob.glob(swissALTI3D_dir + "*")
        for folder in folders:
            if gl_name in folder:
                dem_dir = folder + "/merged_ortho.tif"
                break

        if dem_dir == "":
            print("Could not find the DEM folder for the glacier:", gl_name)
        else:
            print("DEM directory:", dem_dir)

        # read glaciers ice temperature data per elevation band
        temp_data = Read_GloGEM.elevation_band_firnice_temperature(bands_dir)

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
        im = plt.imshow(temperature_dem, cmap=icetemp, vmin=-10, vmax=-0.05)
        plt.title(f"{bands_dir[-13:-10]} ice temperatures for {gl_name} in {year}", fontsize=22, fontweight='bold', pad=20)  # Add pad=20 to increase the distance of the title from the plot
        plt.grid()
        
        # Add colorbar that is the same height as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, label='Ice Temperature [°C]')
        
        # Modify the colorbar ticks
        cbar.ax.set_yticks([-10, -8, -6, -4, -2, -0.05])
        cbar.ax.set_yticklabels(['< -10','-8','-6','-4','-2','> -0.05'])

        # Find the equilibrium line altitude (ELA) for the glacier
        elas = pd.read_csv(ela_dir, delimiter=r"\s+", header=0)
        ela = elas[year]  # set ELA

        # Draw contour lines & ELA
        all_contours = ax.contour(glacier_dem, levels=range(0, int(glacier_dem.max()), 50), colors='black', linewidths=0.5)
        ela_contour  = ax.contour(glacier_dem, levels=[ela], colors='black', linewidths=3, linestyles='dashed')
        el1,_ = all_contours.legend_elements()
        el2,_ = ela_contour.legend_elements()

        # Add the legend inside the plot area
        legend = ax.legend(el1 + el2, ['Elevation Contour', 'ELA'], loc='upper left', handles=[el1[0], el2[0]], labels=['50m contours', 'ELA'])
        legend.get_frame().set_edgecolor('black')  # Set the color of the legend frame
        legend.get_frame().set_linewidth(1.0)  # Set the width of the legend frame

        # Set x and y axis labels to CRS coordinates without decimals
        x_ticks = np.arange(0, temperature_dem.shape[1], temperature_dem.shape[1] // 10)  # Adjust the number of ticks based on the size of the overall image
        y_ticks = np.arange(0, temperature_dem.shape[0], temperature_dem.shape[0] // 10)
        x_labels = [f"{int(coord):.0f}" for coord in temperature_dem.x.values[x_ticks]]
        y_labels = [f"{int(coord):.0f}" for coord in temperature_dem.y.values[y_ticks]]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # set x and y axis labels
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")

        plt.tight_layout()

        # Save the plot
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/" + gl_name + "/englacial_temperature_no_permeability/10m_ice_temp_map_" + gl_name + "_" + year + ".png")

        # Write temperature_dem to a new raster
        temperature_dem.rio.to_raster("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/rasters/10m_ice_temp_map_" + gl_name + "_" + year +".tif")

    def gl_profile(self, rgiid_to_find, year, gl_names):
        # check glacier name
        gl_name = gl_names[rgiid_to_find]

class plot_ice_thickness:
    def consensus(self, rgiid_to_find, gl_names):
        """
            Function to plot the ice thickness of a glacier (per elevation band) on the basis of rgi60 bands consensus 2019 (Farinotti et al. 2019)
        """
        # check glacier name
        gl_name = gl_names[rgiid_to_find]

        # set directions
        thickness_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/global_ice_thickness/centraleurope/" + rgiid_to_find + ".dat" # ice thickness per elevation band
        rgi_dir = "/Users/janoschbeer/iCloud/PhD/data/RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp"
        ela_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/GloGEM/main_output/" + rgiid_to_find + "_ELA_r1.dat"

        # set dem directory -> find the folder that contains the gl_name
        dem_dir = ""
        swissALTI3D_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/swissALTI3D/"
        folders = glob.glob(swissALTI3D_dir + "*")
        for folder in folders:
            if gl_name in folder:
                dem_dir = folder + "/merged_ortho.tif"
                break

        thickness_data = Read_GloGEM.elevation_band_consensus_ice_thickness(thickness_dir)
        
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

        # Initialize thickness_dem with NaN values
        glacier_thickness = xr.full_like(glacier_dem, np.nan)

        # Iterate over the rows in temp_data
        for elev, row in thickness_data.iterrows():
            # Get the temperature
            thickness = row['Thickness(m)']

            # Find all pixels in glacier_dem that fall within the elevation range
            in_range = ((glacier_dem >= elev) & (glacier_dem < elev+10))

            # Assign the temperature to those pixels in thickness_dem
            glacier_thickness = xr.where(in_range, thickness, glacier_thickness)

        # Plot the ice thickness_dem
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.rcParams.update({'font.size': 20})  # Adjust font size as needed
        im = plt.imshow(glacier_thickness, cmap='viridis_r')
        plt.title(f"Ice thickness of {gl_name} in 2019 (Farinotti et al. 2019)", fontsize=22, fontweight='bold', pad=20)  # Add pad=20 to increase the distance of the title from the plot
        plt.grid()
        
        # Add colorbar that is the same height as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, label='Ice thickness [m]')

        # Find the equilibrium line altitude (ELA) for the glacier
        elas = pd.read_csv(ela_dir, delimiter=r"\s+", header=0)
        ela = elas['2019']  # set ELA

        # Draw contour lines & ELA
        all_contours = ax.contour(glacier_dem, levels=range(0, int(glacier_dem.max()), 50), colors='black', linewidths=0.5)
        ela_contour  = ax.contour(glacier_dem, levels=[ela], colors='black', linewidths=3, linestyles='dashed')
        el1,_ = all_contours.legend_elements()
        el2,_ = ela_contour.legend_elements()

        # Add the legend inside the plot area
        legend = ax.legend(el1 + el2, ['Elevation Contour', 'ELA'], loc='upper left', handles=[el1[0], el2[0]], labels=['50m contours', 'ELA'])
        legend.get_frame().set_edgecolor('black')  # Set the color of the legend frame
        legend.get_frame().set_linewidth(1.0)  # Set the width of the legend frame

        # Set x and y axis labels to CRS coordinates without decimals
        x_ticks = np.arange(0, glacier_thickness.shape[1], glacier_thickness.shape[1] // 10)  # Adjust the number of ticks based on the size of the overall image
        y_ticks = np.arange(0, glacier_thickness.shape[0], glacier_thickness.shape[0] // 10)
        x_labels = [f"{int(coord):.0f}" for coord in glacier_thickness.x.values[x_ticks]]
        y_labels = [f"{int(coord):.0f}" for coord in glacier_thickness.y.values[y_ticks]]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # set x and y axis labels
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        plt.tight_layout()

        # Save the plot
        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/" + gl_name + "/ice_thickness/consensus_ice_thickness_2019_" + gl_name + ".png")

    def sgi(self,sgiid_to_find, gl_names):
        """
            Function to plot the ice thickness of all Swiss glaciers on the basis of radar data and glaciological modelling (Grab et al. 2021)
            taken from the Swiss Glacier Inventory (SGI)
        """
        # check glacier name
        gl_name = gl_names[sgiid_to_find]

        # set directions
        thickness_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/04_IceThickness_SGI/" + sgiid_to_find + "_IceThickness.tif"
    
        # read ice thickness data
        glacier_thickness = rxr.open_rasterio(thickness_dir, masked=True).squeeze()

        fig, ax = plt.subplots(figsize=(15,15))
        plt.rcParams.update({'font.size': 20})
        im = plt.imshow(glacier_thickness, cmap='viridis_r')
        plt.title(f"Ice thickness of {gl_name} in 2020 (Grab et al. 2021)", fontsize=22, fontweight='bold', pad=20)  # Add pad=20 to increase the distance of the title from the plot
        plt.grid()
        
        # Add colorbar that is the same height as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, label='Ice thickness [m]')

        # Set x and y axis labels to CRS coordinates without decimals
        x_ticks = np.arange(0, glacier_thickness.shape[1], glacier_thickness.shape[1] // 10)  # Adjust the number of ticks based on the size of the overall image
        y_ticks = np.arange(0, glacier_thickness.shape[0], glacier_thickness.shape[0] // 10)
        x_labels = [f"{int(coord):.0f}" for coord in glacier_thickness.x.values[x_ticks]]
        y_labels = [f"{int(coord):.0f}" for coord in glacier_thickness.y.values[y_ticks]]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # set x and y axis labels
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        plt.tight_layout()

        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/" + gl_name + "/ice_thickness/sgi_ice_thickness_2020_" + gl_name + ".png")

class plot_glacier_slope:
    def from_dem(self, rgiid_to_find, gl_names):
        """
            Function to plot the slope of a glacier on the basis of a digital elevation model (DEM)
        """
        # check glacier name
        gl_name = gl_names[rgiid_to_find]

        # set directions
        rgi_dir = "/Users/janoschbeer/iCloud/PhD/data/RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp"

        # set dem directory -> find the folder that contains the gl_name
        dem_dir = ""
        swissALTI3D_dir = "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/swissALTI3D/"
        folders = glob.glob(swissALTI3D_dir + "*")
        for folder in folders:
            if gl_name in folder:
                dem_dir = folder + "/merged_ortho.tif"
                break

        # read RGI60 shapefile
        rgi_gdf = gpd.read_file(rgi_dir)

        # read DEM
        dem = rxr.open_rasterio(dem_dir, masked=True).squeeze()

        # find the glacier of interest
        glacier_geom = rgi_gdf[rgi_gdf['RGIId'] == "RGI60-11." + rgiid_to_find]

        # Calculate the slope
        slope = rd.TerrainAttribute(shasta_dem, attrib='slope_riserun')

        glacier_slope = dem.rio.calculate_slope()

        # clip the slope to the glacier
        glacier_slope = glacier_slope.rio.clip(glacier_geom.geometry.apply(mapping),
                                    # This is needed if your GDF is in a diff CRS than the raster data
                                    glacier_geom.crs)
        
        # clip the DEM to the glacier
        glacier_dem = dem.rio.clip(glacier_geom.geometry.apply(mapping),
                                    # This is needed if your GDF is in a diff CRS than the raster data
                                    glacier_geom.crs)
        

        # Plot the slope
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.rcParams.update({'font.size': 20})
        im = plt.imshow(glacier_slope, cmap='Reds')
        plt.title(f"Slope of {gl_name}", fontsize=22, fontweight='bold', pad=20)  # Add pad=20 to increase the distance of the title from the plot
        plt.grid()

        # Draw contour lines & ELA
        all_contours = ax.contour(glacier_dem, levels=range(0, int(glacier_dem.max()), 50), colors='black', linewidths=0.5)
        el1,_ = all_contours.legend_elements()

        # Add the legend inside the plot area
        legend = ax.legend(el1 ['Elevation Contour'], loc='upper left', handles=[el1[0]], labels=['50m contours'])
        legend.get_frame().set_edgecolor('black')  # Set the color of the legend frame
        legend.get_frame().set_linewidth(1.0)  # Set the width of the legend frame

        # Set x and y axis labels to CRS coordinates without decimals
        x_ticks = np.arange(0, glacier_slope.shape[1], glacier_slope.shape[1] // 10)  # Adjust the number of ticks based on the size of the overall image
        y_ticks = np.arange(0, glacier_slope.shape[0], glacier_slope.shape[0] // 10)
        x_labels = [f"{int(coord):.0f}" for coord in glacier_slope.x.values[x_ticks]]
        y_labels = [f"{int(coord):.0f}" for coord in glacier_slope.y.values[y_ticks]]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # set x and y axis labels
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        plt.tight_layout()

        plt.savefig("/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/Code/plot_GloGEM/plots/" + gl_name + "/slope/slope_" + gl_name + ".png")

## Plotting ##

# create instance of the plot classes
IceTempPlot = plot_firnice_temperature() # ice temperature
IceThickPlot = plot_ice_thickness()      # ice thickness
IceSlopePlot = plot_glacier_slope()      # glacier slope

## Glacier Candidates (according to Doctoral Plan)

# create dictionary containing rgiids and glacier names for the glacier candidates
rgi_names = {"01450": "Aletsch Glacier",
            # "02671": "Chessjengletscher East",
            # "02624": "Feegletscher",
            # "02526": "Hohlaubgletscher",
            # "02244": "Sex Rouge",
            # "02600": "Glacier de Tortin",
            # "01931": "Milibachgletscher",
            # "01962": "Vadret dal Corvatsch",
            # "02803": "Triftjigletscher at Gornergrat",
            # "02692": "Alphubel South"
}

# create dictionary containing sgiids and glacier names for the Swiss Glacier Inventory
sgi_names = {"B36-26" : "Aletsch Glacier",
             "B52-33" : "Chessjengletscher East",
             "B53-04" : "Feegletscher",
             "B51-12" : "Hohlaubgletscher",
             "B16-01" : "Sex Rouge",
             "B75-12" : "Glacier de Tortin",
             "B30-05" : "Milibachgletscher",
             "E23-18" : "Vadret dal Corvatsch",
             "B56-04" : "Triftjigletscher at Gornergrat",
             "B55-15" : "Alphubel South"
}

# single point, different depths
depths = ['3','5','9','14','24','34']
for rgiid, gl_name in rgi_names.items():
    IceTempPlot.single_point(dir + rgiid + ".dat", depths, gl_name)

# map of ice temperatures
for rgiid, gl_name in rgi_names.items():
    for year in range(1980, 2020):
        IceTempPlot.map10m(rgiid, str(year), rgi_names)

# ice thickness consensus estimate
for rgiid, gl_name in rgi_names.items():
    IceThickPlot.consensus(rgiid, rgi_names)

# ice thickness SGI
for sgiid, gl_name in sgi_names.items():
    IceThickPlot.sgi(sgiid, sgi_names)

# glacier slope
for rgiid, gl_name in rgi_names.items():
    IceSlopePlot.from_dem(rgiid, rgi_names)