"""
    This class helps to read, filter & organise the different output files of the Global Glacier Evolution Model (GloGEM) 
    written in IDL. Output files are explicitly ".dat" files and can thus easily be interpreted using python or any other
    programming language.

    Code written by: Janosch Beer
"""

import os
import numpy as np
import pandas as pd

class Read_GloGEM():
    def point_firnice_temperature(dir):
        """reads the point wise output of the firnice_temperature model & returns it as a dataframe"""
        temp_data = pd.read_csv(dir, delimiter=r"\s+", header=1, index_col=False)
        with open(dir,'r') as file:
            elevation = file.readline().strip()[17:21]
        return (temp_data,elevation)
    
    def elevation_band_firnice_temperature(dir):
        """reads the elevation band ouput of the firnice_temperature model & returns it as a dataframe"""
        temp_data = pd.read_csv(dir, delimiter=r"\s+", header=0, index_col='Elev')
        temp_data.replace(-99.0, np.nan, inplace=True)
        return (temp_data)
    
    def elevation_band_consensus_ice_thickness(dir):
        """reads the elevation band ouput of the consensus_ice_thickness of 2019 & returns it as a dataframe"""
        thickness_data = pd.read_csv(dir, delimiter=r"\s+", header=4, index_col=False)
        thickness_data.replace(-99.0, np.nan, inplace=True)
        return (thickness_data['Thickness(m)'])
    
    def elevation_band_slope(dir):
        """reads the elevation band ouput of the consensus_ice_thickness of 2019 & returns it as a dataframe"""
        slope_data = pd.read_csv(dir, delimiter=r"\s+", header=4, index_col=False)
        slope_data.replace(-99.0, np.nan, inplace=True)
        return (slope_data['Slope(deg)'])
    
    def elevation_band_elevation(dir):
        """reads the elevation band ouput of the consensus_ice_thickness of 2019 & returns it as a dataframe"""
        elev_data = pd.read_csv(dir, delimiter=r"\s+", header=4, index_col=False)
        elev_data.replace(-99.0, np.nan, inplace=True)
        return (elev_data['Elev_start'])
    