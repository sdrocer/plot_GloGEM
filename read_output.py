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
    def read_firnice_temperature(self):
        """reads the output of the firnice_temperature model & returns it as a np array"""
        temp_data = np.loadtxt(self)
        return temp_data