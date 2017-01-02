import sys
import os
import shutil
sys.path.append("/Users/abhiram/Dropbox/Adaptive-systems/SWMM_inp_edit/")
from inp_edit_fun import storm_event  # .inp file editor
import numpy as np
from scipy import signal

input_files = 'Parallel.inp'

rain_12_hr = {"1": 1.82, "2": 2.06, "5": 2.49, "10": 2.90, "25": 3.54, "50": 4.09, "100": 4.68, "200": 5.34, "500": 6.29, "1000": 7.07}
rain_24_hr = {"1": 2.10, "2": 2.35, "5": 2.82, "10": 3.26, "25": 3.93, "50": 4.49, "100": 5.11, "200": 5.78, "500": 6.74, "1000": 7.52}

for i in rain_12_hr:

    output_files = 'Parallel_year.inp'

    dir_path = "/Users/abhiram/Dropbox/Adaptive-systems/Cyndee Test Cases/Parallel_Storms_Years_" + i
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    os.chdir(dir_path)
    shutil.copy('/Users/abhiram/Dropbox/Adaptive-systems/Cyndee Test Cases/Parallel.inp', dir_path)
    shutil.copy('/Users/abhiram/Dropbox/Adaptive-systems/Cyndee Test Cases/swmm5.so', dir_path)
    shutil.copy('/Users/abhiram/Dropbox/Adaptive-systems/Cyndee Test Cases/swmm.py', dir_path)
    shutil.copy('/Users/abhiram/Dropbox/Adaptive-systems/Cyndee Test Cases/rain_test_swmm.py', dir_path)

    storm = signal.gaussian(12, std=2)
    rain_interval = (np.linspace(0, 12, 12, dtype=int))
    rain_interval = map(str, rain_interval)
    temp = rain_12_hr[i]
    storm = temp*storm
    storm = map(str, storm)

    storm_event(input_files, output_files, 'TIMESERIES', 'T1', rain_interval, storm)

    os.system('python rain_test_swmm.py')
    os.chdir('/Users/abhiram/Dropbox/Adaptive-systems/Cyndee Test Cases/')
