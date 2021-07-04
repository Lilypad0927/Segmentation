#!usr/bin/env python
# -*- coding:utf-8 -*-
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from segment_histogram import find_frequents, find_vertical_lines, plot_segments, normalize_data

if __name__ == '__main__':
    # Please set these parameters manually
    #file_name = 'measurements.csv'  # Should be: 10 vertical lines
    file_name = 'measurements1.csv'  # Should be: 25 vertical lines
    columns = ['SS']
    bin_size = 10  # size of a bin (Not important if normalized)
    min_length = 500  # for horizontal lines, e.g. 500 means 500*2ms = 1s
    debug = False  # if true, print log and plot

    start_time = time.time()
    measurements = pd.read_csv(file_name, usecols=columns, header=0, sep=';')
    SS = measurements[columns[0]]
    # Normalize
    SS = normalize_data(SS, 10000)

    # Step 1: find the frequent data (as peaks in histogram) and merge if it's continuous
    merged_frequents, bin_size, mean_len = find_frequents(SS, bin_size, debug)

    # Step 2: Examine each middle point and extend
    vertical_lines = find_vertical_lines(SS, merged_frequents, bin_size, min_length, debug)

    print("Found {0} vertical lines, Max length: {1}".format(len(vertical_lines),
                                                             max([(line[1] - line[0]) for line in vertical_lines])))
    final_result = np.sort(np.array(vertical_lines).flatten())
    print(final_result)

    print("Time: {0}s".format(round(time.time() - start_time)))

    plot_segments(SS, final_result)

