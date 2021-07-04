#!usr/bin/env python
# -*- coding:utf-8 -*-
import math
import time

import numpy as np
import pandas as pd

from bin_elbow2 import bin_elbow2
from segment_histogram import find_frequents, find_vertical_lines, plot_segments, normalize_data, \
    print_horizontal_lines, running_mean


def generate_F(measurements, columns, moving_mean=50, normalize_bound=10000):
    # Calculate moving distances
    X1_squared = [i ** 2 for i in np.diff(measurements[columns[0]])]
    Y1_squared = [i ** 2 for i in np.diff(measurements[columns[1]])]
    Z1_squared = [i ** 2 for i in np.diff(measurements[columns[2]])]
    result = np.sqrt(np.sum([X1_squared, Y1_squared, Z1_squared], axis=0))
    # Distinguish G0 / G1 mode
    sorted_result = sorted(result)
    elbow_index = bin_elbow2(sorted_result)
    elbow_value = sorted_result[elbow_index]
    print("Elbow value: {0}".format(elbow_value))
    # Everything above the elbow is G0
    result = [x for x in result if x <= elbow_value]
    # Moving mean
    result = running_mean(result, moving_mean)  # dataset length will be shorten by (moving_mean-1)
    # Normalize
    result = normalize_data(result, normalize_bound)
    return result


if __name__ == '__main__':
    # Please set these parameters manually
    file_name = 'measurements1.csv'
    # columns = ['X1', 'Y1', 'Z1']
    columns = ['X2', 'Y2', 'Z2']
    moving_mean = 10  # Moving mean filter size
    bin_size = 10  # size of a bin, Can leave it alone as dataset is normalized to (0,10000)
    # min_length = 200  # for horizontal lines
    debug = True  # if true, print log

    start_time = time.time()
    measurements = pd.read_csv(file_name, usecols=columns, header=0, sep=';')

    # Generate F
    F = generate_F(measurements, columns, moving_mean)

    # Step 1: find the frequent data (as peaks in histogram) and merge if it's continuous
    merged_frequents, bin_size, mean_len = find_frequents(F, bin_size, debug)
    # min_length = math.ceil(mean_len) + 60  # a little higher than average height
    min_length = math.ceil(mean_len/100)*100  # Upper multiples of 100

    # Step 2: Examine each middle point and extend
    vertical_lines = find_vertical_lines(F, merged_frequents, bin_size, min_length, debug, True, True)

    print("Time: {0}s".format(round(time.time() - start_time)))

    print("Found {0} vertical parts".format(len(vertical_lines)))

    shortened = max(moving_mean - 1, 0)
    horizontal_lines = print_horizontal_lines(vertical_lines, len(F) + shortened, shortened)

    final_result = np.sort(np.array(horizontal_lines).flatten())
    print(",".join(str(i) for i in final_result))

    plot_segments(F, final_result)
