#!usr/bin/env python
# -*- coding:utf-8 -*-
import math
import time

import numpy as np
import pandas as pd

from bin_elbow2 import bin_elbow2
from SS_partitioning import SS_partition
from segment_histogram import find_frequents, find_vertical_lines, plot_segments, normalize_data, \
    print_horizontal_lines, running_mean


def get_continuous_index(indexes):
    """
    If a subset are continuous numbers, put them together
    :param indexes: list
    :return: list of segments
    """
    segments = []
    if len(indexes) == 0:
        return segments
    start = 0
    end = 1
    while end < len(indexes):
        if indexes[end] - indexes[end - 1] > 1:
            segments.append([indexes[start], indexes[end - 1]])
            start = end
        end += 1
    segments.append([indexes[start], indexes[end - 1]])
    return segments


def get_G0_segments(dataset, elbow_value):
    """
    Get Global G0 segments (Everything above the elbow value)
    :param dataset:
    :param elbow_value:
    :return: list of segments' indexes, [[s1_start,s1_end],[s2_start,s2_end],...]
    """
    G0_indexes = [i for i, v in enumerate(dataset) if v > elbow_value]
    G0_segments = get_continuous_index(G0_indexes)
    print("Global G0 segments: {0}".format(G0_segments))
    return G0_segments


def generate_F(dataset, columns, moving_mean=50, normalize_bound=10000):
    """
    Calculate F, Distinguish G0 / G1 mode, Filter and Normalize
    :param dataset:
    :param columns: columns that need to be used
    :param moving_mean: like Matlab movmean()
    :param normalize_bound: To normalize to [0,normalize_bound]
    :return: F of global G1 (need to be segmented further), F, Global G0 segments, elbow_value
    """
    # Calculate moving distances
    X1_squared = [i ** 2 for i in np.diff(dataset[columns[0]])]
    Y1_squared = [i ** 2 for i in np.diff(dataset[columns[1]])]
    Z1_squared = [i ** 2 for i in np.diff(dataset[columns[2]])]
    result = np.sqrt(np.sum([X1_squared, Y1_squared, Z1_squared], axis=0))
    # Distinguish G0 / G1 mode
    sorted_result = sorted(result)
    elbow_index = bin_elbow2(sorted_result)
    elbow_value = sorted_result[elbow_index]
    print("Elbow value: {0}".format(elbow_value))
    # Everything above the elbow is G0
    G0_segments = get_G0_segments(result, elbow_value)
    G1_result = [x for x in result if x <= elbow_value]
    # Moving mean
    G1_result = running_mean(G1_result, moving_mean)  # dataset length will be shorten by (moving_mean-1)
    # Normalize
    G1_result = normalize_data(G1_result, normalize_bound)
    return G1_result, result, G0_segments, elbow_value


def adjust_index(horizontal_lines, Global_G0_segments):
    """
    Add Global G0 back to get complete index for horizontal_lines
    :param horizontal_lines: list of segments
    :param Global_G0_segments: list of segments
    :return: list of segments for new horizontal_lines
    """
    Global_G0_index = 0
    global_horizontal_lines = []
    sum = 0
    i = 0
    while i < len(horizontal_lines):
        while Global_G0_index < len(Global_G0_segments) and horizontal_lines[i][0]+sum > Global_G0_segments[Global_G0_index][0]:
            sum += Global_G0_segments[Global_G0_index][1] - Global_G0_segments[Global_G0_index][0] + 1
            Global_G0_index += 1
        global_horizontal_lines.append([horizontal_lines[i][0] + sum, horizontal_lines[i][1] + sum])
        i += 1
    return global_horizontal_lines


def print_with_value(global_horizontal_lines, F_complete):
    """
    print horizontal_lines segments with its F value
    :param global_horizontal_lines: list of segments
    :param F_complete: list
    :return: A list of [line_begin, line_end, line_value]
    """
    segments_with_value = []
    i = 0
    while i < len(global_horizontal_lines):
        mean = np.mean(F_complete[global_horizontal_lines[i][0]:global_horizontal_lines[i][1]])
        mean = round(mean, 4)
        segments_with_value.append([global_horizontal_lines[i][0], global_horizontal_lines[i][1], mean])
        i += 1
    return segments_with_value


def calc_TF_HV(global_horizontal_lines, total_length, TS=0.002):
    """
    Calculate overall time and fraction of horizontal lines vs. non-horizontal lines
    :param global_horizontal_lines:
    :param total_length:
    :param TS: one timestamp interval, 2ms by default
    :return: time of horizontal, fraction of horizontal, time of non-horizontal, fraction of non-horizontal
    """
    ndigits = 2  # how many digits in final results
    total_time = TS*(total_length-1)
    if len(global_horizontal_lines) == 0:
        return 0, 0, total_time, 100
    sum_horizontal = sum([item[1]-item[0] for item in global_horizontal_lines])
    time_H = round(sum_horizontal * TS, 3)
    fraction_H = round(time_H*100/total_time, ndigits)
    time_V = round(total_time-time_H, 3)
    fraction_V = round(100-fraction_H, ndigits)
    return time_H, fraction_H, time_V, fraction_V


def calc_SS_TF_HV(SS_segment, global_horizontal_lines, TS=0.002):
    """
    Calculate overall time and fraction of horizontal lines vs. non-horizontal lines for SS segment, and get horizontal
    lines in this SS segment
    :param SS_segment:
    :param global_horizontal_lines:
    :param TS: one timestamp interval, 2ms by default
    :return: list of horizontal lines in this SS segment, time of horizontal, fraction of horizontal,
    time of non-horizontal, fraction of non-horizontal
    """
    SS_horizontal_lines = []
    for global_horizontal_line in global_horizontal_lines:
        if global_horizontal_line[0] in range(SS_segment[0], SS_segment[1]) or global_horizontal_line[1] in range(SS_segment[0], SS_segment[1]):
            start = max(SS_segment[0], global_horizontal_line[0])
            end = min(SS_segment[1], global_horizontal_line[1])
            SS_horizontal_lines.append([start, end])
    time_H, fraction_H, time_V, fraction_V = calc_TF_HV(SS_horizontal_lines, SS_segment[1] - SS_segment[0], TS)
    return SS_horizontal_lines, time_H, fraction_H, time_V, fraction_V


def calc_SS_global_G0(SS_segment, Global_G0_segments, TS=0.002):
    """
    Print Global G0 info in this SS segment, global G0 is those above elbow
    :param SS_segment:
    :param Global_G0_segments:
    :param TS: one timestamp interval, 2ms by default
    :return: segments, time, fraction
    """
    ndigits = 2
    SS_global_G0 = []
    time_global_G0, frac_global_G0 = 0, 0
    for Global_G0_segment in Global_G0_segments:
        if Global_G0_segment[0] in range(SS_segment[0], SS_segment[1]) or Global_G0_segment[1] in range(SS_segment[0], SS_segment[1]):
            start = max(SS_segment[0], Global_G0_segment[0])
            end = min(SS_segment[1], Global_G0_segment[1])
            SS_global_G0.append([start, end])
    if len(SS_global_G0) > 0:
        sum_len = sum([item[1]-item[0] for item in SS_global_G0])
        time_global_G0 = round(sum_len * TS, 3)
        total_time = TS * (SS_segment[1] - SS_segment[0])
        frac_global_G0 = round(time_global_G0*100/total_time, ndigits)
    return SS_global_G0, time_global_G0, frac_global_G0


def calc_local_G0(SS_segment, SS_horizontal_lines, F_complete, elbow_value, TS=0.002):
    """
    all values above upper bound of the maximum bin with a horizontal line & below elbow in that Partition is local G0
    :param SS_segment:
    :param SS_horizontal_lines:
    :param F_complete:
    :param elbow_value:
    :param TS: one timestamp interval, 2ms by default
    :return: segments, time, fraction
    """
    ndigits = 2
    total_time = TS * (SS_segment[1] - SS_segment[0])
    bins_val = []
    for SS_horizontal_line in SS_horizontal_lines:
        bins_val.append(max(F_complete[SS_horizontal_line[0]:SS_horizontal_line[1]]))
    max_bin_val = max(bins_val)
    G0_indexes = [i for i, v in enumerate(F_complete) if max_bin_val<v<elbow_value and i in range(SS_segment[0],SS_segment[1])]
    SS_local_G0 = get_continuous_index(G0_indexes)
    time_local_G0 = round((sum([item[1]-item[0] for item in SS_local_G0])) * TS, 3)
    frac_local_G0 = round(time_local_G0*100/total_time, ndigits)
    return SS_local_G0, time_local_G0, frac_local_G0


def calc_G2(SS_segment, SS_horizontal_lines, F_complete, TS=0.002):
    """
    All values below lower bound of the min bin with a horizontal line in that Partition is G2
    :param SS_segment:
    :param SS_horizontal_lines:
    :param F_complete:
    :param TS: one timestamp interval, 2ms by default
    :return: segments, time, fraction
    """
    ndigits = 2
    total_time = TS * (SS_segment[1] - SS_segment[0])
    bins_val = []
    for SS_horizontal_line in SS_horizontal_lines:
        bins_val.append(min(F_complete[SS_horizontal_line[0]:SS_horizontal_line[1]]))
    min_bin_val = min(bins_val)
    G0_indexes = [i for i, v in enumerate(F_complete) if v < min_bin_val and i in range(SS_segment[0],SS_segment[1])]
    SS_G2 = get_continuous_index(G0_indexes)
    time_G2 = round((sum([item[1]-item[0] for item in SS_G2])) * TS, 3)
    frac_G2 = round(time_G2*100/total_time, ndigits)
    return SS_G2, time_G2, frac_G2


if __name__ == '__main__':
    # Please set these parameters manually
    Dataset_name = "Demo"  # Demo / Marv
    path = "Datasets/{0}/Trace/".format(Dataset_name)  # path for your "Trace" folder

    # <editor-fold desc="Parameters that can be left alone">
    moving_mean = 10  # Moving mean filter size
    TS_interval = 0.002  # one timestamp interval, 2ms by default
    measurements_file_name = "{0}measurements.csv".format(path)
    events_file_name = "{0}events.csv".format(path)
    columns = ['X1', 'Y1', 'Z1']  # for Demo
    if Dataset_name == "Marv":
        columns = ['X2', 'Y2', 'Z2']  # for Marv
    bin_size = 10  # size of a bin, Can leave it alone as dataset is normalized to (0,10000)
    # min_length = 200  # for horizontal lines, Can leave it alone as it's automatically decided by histogram
    debug = False  # if true, print log
    output_file = open("result_{0}.txt".format(Dataset_name), "w")  # for output
    # </editor-fold>

    start_time = time.time()

    # Read csv
    measurements = pd.read_csv(measurements_file_name, delimiter=";")
    events = pd.read_csv(events_file_name, delimiter=";")

    # Get SS segments
    SS_segments = SS_partition(measurements, events, debug)

    # Generate F
    F_G1, F_complete, Global_G0_segments, elbow_value = generate_F(measurements, columns, moving_mean)
    output_file.writelines("Elbow value for global G0/G1: {0} \n\n".format(elbow_value))
    output_file.writelines("Global G0 segments: {0} \n".format(Global_G0_segments))
    output_file.writelines("Global G0 fraction: {0}% \n\n".format(round((len(F_complete)-len(F_G1))*100/len(F_complete), 2)))

    # Step 1: find the frequent data (as peaks in histogram) and merge if it's continuous
    merged_frequents, bin_size, mean_len = find_frequents(F_G1, bin_size, debug)
    min_length = math.ceil(mean_len / 100) * 100  # Upper multiples of 100

    # Step 2: Examine each middle point and extend to find vertical segments
    vertical_lines = find_vertical_lines(F_G1, merged_frequents, bin_size, min_length, debug, True, True)
    print("Time: {0}s".format(round(time.time() - start_time)))
    print("Found {0} vertical parts".format(len(vertical_lines)))

    # Get horizontal segments (And add Global G0 back so we get complete index)
    shortened = max(moving_mean - 1, 0)
    horizontal_lines = print_horizontal_lines(vertical_lines, len(F_G1) + shortened, shortened)
    global_horizontal_lines = adjust_index(horizontal_lines, Global_G0_segments)
    horizontal_segments_valued = print_with_value(global_horizontal_lines, F_complete)
    output_file.writelines("F Horizontal segments with value:\n{0} \n\n".format(horizontal_segments_valued))

    # <editor-fold desc="Statistics">
    # 1. overall time and fraction of horizontal lines vs. non-horizontal lines
    time_H, fraction_H, time_V, fraction_V = calc_TF_HV(global_horizontal_lines, len(F_complete), TS_interval)
    output_file.writelines("Overall time of horizontal lines: {0}s, Fraction: {1}% \nOverall time of non-horizontal "
                           "lines: {2}s, Fraction: {3}% \n\n".format(time_H, fraction_H, time_V, fraction_V))
    # 2. per SS-Partition :
    # 2.1 time and fraction of horizontal lines vs. non-horizontal lines;
    # 2.2 time and fraction of local G0 (For each SS-Partition we identify all values above the maximum bin with a
    # horizontal line in that Partition as local G0);
    # 2.3 time and fraction of G2 (define all values below the min bin with a horizontal line in that Partition as G2)
    for SS_segment in SS_segments:
        SS_value = round(np.mean(measurements["SS"][SS_segment[0]: SS_segment[1]]))
        if SS_value > 0:
            SS_time = TS_interval * (SS_segment[1]-SS_segment[0])
            output_file.writelines("For SS_segment {0} : SS = {1}, time = {2}s\n".format(SS_segment, SS_value, SS_time))
            SS_horizontal_lines, time_H, fraction_H, time_V, fraction_V = calc_SS_TF_HV(SS_segment, global_horizontal_lines, TS_interval)
            output_file.writelines("1. time of horizontal lines = {0}s, Fraction = {1}% ; time of non-horizontal "
                           "lines = {2}s, Fraction = {3}% \n".format(time_H, fraction_H, time_V, fraction_V))
            SS_global_G0, time_global_G0, frac_global_G0 = calc_SS_global_G0(SS_segment, Global_G0_segments, TS_interval)
            output_file.writelines("2. Global G0: time = {1}s, Fraction = {2}%, segments: {0}\n".format(SS_global_G0, time_global_G0, frac_global_G0))
            SS_local_G0, time_local_G0, frac_local_G0 = calc_local_G0(SS_segment, SS_horizontal_lines, F_complete, elbow_value, TS_interval)
            output_file.writelines("3. Local G0: time = {1}s, Fraction = {2}%, segments: {0}\n".format(SS_local_G0, time_local_G0, frac_local_G0))
            SS_G2, time_G2, frac_G2 = calc_G2(SS_segment, SS_horizontal_lines, F_complete, TS_interval)
            output_file.writelines("4. G2: time = {1}s, Fraction = {2}%, segments: {0}\n".format(SS_G2, time_G2, frac_G2))
            time_none = round(time_V - time_global_G0 - time_local_G0 - time_G2, 3)
            frac_none = round(fraction_V - frac_global_G0 - frac_local_G0 - frac_G2, 2)
            output_file.writelines("5. None: time = {0}s, Fraction = {1}% \n\n".format(time_none, frac_none))
            plot_segments(F_complete, np.sort(np.array(SS_horizontal_lines).flatten()), SS_segment, "{0}_{1}".format(Dataset_name,SS_segment))
    # </editor-fold>

    # plot
    final_result = np.sort(np.array(global_horizontal_lines).flatten())
    print(",".join(str(i) for i in final_result))
    plot_segments(F_complete, final_result, [], Dataset_name)
