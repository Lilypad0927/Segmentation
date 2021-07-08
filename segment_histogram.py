#!usr/bin/env python
# -*- coding:utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def merge_frequents(frequent_bins, bin_size):
    """
    Some frequents are continuous, thus should be treated as one large bin.
    :param frequent_bins: list of bins from histogram
    :param bin_size: the size of one bin
    :return: A list of [merged bin value, its number of continuous bins]
    """
    i = 1
    merged_frequents = []
    while i < len(frequent_bins):
        if frequent_bins[i] - frequent_bins[i - 1] <= bin_size:
            # found continuous bin
            tmp = i - 1  # start point
            i += 1
            merged_num = np.mean(frequent_bins[tmp:i])
            while i < len(frequent_bins):
                if frequent_bins[i] - frequent_bins[i - 1] > bin_size:
                    merged_num = np.mean(frequent_bins[tmp:i])
                    i += 1
                    if i == len(frequent_bins):
                        merged_frequents.append([frequent_bins[i - 1], 1])
                    break
                i += 1
            # (i - tmp - 1) bins form this merged bin
            merged_frequents.append([merged_num, i - tmp - 1])
        else:
            # not continuous, directly append
            merged_frequents.append([frequent_bins[i - 1], 1])
            i += 1
            if i == len(frequent_bins):
                merged_frequents.append([frequent_bins[i - 1], 1])
    merged_frequents = sorted(merged_frequents, key=lambda x: x[0])
    return merged_frequents


def find_frequents(dataset, bin_size, debug=False):
    """
    Build a histogram to find frequent values (tall bins above average), And merge
    :param dataset: list
    :param bin_size: size of one bin
    :param debug: whether to print log
    :return: list of merged frequents, real bin_size, average bin height
    """
    data_range = np.ptp(dataset)  # get data range
    bin_num = math.ceil(data_range / bin_size)  # number of bins
    interval = data_range / bin_num  # real bin size (usually smaller than setting)
    values, bins, patches = plt.hist(dataset, bins=bin_num)
    plt.clf()  # clear histogram figure
    frequent_bins = []
    mean_val = np.mean(values)  # average bin height
    for i in range(len(values)):
        if values[i] > mean_val:
            frequent_bins.append(bins[i])  # get tall bins above average
    merged_frequents = merge_frequents(frequent_bins, bin_size)  # merge continuous tall bins to one
    if debug:
        print("bin num: {0}, interval: {1}".format(bin_num, round(interval)))
        print("mean value in histogram: {0}, max: {1}".format(round(mean_val), np.max(values)))
        print("frequent set: ", [round(num) for num in frequent_bins])
        print("merged set: ", [[round(num), cnt] for num, cnt in merged_frequents])
    return merged_frequents, interval, mean_val


def find_middle_points(middle_value, dataset, min_length, debug=False):
    """
    Given a horizontal line y=middle_value, find closest points in dataset to this line
    :param middle_value: Y-axis value
    :param dataset: list
    :param min_length: min length for a segment to be considered a wanted horizontal line
    :param debug: whether to print log
    :return: A list of indexes in dataset
    """
    middle_point_index = []
    j = 0
    while j < len(dataset):
        try:
            # find the intersection, i.e. middle point value should be between dataset_n and dataset_n+1
            j = next(item[0] for item in enumerate(dataset) if item[0] > j and
                     (item[1] - middle_value) * (dataset[item[0] - 1] - middle_value) <= 0)
            middle_point_index.append(j)
            j += min_length
        except StopIteration:
            break
    if debug:
        print("middle value: {0}, found index: {1}".format(round(middle_value), middle_point_index))
    return middle_point_index


def find_nearest_bin(middle_index, merged_frequents, direction, dataset, bin_size, min_length):
    """
    From middle_index, Search for a closest point whose Y-value is in a bin from merged_frequents.
    And make sure it reaches a "horizontal" segment. (How to define "horizontal" is important)
    :param middle_index: index of middle point in dataset
    :param merged_frequents: Also can be thought as Y-values of horizontal lines
    :param direction: positive -> right, negative -> left. Value is step size.
    :param dataset: list
    :param bin_size: size of one bin
    :param min_length: min length for a segment to be considered a wanted horizontal line
    :return: Index of the found data point and the corresponding bin from merged_frequents
    """
    cur = middle_index + direction
    while cur in range(0, len(dataset) - 1):
        for frequent in merged_frequents:
            # some horizontal lines are noisy, thus one original bin_size is not enough to cover it
            cur_bin_size = frequent[1] * bin_size
            # check whether reach a bin from merged_frequents
            if abs(dataset[cur] - frequent[0]) <= cur_bin_size:
                # make sure it reaches a long-enough horizontal line
                tmp = cur + min_length * direction
                # make sure the index doesn't go out of range
                tmp = min(tmp, len(dataset) - 1)
                tmp = max(tmp, 0)
                tmp_dataset = dataset[min(cur, tmp): max(cur, tmp)]
                # the mean of the dataset should be close to the frequent value
                # divide tmp_dataset into 2 parts for accuracy
                mean_diff1 = abs(np.mean(tmp_dataset[:math.floor(len(tmp_dataset)/3)]) - frequent[0])
                mean_diff2 = abs(np.mean(tmp_dataset[math.floor(len(tmp_dataset)/3):]) - frequent[0])
                # reduce cur_bin_size for accuracy
                if frequent[1] > 2:
                    cur_bin_size /= 2
                if mean_diff1 <= cur_bin_size and mean_diff2 <= cur_bin_size:
                    return cur, frequent
                else:
                    # "horizontal" segment not found yet, keep looking
                    cur += direction
        cur += direction
    # if this return's used, it means the algorithm reached the end of the dataset but still didn't find "horizontal"
    return cur - direction, [dataset[cur - direction], 1]


def find_vertical_lines(dataset, merged_frequents, bin_size, min_length, debug=False, same_bin=False,
                        final_check=False):
    """
    Examine each middle point and look for its left and right ends, to form a vertical segment
    :param final_check: Whether to do a final check for all segments
    :param dataset: list
    :param merged_frequents: Also can be thought as Y-values of horizontal lines
    :param bin_size: size of one bin
    :param min_length: min length for a segment to be considered a wanted horizontal line
    :param debug: whether to print log
    :param same_bin: Whether to allow left and right ends to be in the same bin
    :return: A list of vertical_lines' indexes, [[line1_left, line1_right],[line2_left, line2_right],...]
    """
    vertical_lines = []
    i = 1
    while i < len(merged_frequents):
        # 1. Find middle points
        middle_value = np.mean(([item[0] for item in merged_frequents])[i - 1:i + 1])
        middle_point_index = find_middle_points(middle_value, dataset, min_length, debug)
        # 2. Examine each and look for both left and right ends to form a vertical segment
        for middle_index in middle_point_index:
            # abandon if the point is already in a detected vertical line
            for line in vertical_lines:
                if middle_index in range(line[0], line[1]):
                    break
            else:
                # index for right and left end
                right_index, right_bin = find_nearest_bin(middle_index, merged_frequents, 1, dataset, bin_size,
                                                          min_length)
                left_index, left_bin = find_nearest_bin(middle_index, merged_frequents, -1, dataset, bin_size,
                                                        min_length)
                continuous_cnt = max(right_bin[1], left_bin[1])  # number of the contained continuous bins
                if right_index - left_index > 3:
                    # same_bin parameter decides whether the left and right ends should be in the same bin
                    if same_bin or abs(dataset[right_index] - dataset[left_index]) > continuous_cnt * bin_size:
                        # there might be same vertical lines from different middle points, drop them
                        for line in vertical_lines:
                            if abs(line[0] - left_index) < min_length or abs(line[1] - right_index) < min_length:
                                break
                        else:
                            vertical_lines.append([left_index, right_index, left_bin[0], right_bin[0]])
        i += 1
    vertical_lines = sorted(vertical_lines, key=lambda x: x[0])
    if final_check:
        if debug:
            print("Vertical lines: {0}".format([[row[0], row[1]] for row in vertical_lines]))
        vertical_lines = check_horizontal_shape(vertical_lines, dataset, debug)
        if debug:
            print("Checked vertical lines: {0}".format(vertical_lines))
    else:
        vertical_lines = [[row[0], row[1]] for row in vertical_lines]
    return vertical_lines


def check_horizontal_shape(vertical_lines, dataset, debug=False):
    """
    Final checking for the segments' shapes. Correct the result if wrong.
    A "horizontal" segment should look like a short rectangle, a "vertical" segment should look like a tall rectangle
    :param vertical_lines: current result of segment points, to be checked
    :param dataset: list
    :param debug: whether to print log
    :return: A list of vertical_lines' indexes (segment points)
    """
    if len(vertical_lines) == 1:
        return [vertical_lines[0], vertical_lines[1]]
    final_vertical_lines = []
    screen_ratio = 16 / 9  # assume the whole dataset is drawn on a plot with 16:9 screen ratio
    upper_quantile = 0.85  # ignore some extremely noisy data
    lower_quantile = 0.15
    rectangle_ratio = 5 / 1  # expect a "vertical" segment to be taller than this rectangle_ratio
    reshape_factor = (len(dataset) / np.ptp(dataset)) / screen_ratio  # range of X / range of Y
    i = 0
    while i < len(vertical_lines) - 1:
        # if the segment between two vertical parts are not horizontal, merge as one vertical part
        tmp_dataset = dataset[vertical_lines[i][1]: vertical_lines[i + 1][0]]
        y_len = abs(np.quantile(tmp_dataset, lower_quantile) - np.quantile(tmp_dataset, upper_quantile))
        # expect a "horizontal" segment to be shorter than a square
        if (vertical_lines[i + 1][0] - vertical_lines[i][1]) / reshape_factor < y_len:
            if debug:
                print("Not seem horizontal: {0}".format([vertical_lines[i][1], vertical_lines[i + 1][0]]))
            start = vertical_lines[i][0]
            i += 1
            while i < len(vertical_lines) - 1:
                tmp_dataset = dataset[vertical_lines[i][1]: vertical_lines[i + 1][0]]
                y_len = abs(np.quantile(tmp_dataset, lower_quantile) - np.quantile(tmp_dataset, upper_quantile))
                if (vertical_lines[i + 1][0] - vertical_lines[i][1]) / reshape_factor >= y_len:
                    final_vertical_lines.append([start, vertical_lines[i][1]])
                    i += 1
                    break
                i += 1
        else:
            # same logic to check "vertical" segment
            tmp_dataset = dataset[vertical_lines[i][0]: vertical_lines[i][1]]
            if vertical_lines[i][1] - vertical_lines[i][0] < np.ptp(tmp_dataset) / rectangle_ratio:
                final_vertical_lines.append([vertical_lines[i][0], vertical_lines[i][1]])
            else:
                if debug:
                    print("Not seem vertical: {0}".format([vertical_lines[i][0], vertical_lines[i][1]]))
            i += 1
            if i == len(vertical_lines) - 1:
                tmp_dataset = dataset[vertical_lines[i][0]: vertical_lines[i][1]]
                if vertical_lines[i][1] - vertical_lines[i][0] < np.ptp(tmp_dataset) / rectangle_ratio:
                    final_vertical_lines.append([vertical_lines[i][0], vertical_lines[i][1]])
    if debug:
        print("check_horizontal_shape: Changed {0} part".format(len(vertical_lines) - len(final_vertical_lines)))
    return final_vertical_lines


def print_horizontal_lines(vertical_lines, dataset_length, shortened_len=0):
    """
    Print horizontal lines indexes.
    :param vertical_lines:
    :param dataset_length: the length of entire dataset
    :param shortened_len: the length of shortened part (due to moving mean)
    """
    horizontal_lines = []
    shortened_len = math.ceil(0.8 * shortened_len)
    if vertical_lines[0][0] > 0:
        horizontal_lines.append([0, vertical_lines[0][0] - 1 + shortened_len])
    i = 0
    while i < len(vertical_lines) - 1:
        horizontal_lines.append(
            [vertical_lines[i][1] + 1 + shortened_len, vertical_lines[i + 1][0] - 1 + shortened_len])
        i += 1
    if vertical_lines[len(vertical_lines) - 1][1] < dataset_length - 1:
        horizontal_lines.append([vertical_lines[len(vertical_lines) - 1][1] + 1 + shortened_len, dataset_length - 1])
    print("horizontal lines: {0}".format(horizontal_lines))
    return horizontal_lines


def plot_segments(dataset, result, SS=[], name=""):
    """
    Draw the plot of original dataset and segment points
    :param dataset:
    :param result: segment points
    """
    figure(figsize=(12, 6))
    x = np.arange(len(dataset))
    plt.plot(x, dataset, lw=0.1)
    for xc in result:
        plt.axvline(x=xc, color='r', lw=0.1)
    plt.ylim(min(dataset), max(dataset))
    if len(SS) > 1:
        plt.xlim(SS[0], SS[1])
    plt.legend()
    plt.savefig('plot_segments_{0}.pdf'.format(name))
    plt.close()


def normalize_data(data, upper):
    """
    Normalize the dataset to [0,upper]
    :param data:
    :param upper:
    :return: new list of data
    """
    return (data - np.min(data)) * upper / (np.max(data) - np.min(data))


def running_mean(data, size):
    """
    Equivalent to Matlab's movmean()
    :param data:
    :param size:
    :return: new list of data which is filtered/smoothed
    """
    if size <= 0:
        return data
    # dataset length will be shorten by (size-1)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[size:] - cumsum[:-size]) / float(size)
