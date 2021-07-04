#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:lilypad
@File:ss_segment_LR.py
@Time:2021/5/5 9:58
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def fit_LR(x, y):
    # reshape data to standard input format for LinearRegression
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    # fit a straight line (y=kx+b) on these 2D data points
    LR = LinearRegression().fit(x, y)
    # return the accuracy (best possible score is 1.0 and it can be negative)
    return LR.score(x, y)


if __name__ == '__main__':
    # read certain columns
    columns = ['TS', 'SS']
    measurements = pd.read_csv('measurements.csv', usecols=columns, header=0, sep=';')
    # adjustable parameters
    min_len = 30  # min length of a segment
    crop = 0.2  # determine when to crop
    start = 0  # start point of a segment
    step = 5  # step length
    end = start + step  # end point of a segment
    segments_index = []  # stores indexes of segments (final result)
    scores = []  # scores of all segments
    # detecting straight lines from the beginning of the file
    while end < len(measurements):
        dataset = measurements[start:end]  # current dataset that's being processing
        score = fit_LR(dataset[columns[0]], dataset[columns[1]])  # How good it fits a straight line
        # record the initial score of a new segment : pre_score
        if (end - start) == step:
            pre_score = fit_LR(dataset[columns[0]], dataset[columns[1]])
        # if score grows, should be a better straight line
        if score > pre_score:
            pre_score = score  # keep pre_score as high as possible for better segment
            end += step  # extend the segment
        # if score declines noticeably, crop, record the segment
        elif pre_score - score > crop * pre_score:
            if (end - start) > min_len:
                if start not in segments_index:
                    segments_index.append(start)
                segments_index.append(end)
                scores.append(score)
            start = end  # set starting point for a new segment
            end = start + step
        else:
            end += step
    print(len(segments_index))
    print(segments_index)
    print(np.mean(scores))
    print(scores)
    plt.plot(measurements[columns[0]], measurements[columns[1]])
    plt.show()
    for xc in segments_index:
        plt.axvline(x=xc)
    plt.legend()
    plt.show()
