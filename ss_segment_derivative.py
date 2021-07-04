#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:lilypad
@File:ss_segment_derivative.py
@Time:2021/5/4 15:31
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_derivative(x, y):
    try:
        if len(x) == len(y):
            return np.diff(y) / np.diff(x)
        elif (len(x) - 1) == len(y):
            if len(pd.unique(np.diff(x))) == 1:
                x = x[:-1]
                return np.diff(y) / np.diff(x)
            else:
                x = [np.mean(x[i:i + 2]) for i in range(len(x) - 1)]
                return np.diff(y) / np.diff(x)
        else:
            raise Exception('Not supported')
    except:
        print("get_derivative error:", sys.exc_info()[0])


if __name__ == '__main__':
    measurements = pd.read_csv('measurements.csv', header=0, sep=';')
    # print(measurements.head())
    SS = measurements["SS"]  # Spindle Speed
    TS = measurements["TS"]  # Timestamp

    # correlation = np.corrcoef(SS, measurements["SC"])
    first_derivative = get_derivative(TS, SS)
    np.savetxt("SS_first_derivative.csv", first_derivative)

    second_derivative = get_derivative(TS, first_derivative)
    second_derivative = np.absolute(second_derivative)
    np.savetxt("SS_second_derivative_pos.csv", second_derivative)

    plt.plot(second_derivative)
    plt.show()
    segment_number = 200
    index = np.argpartition(second_derivative, -segment_number)[-segment_number:]
    print(index)
    print(second_derivative[index])
