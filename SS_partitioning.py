import os
import time
import pandas as pd
from math import ceil
import numpy as np
from sklearn import preprocessing
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


def add_events_to_measurements(measurements, events, properties):
    """
    Finds specified events in events-Dataframe and assigns values to each row of measurments-Dataframe in a specified
    column.

    :param measurements: measurements dataframe the events-values shall be assigned to
    :param events: events-dataframe containing the relevant events
    :param properties: A list of dictionaries, each containing

        1. eventName (string)
            The EventName to be find contained in the events-dataframe.
        2. shortName (string)
            The name of the measurements-column the event-values are assigned to.
        3. standardValue
            The standard value which will be assigned in an undefined case.
            For instance, measurement.TS < first specified_event.TS
            Example: 100.0 for Feedrate, -1 for Toolnumber
        4. columnOverwrite (bool)
            If shortname is already contained as a column in measurements, it will be overwritten
            (true) or skipped (false). If no such column exists, a new column is created.

    :return: A copy of the measurements-dataframe, now containing the specified event values.
    """

    for i, aDict in enumerate(properties):
        if aDict['shortName'] not in measurements.columns or aDict['columnOverwrite'] is True:
            """Filter Events by specified EventName"""
            specified_events = events.loc[events['EventName'] == aDict['eventName']].to_numpy()

            """Get TimeStamps"""
            time_m = measurements['TS'].to_numpy()
            time_e = specified_events[:, 0]

            """Preallocate memory for new column"""
            event_to_measurement = np.full((len(time_m), 1), aDict["standardValue"])

            """Assign values to each row of measurements"""
            for j, time_e_value in enumerate(time_e):
                diff = time_m - time_e_value
                idx = np.where(diff >= 0)[0]
                if idx.size > 0:
                    event_to_measurement[idx[0]:] = specified_events[j, 2]
            measurements[aDict['shortName']] = event_to_measurement

    return measurements


def find_step(a, pad=5):
    """
    Finds the step of the curve in a binary/linear search fashion.
    Expects a single downwards or upwards step in the given array.

    :param a: The (1-d) input array
    :param pad: Number of elements to pad left and right to improve robustness
    :return: The step bounds as start and end indices
    """
    curve = np.copy(a).astype(float)
    # Rescale y-axis to scale of x-axis
    curve /= curve.max()
    curve *= curve.shape[0] - 1
    curve = np.pad(curve, (pad, pad), constant_values=(curve[0], curve[-1]))
    # Initialize window to cover full curve
    window_size = a.shape[0]
    mask = np.arange(window_size)  # index mask for window of curve
    last_n = window_size
    n = ceil(last_n / 2)
    # Document last window difference on y-axis
    last_diff = -np.inf
    while 0 < n < last_n:
        # Find new window of size n (+1) with maximum difference on y-axis (or equivalently with maximum slope)
        window = curve[mask]
        minuend = window[n:]
        subtrahend = window[:-n]
        diff = np.abs(minuend - subtrahend)
        start = np.argmax(diff)
        max_diff = diff[start]
        # Early stopping if shrinkage on y-axis (compared to last window) is greater than on the x-axis
        if last_diff - max_diff < last_n - n:
            # Update state and window index mask, halve next window size
            mask = mask[start:start + n + 1]
            last_diff = max_diff
            last_n = n
            n = ceil(n / 2)
        else:
            # Increase window size linearly
            n += 1
    start, end = mask[0] - pad, mask[-1] - pad
    if start < 0:
        start = 0
        end = 1
    return start, end


def partition_signals_fast(data, window_size=1000, min_step=100):
    """
    Partitions the given data into subsequences with constant spindle speed SS by finding the steps first.
    Filters out partition boundaries with increasing or decreasing SS as well as subsequences including a tool change.

    :param data: Pandas dataframe containing the entire time series including spindle speed SS and tool numbers TN
    :param window_size: A window size (number of data points) large enough to cover all steps but small enough to not
                        cover a complete constant SS partition.
    :param min_step: A minimum absolute step size between two constant SS partitions to discover (do not set this value
                     too low in order to keep robustness)
    :return: A list of partition bounds (lists) determining the start and end indices of each partition
    """
    # Find indices of TN changes
    tn_changes = data.index[data["TN"].diff() != 0][1:]
    # Use unnormalized SS to apply absolute min_step parameter
    SS = data["SS"].to_numpy()
    ts_len = SS.shape[0]
    # Define start and end points of possible step windows
    end = SS[window_size:]
    start = SS[:-window_size]
    # Compute absolute step on y-axis for each window
    diff = np.abs(end - start)
    # Keep windows (start indices) with minimum step as candidates
    step_starts = np.where(diff >= min_step)[0]
    # Merge candidates corresponding to same step (neighbors) to blocks
    tmp = np.diff(step_starts, prepend=step_starts[0])
    tmp1 = np.where(tmp > 1)[0]
    step_start_blocks = np.split(step_starts, tmp1)
    # Normalization does not really matter here
    SS = preprocessing.scale(SS)
    bounds = []
    for block in step_start_blocks:
        # Find actual step window in candidate windows using a similar procedure as bin_elbow2
        window = SS[block[0]:block[-1]+window_size]
        start, end = find_step(window)
        # Only keep real steps (no single peaks) with long enough preceding and subsequent constant segments
        length = end - start
        if len(block) >= window_size - length:
            bounds.append([start+block[0], end+block[0]])
    step_bounds = np.array(bounds)
    # Compute partition bounds from step bounds and ignore partitions with TN changes
    partition_bounds = []
    start = 0
    cnt_changes = 0
    num_changes = tn_changes.shape[0]
    for step_start, step_end in step_bounds:
        if cnt_changes < num_changes and start <= tn_changes[cnt_changes] < step_start:
            cnt_changes += 1
        else:
            partition_bounds.append([start, step_start])
        start = step_end
    if cnt_changes >= num_changes or tn_changes[cnt_changes] < start:
        partition_bounds.append([start, ts_len])
    return partition_bounds


def get_partitions(measurements):
    """
    Returns the SS partition bounds of the given dataset.

    :param measurements: Pandas dataframe containing the entire time series including spindle speed SS and
                         tool numbers TN (only used if the bounds cannot be loaded)
    :return: List of partition bounds (lists) determining the start and end indices of each partition
    """
    start = time.time()
    # costs, max_cost, SS, partition_bounds = partition_signals(measurements)
    # partition_bounds = partition_signals(partition_results, measurements, num_processes)
    partition_bounds = partition_signals_fast(measurements)
    elapsed = time.time() - start
    print(f"Elpased Time for SS Partitioning: {elapsed}")
    print(f"Found {len(partition_bounds)} SS partitions")
    print("SS segments: {0}".format(partition_bounds))
    return partition_bounds


def SS_partition(measurements, events, debug=False):
    #path = os.path.join("D:/", "Trace")
    #measurements = pd.read_csv(os.path.join(path, "measurements.csv"), delimiter=";")
    if "TN" not in measurements.columns:
        #events = pd.read_csv(os.path.join(path, "events.csv"), delimiter=";")
        measurements = add_events_to_measurements(measurements, events, [{
            "eventName": "ACTUAL_TOOL_NUMBER_MANUAL",
            "shortName": "TN",
            "standardValue": -1,
            "columnOverwrite": True
        }])
    results = get_partitions(measurements)

    if debug:
        figure(figsize=(12, 6))
        plt.scatter(range(len(measurements["SS"])), measurements["SS"], s=5, label='data')
        for xc in results:
            plt.axvline(x=xc[0], color='r')
            plt.axvline(x=xc[1], color='r')
        plt.ylim(min(measurements["SS"]), max(measurements["SS"]))
        plt.legend()
        plt.show()

    return results