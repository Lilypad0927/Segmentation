import numpy as np
import math

def bin_elbow2(a, pad=5):
    """
    Finds the largest elbow of the curve in a binary/linear search fashion.
    Expects a monotonically increasing input curve.

    :param a: The (1-d) input array
    :param pad: Number of elements to pad left and right to improve robustness
    :return: The index of the elbow
    """
    curve = np.copy(a).astype(float)
    # Rescale y-axis to scale of x-axis
    curve /= curve.max()
    curve *= curve.shape[0] - 1
    curve = np.pad(curve, (pad, pad), constant_values=(curve[0], curve[-1]))
    # Initialize window to cover full curve
    window_size = curve.shape[0]
    mask = np.arange(window_size)  # index mask for window of curve
    last_n = window_size
    n = math.ceil(last_n / 2)
    # Document last window difference on y-axis
    last_diff = -np.inf
    while 0 < n < last_n:
        # Find new window of size n (+1) with maximum difference on y-axis (or equivalently with maximum slope)
        window = curve[mask]
        minuend = window[n:]
        subtrahend = window[:-n]
        diff = minuend - subtrahend
        start = np.argmax(diff)
        max_diff = diff[start]
        # Early stopping if shrinkage on y-axis (compared to last window) is greater than on the x-axis
        if last_diff - max_diff < last_n - n:
            # Update state and window index mask, halve next window size
            mask = mask[start:start + n + 1]
            last_diff = max_diff
            last_n = n
            n = math.ceil(n / 2)
        else:
            # Increase window size linearly
            n += 1
    # Return index of left window border as elbow index
    elbow_index = mask[0] - pad
    if elbow_index < 0:
        print("Warning: Could not find elbow; returning 0 as elbow index")
        elbow_index = 0
    return elbow_index