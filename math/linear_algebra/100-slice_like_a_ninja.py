#!/usr/bin/env python3
"""
This module has a function that slices a matrix along
specific axes
"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    slices a matrix along specific axes
    axes is a dict, key is axis and value is tuple
    tuple is the slice to be made
    """
    # slices = [slice(None)] * matrix.ndim

    # # Update the slices based on the provided axes dictionary
    # for axis, slice_tuple in axes.items():
    #     slices[axis] = slice(*slice_tuple)

    # # Use numpy.newaxis to expand dimensions as needed
    # for axis, slice_tuple in axes.items():
    #     if len(slice_tuple) == 3:
    #         start, stop, step = slice_tuple
    #     elif len(slice_tuple) == 2:
    #         start, stop = slice_tuple
    #         step = 1
    #     else:
    #         raise ValueError("Invalid slice tuple length")

    #     new_mat = np.take(matrix, slice(start, stop, step), axis=axis)

    #     return new_mat

    result = matrix.copy()

    # Iterate through the axes and their corresponding slices
    for axis, slice_tuple in axes.items():
        # Use slice(None) to represent a full range if None is provided in the slice_tuple
        axis_slice = [slice(None) if s is None else slice(*s) for s in slice_tuple]

        # Apply the slice to the corresponding axis
        result = result[(Ellipsis, *axis_slice)]

    return result


