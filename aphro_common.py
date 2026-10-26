"""Shared helpers for the APHRODITE preprocessing scripts.

Missing grid cells are stored as MASK_VALUE (-99) in every saved array.
fill() turns a masked array into a plain ndarray with MASK_VALUE,
and mask() turns such an ndarray back into a masked array.
"""
import numpy.ma as ma

MASK_VALUE = -99


def fill(x, mask_value=MASK_VALUE):
    return ma.filled(x, fill_value=mask_value)


def mask(x, mask_value=MASK_VALUE):
    return ma.masked_where(x <= mask_value, x)
