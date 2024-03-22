import random

import numpy as np

def normalize_values(value_list, ranges):
    normalized_values = []
    for i, value in enumerate(value_list):
        field_name = ranges[i]['field']
        min_value, max_value = ranges[i]['range']
        normalized_value = float((value - min_value) / (max_value - min_value))
        normalized_values.append(normalized_value)
    return normalized_values



def string_to_array(string):
    values = string.split(',')
    values = [float(val) for val in values]
    ranges = [
        {'field': 'year', 'range': (1980, 2024)},
        {'field': 'month', 'range': (0, 12)},
        {'field': 'day', 'range': (0, 31)},
        {'field': 'gsd', 'range': (0, 5)},
        {'field': 'cloud_cover', 'range': (0, 100)},
        {'field': 'latitude', 'range': (-90, 90)},
        {'field': 'longitude', 'range': (-180, 180)},
    ]

    normalized_values = normalize_values(values, ranges)
    normalized_values = np.array(normalized_values)

    return normalized_values