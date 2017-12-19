from pyml.system_paths import get_pyml_directory
import os
from collections import defaultdict


def to_number(s):
    try:
        float(s)
        return float(s)
    except ValueError:
        return s


def csv_parser(filename, column_names=None):

    """
    CSV file parser.

    Args:
        filename: path to file from pyml directory
        column_names (list): if None uses default names [1, 2, 3, etc.]

    Returns:
        dict: dictionary with keys with the name of column and values
            of each column
    """

    csv_to_dict = defaultdict(list)

    pyml_dir = get_pyml_directory()

    with open(os.path.join(pyml_dir, filename), 'r') as f:
        for row in f.readlines():
            row = row.strip()
            if len(row) > 0:
                for i, el in enumerate(row.split(',')):
                    csv_to_dict[i].append(to_number(el))

    if column_names is not None:
        for i, x in enumerate(column_names):
            csv_to_dict[x] = csv_to_dict.pop(i)

    return csv_to_dict
