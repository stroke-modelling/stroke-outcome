"""
Utilities for the stroke outcome package.

Contents:
+ import_mrs_dists_from_file
+ import_utility_dists_from_file
"""
import pandas as pd
from importlib_resources import files


def import_mrs_dists_from_file():
    """Import cumulative mRS distributions as pandas DataFrame."""
    filename = files('stroke_outcome.data').joinpath(
        'mrs_dist_probs_cumsum.csv')

    # Import all lines of the file:
    with open(filename) as f:
        mrs_dists_notes = f.readlines()
    # Find the number of lines in the header.
    # Make a list of True for header lines and False for data lines...
    header_bool = [True if line[0] == '#' else False
                   for line in mrs_dists_notes]
    # ... and find where the first False value is:
    n_lines_header = header_bool.index(False)

    # Store the preamble text as a string:
    mrs_dists_notes = ''.join(mrs_dists_notes[:n_lines_header])

    # Import the table data:
    mrs_dists = pd.read_csv(
        filename,
        index_col='Stroke type',
        skiprows=n_lines_header  # Avoid the header.
        )
    return mrs_dists, mrs_dists_notes


def import_utility_dists_from_file():
    """Import utility distributions as pandas DataFrame."""
    filename = files('stroke_outcome.data').joinpath(
        'utility_dists.csv')

    # Import all lines of the file:
    with open(filename) as f:
        utility_dists_notes = f.readlines()
    # Find the number of lines in the header.
    # Make a list of True for header lines and False for data lines...
    header_bool = [True if line[0] == '#' else False
                   for line in utility_dists_notes]
    # ... and find where the first False value is:
    n_lines_header = header_bool.index(False)

    # Store the preamble text as a string:
    utility_dists_notes = ''.join(utility_dists_notes[:n_lines_header])

    # Import the table data:
    utility_dists = pd.read_csv(
        filename,
        index_col='Name',
        skiprows=n_lines_header  # Avoid the header.
        )
    return utility_dists, utility_dists_notes
