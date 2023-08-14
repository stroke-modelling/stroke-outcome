"""
Utilities for the stroke outcome package.

Contents:
+ import_mrs_dists_from_file
+ import_utility_dists_from_file
+ calculate_mRS_dist_at_treatment_time
"""
import pandas as pd
import numpy as np
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


def calculate_mRS_dist_at_treatment_time(
        time_to_treatment_mins,
        time_no_effect_mins,
        t0_logodds,
        no_effect_logodds
        ):
    """
    Calculate the mRS distribution at arbitrary treatment time(s).

    If the input log-odds arrays contain one value per mRS score,
    then the output treated probabilities will make the
    mRS distribution at the treatment time.

    The new distributions are created by calculating log-odds at
    the treatment time. For each mRS band, the method is:

    l |                Draw a straight line between the log-odds
    o |x1    treated   at time zero and the time of no effect.
    g |  \    at "o"   Then the log-odds at the chosen treatment
    o |    \           time lies on this line.
    d |      o
    d |        \
    s |__________x2__
            time

    The (x,y) coordinates of the two points are:
        x1: (0, t0_logodds)
        x2: (time_no_effect_mins, no_effect_logodds)
        o:  (time_to_treatment_mins, treated_logodds)

    The log-odds are then translated to odds and probability:
        odds = exp(log-odds)
        prob = odds / (1 + odds)

    Inputs:
    -------
    time_to_treatment_mins - np.array. The time to treatment in
                             minutes. Can contain multiple values so
                             long as the array shape is (X, 1).
    time_no_effect_mins    - float. Time of no effect in minutes.
    t0_logodds             - np.array. Log-odds at time zero. Can
                             provide one value per mRS score.
    no_effect_logodds      - np.array. Log-odds at time of no
                             effect. Can provide one value per mRS
                             score.

    Returns:
    --------
    treated_probs   - np.array. mRS probability distribution at the
                      treatment time(s).
    treated_odds    - np.array. As above, but converted to odds.
    treated_logodds - np.array. As above, but converted to log-odds.
    """
    # Calculate fraction of time to no effect passed
    frac_to_no_effect = time_to_treatment_mins / time_no_effect_mins

    # Combine t=0 and no effect distributions based on time passed
    treated_logodds = ((frac_to_no_effect * no_effect_logodds) +
                       ((1 - frac_to_no_effect) * t0_logodds))

    # Convert to odds and probabilties
    treated_odds = np.exp(treated_logodds)
    treated_probs = treated_odds / (1 + treated_odds)

    # Manually set all of the probabilities for mRS<=6 to be 1
    # as the logodds calculation returns NaN.
    treated_probs[:, -1] = 1.0
    return treated_probs, treated_odds, treated_logodds
