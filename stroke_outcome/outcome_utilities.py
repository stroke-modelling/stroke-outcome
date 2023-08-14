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


def sanity_check_input_mrs_dists(mrs_dists):
    # ##### Checks for mRS dists #####
    # Check that everything in the mRS dist arrays is a number.
    # Check that the dtype of each column of data is int or float.
    check_all_mRS_values_are_numbers_bool = np.all(
        [((np.dtype(mrs_dists[d]) == np.dtype('float')) |
            (np.dtype(mrs_dists[d]) == np.dtype('int')))
            for d in mrs_dists.columns]
    )
    if check_all_mRS_values_are_numbers_bool is False:
        exc_string = '''Some of the input mRS values are not numbers'''
        raise TypeException(exc_string) from None

    # Check that the pandas array has a named index column.
    if mrs_dists.index.dtype not in ['O']:
        print('The input mRS distributions might be improperly labelled.')
        # Just print warning, don't stop the code.


def extract_mrs_probs_and_logodds(mrs_dists):
    # Store modified Rankin Scale distributions as arrays in dictionary
    mrs_distribution_probs = dict()
    mrs_distribution_logodds = dict()

    for index, row in mrs_dists.iterrows():
        p = np.array([row[f'mRS<={str(x)}'] for x in range(7)])
        mrs_distribution_probs[index] = p
        # Remove a tiny amount to prevent division by zero.
        p[np.where(p == 1.0)] = 1.0 - 1e-10
        # Convert to log odds
        o = p / (1 - p)
        mrs_distribution_logodds[index] = np.log(o)
    return mrs_distribution_probs, mrs_distribution_logodds


def _calculate_probs_at_treatment_time(
        t0_logodds,
        no_effect_logodds,
        time_to_treatment_mins,
        time_no_effect_mins,
        mask_treated,
        mask_no_effect,
        mask_valid,
        not_treated_probs,
        no_effect_probs
        ):
    """
    Calculates mRS distributions for treatment at a given time.

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

    This function can accept arrays for multiple patients as input.
    Each patient is assigned an mRS distribution for one of the
    following:
        - treated at input time
        - treated after time of no effect
        - not treated
        - not applicable (all values set to NaN)

    Example:
    Key:   mRS  ■ 0   ▥ 1   □ 2   ▤ 3   ▦ 4   ▣ 5   ▧ 6

    Example mRS distributions:
    Pre-stroke:   ■■■■■■■■▥▥▥▥▥▥▥▥▥▥□□□□□□□□▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣
    No effect:    ■■▥▥▥□□□□□□□▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣▣▣▧▧▧▧▧▧▧▧▧
    No treatment: ■■■▥▥▥▥□□□□□□□▤▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣▣▧▧▧▧▧▧▧▧

    First five patients' post-stroke mRS distributions:
    Patient 1:    ■■■■■■■▥▥▥▥▥▥▥▥▥□□□□□□□□▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣▧▧
    Patient 2:    ■■■▥▥▥□□□□□□□▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣▣▧▧▧▧▧▧▧▧▧
    Patient 3:    ■■■■▥▥▥▥□□□□□□□▤▤▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣▣▧▧▧▧▧▧
    Patient 4:    ■■▥▥▥□□□□□□□▤▤▤▤▤▤▤▤▦▦▦▦▦▦▦▦▦▦▣▣▣▣▣▣▣▣▣▣▧▧▧▧▧▧▧▧▧
    Patient 5:    -------------------------------------------------
                    (^ Patient 5 is set to invalid)
    ...

    Inputs:
    -------
    t0_logodds             - np.array. Log-odds at time zero. Can
                             provide one value per mRS score.
    no_effect_logodds      - np.array. Log-odds at time of no
                             effect. Can provide one value per mRS
                             score.
    time_to_treatment_mins - 1 by x array. Time to treatment in
                             minutes for each of x patients.
    time_no_effect_mins    - float. Time of no effect in minutes.
    mask_treated           - 1 by x array. True/False whether the
                             patient was treated, one value per
                             patient.
    mask_no_effect         - 1 by x array. True/False whether the
                             patient was treated after the time of
                             no effect, one value per patient.
    mask_valid             - 1 by x array. True/False whether the
                             patient falls into this category,
                             e.g. has the right occlusion type.
    not_treated_probs      - 1 by 7 array. mRS cumulative prob
                             distribution if patient is not
                             treated.
    no_effect_probs        - 1 by 7 array. mRS cumulative prob
                             distribution if patient is treated
                             after the time of no effect.

    Returns:
    --------
    treated_probs - x by 7 array. mRS cumulative probability
                    distribution(s) at the input treatment time(s).
    """

    # Reshape the arrays to allow for multiple treatment times.
    time_to_treatment_mins = \
        time_to_treatment_mins.reshape(len(time_to_treatment_mins), 1)
    no_effect_logodds = \
        no_effect_logodds.reshape(1, len(no_effect_logodds))
    t0_logodds = \
        t0_logodds.reshape(1, len(t0_logodds))

    treated_probs, treated_odds, treated_logodds = \
        calculate_mRS_dist_at_treatment_time(
            time_to_treatment_mins,
            time_no_effect_mins,
            t0_logodds,
            no_effect_logodds
        )

    # Overwrite these results for patients who do not receive
    # treatment or who are unaffected due to long treatment time.
    treated_probs[mask_treated == 0, :] = not_treated_probs
    treated_probs[mask_no_effect == 1, :] = no_effect_probs

    # Overwrite these results for patients who do not fall into
    # this category, for example who do not have the occlusion
    # type in question.
    treated_probs[mask_valid == 0, :] = np.NaN

    return treated_probs


def calculate_mRS_dist_at_treatment_time(
        time_to_treatment_mins,
        time_no_effect_mins,
        t0_logodds,
        no_effect_logodds,
        final_value_is_mRS6=True
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
    final_value_is_mRS6    - bool. Whether the final logodds value
                             is for mRS<=6. If True, the final
                             probabilities are all set to 1 to replace
                             the default Not A Number values.

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

    if final_value_is_mRS6 is True:
        # Manually set all of the probabilities for mRS<=6 to be 1
        # as the logodds calculation returns NaN.
        treated_probs[:, -1] = 1.0
    return treated_probs, treated_odds, treated_logodds


def calculate_patient_population_stats(trial):
    """
    Create dicts and dataframes of patient breakdown by treatment type.

    Prints numbers and proportions of patients in the following:
    - with each stroke type...
        - ... and treated with IVT
        - ... and treated with MT
        - ... and treated with IVT but no effect
        - ... and treated with MT but no effect
        - ... and not treated.

    Inputs:
    -------
    trial - dict. Contains the patient information
            from the stroke outcome model.
    """
    # Function to create dictionaries of patient statistics
    # for nLVO and LVO patients (nLVO_dict and LVO_dict).
    nLVO_dict = _make_stats_dict(trial, stroke_type_code=1)
    LVO_dict = _make_stats_dict(trial, stroke_type_code=2)
    other_stroke_types_dict = _make_stats_dict(
        trial, stroke_type_code=0)

    # Rearrange the same data into dataframes for easier reading.
    nLVO_df = _make_stats_df(nLVO_dict, 'nLVO')
    LVO_df = _make_stats_df(LVO_dict, 'LVO')
    other_stroke_types_df = _make_stats_df(
        other_stroke_types_dict, 'Other stroke types')

    return (nLVO_dict, LVO_dict, other_stroke_types_dict,
            nLVO_df, LVO_df, other_stroke_types_df)


def _make_stats_dict(trial, stroke_type_code):
    """
    Makes dict of stats for patients in each category.

    Stores number and proportions of patients in the following:
    - with this stroke type...
        - ... and treated with IVT
        - ... and treated with MT
        - ... and treated with IVT but no effect
        - ... and treated with MT but no effect
        - ... and not treated.

    Inputs:
    -------
    trial            - dict. Contains the patient information
                       from the stroke outcome model.
    stroke_type_code - int. 0 for other, 1 for nLVO, 2 for LVO.
                       Matches the code in stroke_type_code
                       patient array.

    Returns:
    --------
    stats_dict - dict. The resulting dictionary.
    """
    # Number of patients
    n_total = len(trial['stroke_type_code'].data)
    n_stroke_type = len((
        trial['stroke_type_code'].data ==
        stroke_type_code).nonzero()[0])

    # Number treated with IVT
    n_IVT = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['ivt_chosen_bool'].data > 0)
        ).nonzero()[0])
    # Number treated with MT
    n_MT = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['mt_chosen_bool'].data > 0)
        ).nonzero()[0])
    # Number treated with IVT after no-effect time
    n_IVT_no_effect = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['ivt_chosen_bool'].data > 0) &
        (trial['ivt_no_effect_bool'].data == 1)
        ).nonzero()[0])
    # Number treated with MT after no-effect time
    n_MT_no_effect = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['mt_chosen_bool'].data > 0) &
        (trial['mt_no_effect_bool'].data == 1)
        ).nonzero()[0])
    # Number not treated
    n_no_treatment = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['mt_chosen_bool'].data < 1) &
        (trial['ivt_chosen_bool'].data < 1)
        ).nonzero()[0])

    # Calculate proportions from the input numbers:
    if n_stroke_type != 0:
        prop_IVT_of_stroke_type = n_IVT / n_stroke_type
        prop_MT_of_stroke_type = n_MT / n_stroke_type
        prop_IVT_no_effect_of_stroke_type = (
            n_IVT_no_effect / n_stroke_type)
        prop_MT_no_effect_of_stroke_type = (
            n_MT_no_effect / n_stroke_type)
        prop_no_treatment_of_stroke_type = (
            n_no_treatment / n_stroke_type)
    else:
        prop_IVT_of_stroke_type = np.NaN
        prop_MT_of_stroke_type = np.NaN
        prop_IVT_no_effect_of_stroke_type = np.NaN
        prop_MT_no_effect_of_stroke_type = np.NaN
        prop_no_treatment_of_stroke_type = np.NaN

    if n_total != 0:
        prop_stroke_type = n_stroke_type / n_total
        prop_IVT_of_total = n_IVT / n_total
        prop_MT_of_total = n_MT / n_total
        prop_IVT_no_effect_of_total = n_IVT_no_effect / n_total
        prop_MT_no_effect_of_total = n_MT_no_effect / n_total
        prop_no_treatment_of_total = n_no_treatment / n_total
    else:
        prop_stroke_type = np.NaN
        prop_IVT_of_total = np.NaN
        prop_MT_of_total = np.NaN
        prop_IVT_no_effect_of_total = np.NaN
        prop_MT_no_effect_of_total = np.NaN
        prop_no_treatment_of_total = np.NaN

    # Add all of this to the dictionary:
    stats_dict = dict()
    # Numbers:
    stats_dict['n_stroke_type'] = n_stroke_type
    stats_dict['n_total'] = n_total
    stats_dict['n_IVT'] = n_IVT
    stats_dict['n_MT'] = n_MT
    stats_dict['n_IVT_no_effect'] = n_IVT_no_effect
    stats_dict['n_MT_no_effect'] = n_MT_no_effect
    stats_dict['n_no_treatment'] = n_no_treatment
    # Proportions:
    stats_dict['prop_stroke_type'] = prop_stroke_type
    stats_dict['prop_IVT_of_stroke_type'] = prop_IVT_of_stroke_type
    stats_dict['prop_IVT_of_total'] = prop_IVT_of_total
    stats_dict['prop_MT_of_stroke_type'] = prop_MT_of_stroke_type
    stats_dict['prop_MT_of_total'] = prop_MT_of_total
    stats_dict['prop_IVT_no_effect_of_stroke_type'] = \
        prop_IVT_no_effect_of_stroke_type
    stats_dict['prop_IVT_no_effect_of_total'] = \
        prop_IVT_no_effect_of_total
    stats_dict['prop_MT_no_effect_of_stroke_type'] = \
        prop_MT_no_effect_of_stroke_type
    stats_dict['prop_MT_no_effect_of_total'] = \
        prop_MT_no_effect_of_total
    stats_dict['prop_no_treatment_of_stroke_type'] = \
        prop_no_treatment_of_stroke_type
    stats_dict['prop_no_treatment_of_total'] = \
        prop_no_treatment_of_total

    return stats_dict


def _make_stats_df(stats_dict, stroke_type_str=''):
    """
    Rearrange the stats dictionary into a pandas DataFrame.

    Inputs:
    -------
    stats_dict      - dict. Output from _make_stats_dict(). Contains
                      various measures of the patient population.
    stroke_type_str - str. Name for the dataframe.

    Returns:
    --------
    df - pandas DataFrame. The info from stats_dict, rearranged by
         treatment category and proportion type.
    """
    # Use this column and row names:
    cols_for_df = [
        'Counts',
        f'Proportion of {stroke_type_str}',
        'Proportion of full cohort'
        ]
    index_for_df = [
        f'{stroke_type_str}',
        'IVT',
        'MT',
        'IVT no effect',
        'MT no effect',
        'No treatment'
    ]

    # Take data from the input stats dictionary.
    # Raw count of each category:
    counts = [
        stats_dict["n_stroke_type"],
        stats_dict["n_IVT"],
        stats_dict["n_MT"],
        stats_dict["n_IVT_no_effect"],
        stats_dict["n_MT_no_effect"],
        stats_dict["n_no_treatment"],
    ]
    # Proportion of this category out of this stroke type:
    props_of_this_stroke_type = [
        1.0,  # prop of this stroke type that have this stroke type
        stats_dict["prop_IVT_of_stroke_type"],
        stats_dict["prop_MT_of_stroke_type"],
        stats_dict["prop_IVT_no_effect_of_stroke_type"],
        stats_dict["prop_MT_no_effect_of_stroke_type"],
        stats_dict["prop_no_treatment_of_stroke_type"]
    ]
    # Proportion of this category out of the full cohort:
    props_of_full_cohort = [
        stats_dict["prop_stroke_type"],
        stats_dict["prop_IVT_of_total"],
        stats_dict["prop_MT_of_total"],
        stats_dict["prop_IVT_no_effect_of_total"],
        stats_dict["prop_MT_no_effect_of_total"],
        stats_dict["prop_no_treatment_of_total"]
    ]

    # Place this data into one 2D array:
    data_for_df = np.vstack((
        counts, props_of_this_stroke_type, props_of_full_cohort))

    # Convert to dataframe:
    df = pd.DataFrame(
        data=data_for_df.T,
        columns=cols_for_df,
        index=index_for_df
    )
    return df


def _merge_results_dicts(
        results_dicts,
        labels_for_dicts,
        final_dict={}
        ):
    """
    Merge multiple dictionaries into one dictionary.

    For example, the same key from three dictionaries:
        nlvo_ivt_dict['mean_added_utility']
        lvo_ivt_dict['mean_added_utility']
        lvo_mt_dict['mean_added_utility']
    becomes three entries in the combined dictionary:
        final_dict['nlvo_ivt_mean_added_utility']
        final_dict['lvo_ivt_mean_added_utility']
        final_dict['lvo_mt_mean_added_utility']

    Inputs:
    -------
    results_dicts    - list of dicts. The dictionaries to be
                       combined.
    labels_for_dicts - list of strings. Labels for the
                       dictionaries for their keys in the combo
                       dictionary.

    Returns:
    --------
    final_dict - dict. The combined dictionary.
    """
    for d, result_dict in enumerate(results_dicts):
        label = labels_for_dicts[d] + '_'
        for (key, value) in zip(result_dict.keys(), result_dict.values()):
            new_key = label + key
            final_dict[new_key] = value

    return final_dict
