"""
Utilities for the stroke outcome package.

Functions for importing data, checking that the data holds sensible
values, and generic combining and sorting of data.

Also contains functions that are method-independent, for example the
calculation of post-stroke mRS distributions which is the same for
the continuous and discrete methods.
"""
import pandas as pd
import numpy as np
from importlib_resources import files
import numpy.typing as npt  # For type hinting.


def import_mrs_dists_from_file(include_notes=False):
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
    if include_notes:
        return mrs_dists, mrs_dists_notes
    else:
        return mrs_dists


def import_utility_dists_from_file(include_notes=False):
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
    if include_notes:
        return utility_dists, utility_dists_notes
    else:
        return utility_dists


def assign_patient_data(data_df: pd.DataFrame, trial: dict):
    """
    Pass a dataframe's data to the trial dictionary.

    Assumes that this input dataframe has column names that
    match keys in the trial dictionary. The trial keys can
    be found in the separate outcome model classes.

    Inputs:
    -------
    data_df - pd.DataFrame. The patient data. Any columns with names
              matching keys in the trial dictionary will be used to
              overwrite the trial dictionary values.
    trial   - dict. The existing trial dictionary. Each entry is an
              array of the same length as in data_df.

    Returns:
    --------
    trial - dict. The input trial dictionary with some data updated.
    """
    # For each entry in the trial dictionary...
    for key in list(trial.keys()):
        # ... check if the same name is a column of the dataframe...
        if key in data_df.columns:
            # ... and if so, overwrite the trial dictionary's data
            # with data from that column of the dataframe.
            trial[key].data = data_df[key].to_numpy().copy()
            # Keep the .copy() here to prevent updating the
            # original input data frame if the contents of trial
            # are changed later.
            if 'stroke_type_code' in key:
                # Make sure this is stored in both stroke_type_code
                # and stroke_type_code_on_input:
                for trial_key in ['stroke_type_code',
                                  'stroke_type_code_on_input']:
                    trial[trial_key].data = data_df[key].to_numpy().copy()
    return trial


def sanity_check_input_mrs_dists(mrs_dists: pd.DataFrame):
    """
    Check that a mRS dists pandas DataFrame is legit.

    Checks that all of the values in the DataFrame are numbers
    and that each mRS distribution has a name.
    """
    if len(mrs_dists) < 1:
        # If nothing was provided by the user, then
        # import the mRS dists from file now.
        mrs_dists = import_mrs_dists_from_file()

    # ##### Checks for mRS dists #####
    # Check that everything in the mRS dist arrays is a number.
    # Check that the dtype of each column of data is int or float.
    check_all_mrs_values_are_numbers_bool = np.all(
        [((np.dtype(mrs_dists[d]) == np.dtype('float')) |
            (np.dtype(mrs_dists[d]) == np.dtype('int')))
            for d in mrs_dists.columns]
    )
    if check_all_mrs_values_are_numbers_bool is False:
        exc_string = '''Some of the input mRS values are not numbers'''
        raise TypeError(exc_string) from None

    # Check that the pandas array has a named index column.
    if mrs_dists.index.dtype not in ['O']:
        print('The input mRS distributions might be improperly labelled.')
        # Just print warning, don't stop the code.

    # Return in case mRS distributions were imported from file here:
    return mrs_dists


def sanity_check_mrs_dists_for_stroke_type(
        mrs_distribution_probs: pd.DataFrame,
        stroke_type_codes: npt.ArrayLike,
        ivt_chosen_bool: npt.ArrayLike,
        mt_chosen_bool: npt.ArrayLike
        ):
    """
    Check that all required mRS distributions have been given.

    Sanity check the mRS distributions based on the stroke and treatment
    types. If a stroke type and treatment type combination
    exists in the stroke_type_code data,
    check that the matching mRS distributions have been given.
    """
    nlvo_ivt_keys = [
        'pre_stroke_nlvo',
        'no_treatment_nlvo',
        'no_effect_nlvo_ivt_deaths',
        't0_treatment_nlvo_ivt'
    ]
    lvo_mt_keys = [
        'pre_stroke_lvo',
        'no_treatment_lvo',
        'no_effect_lvo_mt_deaths',
        't0_treatment_lvo_mt'
    ]
    lvo_ivt_keys = [
        'pre_stroke_lvo',
        'no_treatment_lvo',
        'no_effect_lvo_ivt_deaths',
        't0_treatment_lvo_ivt'
    ]
    # Gather the list of keys that needs checking:
    keys_to_check = []
    if np.any((stroke_type_codes == 1) & (ivt_chosen_bool == 1)):
        # Check for nLVO data.
        keys_to_check += nlvo_ivt_keys
    if np.any((stroke_type_codes == 2) & (ivt_chosen_bool == 1)):
        # Check for LVO + IVT data.
        keys_to_check += lvo_ivt_keys
    if np.any((stroke_type_codes == 2) & (mt_chosen_bool == 1)):
        # Check for LVO + MT data.
        keys_to_check += lvo_mt_keys
    if np.any(stroke_type_codes == 0):
        # Currently we have no mRS distributions for
        # "other" stroke types.
        pass

    # Check whether all the required keys exist.
    # If not, add the missing values to a string and
    # at the end of the checks, print the error string.
    error_str = ''
    for key in keys_to_check:
        try:
            mrs_distribution_probs[key]
        except KeyError:
            error_str += f', {key}'
    if len(error_str) > 0:
        error_str = error_str[1:]  # Remove leading comma
        error_str = (
            'Expected the following mRS probability distributions:' +
            error_str)
        raise KeyError(error_str) from None


def sanity_check_utility_weights(utility_weights: npt.ArrayLike):
    """
    Check the input utility weights and replace them if necessary.

    Check that the input utility weights have seven values,
    one for each mRS score. Flatten the array if necessary.
    If the checks fail, use a set of default utility weights instead.
    """
    if np.size(utility_weights) == 7:
        # Use ravel() to ensure array shape of (7, ).
        return utility_weights.ravel()
    else:
        utility_weights_default = np.array(
            [0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

        if len(utility_weights) > 0:
            # i.e. if this isn't the default blank list:
            print(''.join([
                'Problem with the input utility weights. ',
                'Expected one weight per mRS score from 0 to 6. ',
                'Setting self.utility_weights to default values: ',
                f'{utility_weights_default}'
                ]))

        # Use the Wang et al. 2020 values:
        return utility_weights_default


def extract_mrs_probs_and_logodds(mrs_dists: pd.DataFrame):
    """
    Make dictionaries of mRS probs and logodds from a pandas DataFrame.

    The DataFrame must have seven columns, one for each mRS score,
    each named mRS<={x} for the score x. There must also be an index
    column for naming the dictionary keys.
    """
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


def assign_nlvo_with_mt_as_lvo(
        stroke_type_code: npt.ArrayLike,
        mt_chosen_bool: npt.ArrayLike
        ):
    """
    Find nLVO + MT patients and reassign them to LVO.

    Check whether any patients have an nLVO and receive MT.
    There are no probability distributions for that case.
    So change those patients' stroke type to LVO.
    """
    number_of_patients_with_nlvo_and_mt = len((
        (stroke_type_code == 1) &
        (mt_chosen_bool > 0)
        ).nonzero()[0])
    if number_of_patients_with_nlvo_and_mt == 0:
        return stroke_type_code
    else:
        # Change the stroke type to LVO.
        # Assume that the initial diagnosis was incorrect.
        inds_nlvo_and_mt = np.where((
            (stroke_type_code == 1) &
            (mt_chosen_bool > 0)
            ))[0]
        new_stroke_types = stroke_type_code
        new_stroke_types[inds_nlvo_and_mt] = 2
        return new_stroke_types


def assign_treatment_no_effect(
        treatment_chosen_bool: npt.ArrayLike,
        onset_to_treatment_mins: npt.ArrayLike,
        time_no_effect_mins: float
        ):
    """
    Assign which patients receive treatment after the no effect time.
    """
    # From inputs, calculate which patients are treated too late
    # for any effect. Recalculate this on each run in case any
    # of the patient data arrays have changed since the last run.
    no_effect_bool = (
        (treatment_chosen_bool > 0) &
        (onset_to_treatment_mins >= time_no_effect_mins)
        )
    return no_effect_bool


"""
####################
##### WRAPPERS #####
####################

This block of functions contains wrappers. The functions here
gather variables and pass them to other functions to do the
actual calculations.

The wrappers are for:
+ _calculate_probs_at_treatment_time()
"""


def calculate_post_stroke_mrs_dists_for_lvo_ivt(
        mrs_distribution_probs: dict,
        mrs_distribution_logodds: dict,
        trial: dict,
        ivt_time_no_effect_mins: float
        ):
    """
    Calculate post-stroke mRS dists for LVO treated with IVT.

    Wrapper for _calculate_probs_at_treatment_time() for LVO+IVT.

    Inputs:
    -------
    mrs_distribution_probs   - dict. Each entry is array of 7 floats.
                               Must contain the keys:
                                 no_treatment_lvo
                                 no_effect_lvo_ivt_deaths
    mrs_distribution_logodds - dict. Each entry is array of 7 floats.
                               Must contain the keys:
                                 no_effect_lvo_ivt_deaths
                                 t0_treatment_lvo_ivt
    trial                    - dict. Trial dictionary from the outcome
                               class (Discrete_outcome,
                               Continuous_outcome).
    ivt_time_no_effect_mins  - float. Time of no effect for IVT.

    Returns:
    --------
    post_stroke_probs - x by 7 array. Post-stroke mRS distributions,
                        one distribution per patient.
    """
    try:
        # Get relevant distributions
        not_treated_probs = \
            mrs_distribution_probs['no_treatment_lvo']
        no_effect_probs = \
            mrs_distribution_probs['no_effect_lvo_ivt_deaths']
        no_effect_logodds = \
            mrs_distribution_logodds['no_effect_lvo_ivt_deaths']
        t0_logodds = \
            mrs_distribution_logodds['t0_treatment_lvo_ivt']
    except KeyError:
        raise KeyError(
            'Need to create LVO mRS distributions first.')

    # Create an x by 7 grid of mRS distributions,
    # one row of 7 mRS values for each of x patients.
    mask_valid = (trial['stroke_type_code'].data == 2)
    post_stroke_probs = _calculate_probs_at_treatment_time(
        t0_logodds,
        no_effect_logodds,
        trial['onset_to_needle_mins'].data,
        ivt_time_no_effect_mins,
        trial['ivt_chosen_bool'].data,
        trial['ivt_no_effect_bool'].data,
        mask_valid,
        not_treated_probs,
        no_effect_probs
        )
    return post_stroke_probs


def calculate_post_stroke_mrs_dists_for_lvo_mt(
        mrs_distribution_probs: dict,
        mrs_distribution_logodds: dict,
        trial: dict,
        mt_time_no_effect_mins: float
        ):
    """
    Calculate post-stroke mRS dists for LVO treated with MT.

    Wrapper for _calculate_probs_at_treatment_time() for LVO+MT.

    Inputs:
    -------
    mrs_distribution_probs   - dict. Each entry is array of 7 floats.
                               Must contain the keys:
                                 no_treatment_lvo
                                 no_effect_lvo_mt_deaths
    mrs_distribution_logodds - dict. Each entry is array of 7 floats.
                               Must contain the keys:
                                 no_effect_lvo_mt_deaths
                                 t0_treatment_lvo_mt
    trial                    - dict. Trial dictionary from the outcome
                               class (Discrete_outcome,
                               Continuous_outcome).
    mt_time_no_effect_mins  - float. Time of no effect for MT.

    Returns:
    --------
    post_stroke_probs - x by 7 array. Post-stroke mRS distributions,
                        one distribution per patient.
    """
    try:
        # Get relevant distributions
        not_treated_probs = mrs_distribution_probs['no_treatment_lvo']
        no_effect_probs = mrs_distribution_probs[
            'no_effect_lvo_mt_deaths']
        no_effect_logodds = mrs_distribution_logodds[
            'no_effect_lvo_mt_deaths']
        t0_logodds = mrs_distribution_logodds['t0_treatment_lvo_mt']
    except KeyError:
        raise KeyError(
            'Need to create LVO mRS distributions first.')

    # Create an x by 7 grid of mRS distributions,
    # one row of 7 mRS values for each of x patients.
    mask_valid = (trial['stroke_type_code'].data == 2)
    post_stroke_probs = _calculate_probs_at_treatment_time(
        t0_logodds,
        no_effect_logodds,
        trial['onset_to_puncture_mins'].data,
        mt_time_no_effect_mins,
        trial['mt_chosen_bool'].data,
        trial['mt_no_effect_bool'].data,
        mask_valid,
        not_treated_probs,
        no_effect_probs
        )
    return post_stroke_probs


def calculate_post_stroke_mrs_dists_for_nlvo_ivt(
        mrs_distribution_probs: dict,
        mrs_distribution_logodds: dict,
        trial: dict,
        ivt_time_no_effect_mins: float
        ):
    """
    Calculate post-stroke mRS dists for nLVO treated with IVT.

    Wrapper for _calculate_probs_at_treatment_time() for nLVO+IVT.

    Inputs:
    -------
    mrs_distribution_probs   - dict. Each entry is array of 7 floats.
                               Must contain the keys:
                                 no_treatment_nlvo
                                 no_effect_nlvo_ivt_deaths
    mrs_distribution_logodds - dict. Each entry is array of 7 floats.
                               Must contain the keys:
                                 no_effect_nlvo_ivt_deaths
                                 t0_treatment_nlvo_ivt
    trial                    - dict. Trial dictionary from the outcome
                               class (Discrete_outcome,
                               Continuous_outcome).
    ivt_time_no_effect_mins  - float. Time of no effect for IVT.

    Returns:
    --------
    post_stroke_probs - x by 7 array. Post-stroke mRS distributions,
                        one distribution per patient.
    """
    try:
        # Get relevant distributions
        not_treated_probs = \
            mrs_distribution_probs['no_treatment_nlvo']
        no_effect_probs = \
            mrs_distribution_probs['no_effect_nlvo_ivt_deaths']
        no_effect_logodds = \
            mrs_distribution_logodds[
                'no_effect_nlvo_ivt_deaths']
        t0_logodds = \
            mrs_distribution_logodds['t0_treatment_nlvo_ivt']
    except KeyError:
        raise KeyError(
            'Need to create nLVO mRS distributions first.')

    # Create an x by 7 grid of mRS distributions,
    # one row of 7 mRS values for each of x patients.
    mask_valid = (trial['stroke_type_code'].data == 1)
    post_stroke_probs = _calculate_probs_at_treatment_time(
        t0_logodds,
        no_effect_logodds,
        trial['onset_to_needle_mins'].data,
        ivt_time_no_effect_mins,
        trial['ivt_chosen_bool'].data,
        trial['ivt_no_effect_bool'].data,
        mask_valid,
        not_treated_probs,
        no_effect_probs
        )
    return post_stroke_probs


def _calculate_probs_at_treatment_time(
        t0_logodds: npt.ArrayLike,
        no_effect_logodds: npt.ArrayLike,
        time_to_treatment_mins: npt.ArrayLike,
        time_no_effect_mins: float,
        mask_treated: npt.ArrayLike,
        mask_no_effect: npt.ArrayLike,
        mask_valid: npt.ArrayLike,
        not_treated_probs: npt.ArrayLike,
        no_effect_probs: npt.ArrayLike
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
        calculate_mrs_dist_at_treatment_time(
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


def calculate_mrs_dist_at_treatment_time(
        time_to_treatment_mins: npt.ArrayLike,
        time_no_effect_mins: float,
        t0_logodds: npt.ArrayLike,
        no_effect_logodds: npt.ArrayLike,
        final_value_is_mrs6: bool = True
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
    final_value_is_mrs6    - bool. Whether the final logodds value
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

    if final_value_is_mrs6 is True:
        # Manually set all of the probabilities for mRS<=6 to be 1
        # as the logodds calculation returns NaN.
        if len(treated_probs.shape) == 1:
            treated_probs[-1] = 1.0
        else:
            treated_probs[:, -1] = 1.0
    return treated_probs, treated_odds, treated_logodds


def calculate_patient_population_stats(trial: dict):
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
    stats_df = _make_stats_df(
        [nLVO_dict, LVO_dict, other_stroke_types_dict],
        labels=['nLVO', 'LVO', 'Other']
        )

    return stats_df


def _make_stats_dict(trial: dict, stroke_type_code: int):
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
    n_ivt = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['ivt_chosen_bool'].data > 0)
        ).nonzero()[0])
    # Number treated with MT
    n_mt = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['mt_chosen_bool'].data > 0)
        ).nonzero()[0])
    # Number treated with IVT after no-effect time
    n_ivt_no_effect = len((
        (trial['stroke_type_code'].data == stroke_type_code) &
        (trial['ivt_chosen_bool'].data > 0) &
        (trial['ivt_no_effect_bool'].data == 1)
        ).nonzero()[0])
    # Number treated with MT after no-effect time
    n_mt_no_effect = len((
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
        prop_ivt_of_stroke_type = n_ivt / n_stroke_type
        prop_mt_of_stroke_type = n_mt / n_stroke_type
        prop_ivt_no_effect_of_stroke_type = (
            n_ivt_no_effect / n_stroke_type)
        prop_mt_no_effect_of_stroke_type = (
            n_mt_no_effect / n_stroke_type)
        prop_no_treatment_of_stroke_type = (
            n_no_treatment / n_stroke_type)
    else:
        prop_ivt_of_stroke_type = np.NaN
        prop_mt_of_stroke_type = np.NaN
        prop_ivt_no_effect_of_stroke_type = np.NaN
        prop_mt_no_effect_of_stroke_type = np.NaN
        prop_no_treatment_of_stroke_type = np.NaN

    if n_total != 0:
        prop_stroke_type = n_stroke_type / n_total
        prop_ivt_of_total = n_ivt / n_total
        prop_mt_of_total = n_mt / n_total
        prop_ivt_no_effect_of_total = n_ivt_no_effect / n_total
        prop_mt_no_effect_of_total = n_mt_no_effect / n_total
        prop_no_treatment_of_total = n_no_treatment / n_total
    else:
        prop_stroke_type = np.NaN
        prop_ivt_of_total = np.NaN
        prop_mt_of_total = np.NaN
        prop_ivt_no_effect_of_total = np.NaN
        prop_mt_no_effect_of_total = np.NaN
        prop_no_treatment_of_total = np.NaN

    # Add all of this to the dictionary "sd":
    sd = dict()
    # Numbers:
    sd['n_stroke_type'] = n_stroke_type
    sd['n_total'] = n_total
    sd['n_ivt'] = n_ivt
    sd['n_mt'] = n_mt
    sd['n_ivt_no_effect'] = n_ivt_no_effect
    sd['n_mt_no_effect'] = n_mt_no_effect
    sd['n_no_treatment'] = n_no_treatment
    # Proportions:
    sd['prop_stroke_type'] = prop_stroke_type
    sd['prop_ivt_of_stroke_type'] = prop_ivt_of_stroke_type
    sd['prop_ivt_of_total'] = prop_ivt_of_total
    sd['prop_mt_of_stroke_type'] = prop_mt_of_stroke_type
    sd['prop_mt_of_total'] = prop_mt_of_total
    sd['prop_ivt_no_effect_of_stroke_type'] = prop_ivt_no_effect_of_stroke_type
    sd['prop_ivt_no_effect_of_total'] = prop_ivt_no_effect_of_total
    sd['prop_mt_no_effect_of_stroke_type'] = prop_mt_no_effect_of_stroke_type
    sd['prop_mt_no_effect_of_total'] = prop_mt_no_effect_of_total
    sd['prop_no_treatment_of_stroke_type'] = prop_no_treatment_of_stroke_type
    sd['prop_no_treatment_of_total'] = prop_no_treatment_of_total

    return sd


def _make_stats_df(
        stats_dicts: list, labels: list = ['nLVO', 'LVO', 'Other']):
    """
    Rearrange the stats dictionary into a pandas DataFrame.

    Inputs:
    -------
    stats_dict      - list of dicts. Each dict is the output from
                      _make_stats_dict(). Contains
                      various measures of the patient population.
    stroke_type_str - str. Name for the dataframe.

    Returns:
    --------
    df - pandas DataFrame. The info from stats_dict, rearranged by
         treatment category and proportion type.
    """
    # Use this column and row names:
    cols_each_stroke_type = [
        'Count',
        'Proportion of this stroke type',
        'Proportion of full cohort'
        ]
    cols_for_df = []
    index_for_df = [
        'Total',
        'IVT',
        'MT',
        'IVT no effect',
        'MT no effect',
        'No treatment'
    ]
    data_for_df = []

    for s, stats_dict in enumerate(stats_dicts):
        # Take data from the input stats dictionary.
        # Raw count of each category:
        counts = [
            stats_dict["n_stroke_type"],
            stats_dict["n_ivt"],
            stats_dict["n_mt"],
            stats_dict["n_ivt_no_effect"],
            stats_dict["n_mt_no_effect"],
            stats_dict["n_no_treatment"],
        ]
        # Proportion of this category out of this stroke type:
        props_of_this_stroke_type = [
            1.0,  # prop of this stroke type that have this stroke type
            stats_dict["prop_ivt_of_stroke_type"],
            stats_dict["prop_mt_of_stroke_type"],
            stats_dict["prop_ivt_no_effect_of_stroke_type"],
            stats_dict["prop_mt_no_effect_of_stroke_type"],
            stats_dict["prop_no_treatment_of_stroke_type"]
        ]
        # Proportion of this category out of the full cohort:
        props_of_full_cohort = [
            stats_dict["prop_stroke_type"],
            stats_dict["prop_ivt_of_total"],
            stats_dict["prop_mt_of_total"],
            stats_dict["prop_ivt_no_effect_of_total"],
            stats_dict["prop_mt_no_effect_of_total"],
            stats_dict["prop_no_treatment_of_total"]
        ]

        label = labels[s]
        cols_for_df += [label + ': ' + c for c in cols_each_stroke_type]
        data_for_df += [
            counts, props_of_this_stroke_type, props_of_full_cohort]

    # Place this data into one 2D array:
    data_for_df = np.vstack(data_for_df)

    # Convert to dataframe:
    df = pd.DataFrame(
        data=data_for_df.T,
        columns=cols_for_df,
        index=index_for_df
    )
    return df


def _merge_results_dicts(
        results_dicts: list,
        labels_for_dicts: list,
        final_dict: dict = {}
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


def _convert_eval_dict_to_dict(eval_dict: dict):
    """
    Convert a dictionary containing Evaluated_array to normal arrays.

    Inputs:
    -------
    eval_dict - dict. Dictionary containing Evaluated_array values.

    Returns:
    --------
    normal_dict - dict. Dictionary containing the same data as in the
                  original dictionary, but now in standard np.arrays.
    """
    normal_dict = {}
    for key in list(eval_dict.keys()):
        val = eval_dict[key].data
        normal_dict[key] = val
    return normal_dict


def extrapolate_odds_ratio(
        t_1: float,
        OR_1: float,
        t_2: float,
        OR_2: float,
        p_2: float,
        t_e: float = 0
        ):
    """
    Use two odds ratios to extrapolate the straight line fit and find
    the odds ratio at a given time, then convert to probability.

    The three time parameters MUST use the same units, e.g. hours.

    Inputs:
    t_1, t_2   - float. Times for data points 1 and 2.
    OR_1, OR_2 - float. Odds ratios at times t_1 and t_2.
    p_2        - float. Probability at time t_2.
    t_e        - float. Time to extrapolate the line to.

    Returns:
    OR_e - float. Extrapolated odds ratio at time t_e.
    p_e  - float. Extrapolated probability at time t_e.
    a, b - float. Constants for the straight line fit a+bt.
    """
    # Calculate "a", the log(odds ratio) at time t=0:
    a = (
        (np.log(OR_1) - (t_1/t_2)*np.log(OR_2)) /
        (1.0 - (t_1/t_2))
    )

    # Calculate "b", the gradient of the log(odds ratio) straight line.
    b = (np.log(OR_2) - np.log(OR_1)) / (t_2 - t_1)

    # Use these to calculate the odds ratio at time t_e:
    OR_e = np.exp(a + b * t_e)

    # Rearrange odds ratio formula:
    # ORe = {pe/(1-pe)} / {p2/(1-p2)}
    # pe/(1-pe) = ORe * p2/(1-p2)
    # Calculate R, the right-hand-side of this equation:
    R = OR_e * p_2 / (1 - p_2)

    # Rearrange pe/(1-pe)=R to find pe, probability at time t=t_e:
    p_e = R / (1 + R)

    return OR_e, p_e, a, b


def fudge_sum_one(dist: "list | np.array", dp: int =3):
    """
    Force sum of a distribution to be exactly 1.

    Add up all of the numbers to the given precision. The sum should
    be less than 1 exactly, but this function also works with the sum
    is greater than 1 exactly. Go through the numbers in turn and
    nudge the smallest fractions down or the largest fractions up as
    required until the sum is 1 exactly.

    Convert the numbers to large integers with a target value
    much larger than 1 because integers are easier to deal with.

    Inputs:
    -------
    dist - np.array. mRS distribution (non-cumulative).
    dp   - int. Number of decimal places that the final dist
           will be rounded to.

    Returns:
    -------
    dist_fudged - np.array. The distribution, rounded to the given
                  precision so that it sums to 1.
    """
    if np.round(np.sum(np.round(dist, dp)), dp) == 1.0:
        # Nothing to see here.
        return np.round(dist, dp)

    # Add or subtract from the mRS proportions until the sum is 1.
    # Start by adding to numbers with large fractional parts
    # or subtracting from numbers with small fractional parts.

    # Store the integer part of each value,
    # the fractional part of each value,
    # and values to track how many times the integer part
    # has been fudged upwards and downwards.

    # Split the values into the rounded and non-rounded parts:
    success = False
    while success == False:
        # Convert to integers.
        dist = dist * 10**dp
        dist_int = dist.astype(int)
        dist_frac = dist % dist.astype(int)
        target = 10 ** dp
        if np.all(dist_frac == 0.0):
            # If the dist is already rounded to the requested dp,
            # try the next digit up.
            dist = dist / 10**dp
            dp -= 1
        else:
            # Use this precision.
            success = True

    # Make a grid with four columns.
    arr = np.zeros((len(dist), 4), dtype=int)
    arr[:, 0] = dist_int
    arr[:, 1] = (dist_frac * 1000).astype(int)

    # Cut off this process after 20 loops.
    loops = 0
    sum_dist = np.sum(arr[:, 0])

    while loops < 20:
        if sum_dist < target:
            # Pick out the values that have been added to
            # the fewest times.
            min_change = np.min(arr[:, 2])
            inds_min_change = np.where(arr[:, 2] == min_change)
            # Of these, pick out the value with the largest
            # fractional part.
            largest_frac = np.max(arr[inds_min_change, 1])
            ind_largest_frac = np.where(
                (arr[:, 2] == min_change) &
                (arr[:, 1] == largest_frac)
            )
            if len(ind_largest_frac[0]) > 1:
                # Arbitrarily pick the lowest mRS if multiple options.
                ind_largest_frac = ind_largest_frac[0][0]
            # Add one to the final digit of this mRS proportion
            # and record the change in column 2.
            arr[ind_largest_frac, 0] += 1
            arr[ind_largest_frac, 2] += 1
        elif sum_dist > target:
            # Pick out the values that have been subtracted from
            # the fewest times.
            min_change = np.min(arr[:, 3])
            inds_min_change = np.where(arr[:, 3] == min_change)
            # Of these, pick out the value with the smallest
            # fractional part.
            smallest_frac = np.min(arr[inds_min_change, 1])
            ind_smallest_frac = np.where(
                (arr[:, 3] == min_change) &
                (arr[:, 1] == smallest_frac)
            )
            if len(ind_smallest_frac[0]) > 1:
                # Arbitrarily pick the lowest mRS if multiple options.
                ind_smallest_frac = ind_smallest_frac[0][0]
            # Subtract one from the final digit of this mRS proportion
            # and record the change in column 3.
            arr[ind_smallest_frac, 0] -= 1
            arr[ind_smallest_frac, 3] += 1

        # Have we finished?
        sum_dist = np.sum(arr[:, 0])
        if sum_dist == target:
            # Finish.
            loops = 20
        else:
            # Keep going round the "while" loop.
            loops += 1

    # Take the new fudged distribution.
    # Divide so that it now sums to 1 instead of the large target.
    dist_fudged = arr[:, 0] / 10**dp

    return dist_fudged
