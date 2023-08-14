import numpy as np
import pandas as pd

from .evaluated_array import Evaluated_array
from .outcome_utilities import \
    _calculate_probs_at_treatment_time, \
    _merge_results_dicts, \
    calculate_patient_population_stats, \
    sanity_check_input_mrs_dists, \
    extract_mrs_probs_and_logodds


class Continuous_outcome:
    """
    Predicts modified Rankin Scale (mRS) distributions for ischaemic stroke
    patients depending on time to treatment with intravenous thrombolysis (IVT)
    or mechanical thrombectomy (MT). Results are broken down for large vessel
    occulusions (LVO) and non large vessel occlusions (nLVO).

    Inputs
    ------

    mrs_dists - Pandas DataFrame object. DataFrame of mRS
                cumulative probability distributions for:
      1) Not treated nLVO
      2) nLVO treated at t=0 (time of stroke onset) with IVT
      3) nLVO treated at time of no-effect (includes treatment deaths)
      4) Not treated LVO
      5) LVO treated at t=0 (time of stroke onset) with IVT
      6) LVO treated with IVT at time of no-effect (includes treatment deaths)
      7) LVO treated at t=0 (time of stroke onset) with IVT
      8) LVO treated with IVT at time of no-effect (includes treatment deaths)
    number_of_patients      - int. The number of patients in the array.
    utility_weights         - np.array. The utility weight for each mRS
                              score. Contains seven values.
    ivt_time_no_effect_mins - float. Time of no effect for
                              thrombolysis in minutes.
    mt_time_no_effect_mins  - float. Time of no effect for
                              thrombectomy in minutes.


    Outputs
    -------
    A results dictionary with entries for each of these three
    categories:
    - nLVO
    - LVO not treated with MT
    - LVO treated with MT
    Each category contains the following info:
    - each_patient_post_stroke_mrs_dist
    - not_treated_mean_mrs
    - no_effect_mean_mrs
    - each_patient_post_stroke_mean_mrs
    - each_patient_mean_mrs_shift
    - not_treated_mean_utility
    - no_effect_mean_utility
    - each_patient_post_stroke_mean_utility
    - each_patient_mean_added_utility
    - proportion_of_whole_cohort_improved
    - mean_valid_patients_mean_mrs_shift
    - mean_valid_patients_mean_added_utility
    - treated_population_mean_valid_patients_mean_mrs_shift
    - treated_population_mean_valid_patients_mean_added_utility

    The full_cohort_outcomes results dictionary
    takes the results from the separate results dictionaries and
    pulls out the relevant parts for each patient category
    (nLVO+IVT, LVO+IVT, LVO+MT).
    The output arrays contain x values, one for each patient.
    Contents of returned dictionary:
    - each_patient_post_stroke_mrs_dist                     x by 7 grid
    - each_patient_post_stroke_mean_mrs                        x floats
    - each_patient_mean_mrs_shift                              x floats
    - each_patient_post_stroke_mean_utility                    x floats
    - each_patient_mean_added_utility                          x floats
    - post_stroke_mean_mrs                                      1 float
    - mean_mrs_shift                                            1 float
    - mean_utility                                              1 float
    - mean_added_utility                                        1 float

    Utility-weighted mRS
    --------------------

    In addition to mRS we may calculate utility-weighted mRS. Utility is an
    estimated quality of life (0=dead, 1=full quality of life, neagtive numbers
    indicate a 'worse than death' outcome).

    If not given explicitly, the following mRS utility scores are used.
    mRS Utility scores are based on a pooled Analysis of 20 000+ Patients. From
    Wang X, Moullaali TJ, Li Q, Berge E, Robinson TG, Lindley R, et al.
    Utility-Weighted Modified Rankin Scale Scores for the Assessment of Stroke
    Outcome. Stroke. 2020 Aug 1;51(8):2411-7.

    | mRS Score | 0    | 1    | 2    | 3    | 4    | 5     | 6    |
    |-----------|------|------|------|------|------|-------|------|
    | Utility   | 0.97 | 0.88 | 0.74 | 0.55 | 0.20 | -0.19 | 0.00 |

    General methodology
    -------------------

    The model assumes that log odds of mRS <= x declines uniformally with time.
    The imported distribution give mRS <= x probabilities at t=0 (time of
    stroke onset) and time of no effect. These two distributions are converted
    to log odds and weighted according to the fraction of time, in relation to
    when the treatment no longer has an effect, that has passed. The weighted
    log odds distribution is converted back to probability of mRS <= x. mRS
    are also converted to a utility-weighted mRS.

    If not given explicitly, the time to no-effect is taken as:
    1) 6.3 hours for IVT
      (from Emberson et al, https://doi.org/10.1016/S0140-6736(14)60584-5.)
    2) 8 hours for MT
      (from Fransen et al; https://doi.org/10.1001/jamaneurol.2015.3886.
      this analysis did not include late-presenting patients selected by
      advanced imaging).

    The shift in mRS for each patient
    between not_treated and treated distribution is calculated. A negative
    shift is indicative of improvement (lower mRS disability score).


    Usage:
    ------
    Update the arrays of patient details in the trial dictionary.
    Each will run the sanity checks in the Evaluated_array class
    and display an error message if invalid data is passed in.
    Then run the main calculate_outcomes() function.
    Example:
        # Initiate the object:
        continuous_outcome = Continuous_outcome(
            mrs_dists={pandas dataframe}, number_of_patients=100)
        # Import patient data:
        continuous_outcome.trial['onset_to_needle_mins'].data = {array 1}
        continuous_outcome.trial['ivt_chosen_bool'].data = {array 2}
        continuous_outcome.trial['stroke_type_code'].data = {array 3}
        # Calculate outcomes:
        results_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())


    Limitations and notes:
    ----------------------
    For MT, there are only mRS probability distributions for patients with
    LVOs. If a patient with an nLVO is treated with MT, the patient details
    are quietly updated to reassign them as LVO and so use the base LVO
    mRS distributions for pre-stroke, no-treatment, and no-effect.

    Some patients receive both IVT and MT. Separate outcomes are calculated
    for each case and stored in the LVO+IVT and LVO+MT arrays. However in the
    combined full cohort outcomes, only one of the sets of outcomes can be
    used. The chosen set is the one with the better improvement in outcomes.
    The separate LVO+IVT and LVO+MT arrays are then *not* updated to remove
    the rejected set of outcomes, so any mean shifts in values across the
    LVO+IVT group (for example) will include data that was not actually used
    in the combined full cohort outcomes.

    Internally, stroke types are referred to by a number rather than their
    full name. The codes are:
      0 - "other" stroke type
      1 - nLVO
      2 - LVO
    """

    def __init__(
            self,
            mrs_dists: pd.DataFrame,
            number_of_patients: int,
            utility_weights: np.array = np.array([]),
            ivt_time_no_effect_mins: float = 378.0,
            mt_time_no_effect_mins: float = 480.0
            ):
        """
        Constructor for continuous clinical outcome model.

        Input:
        ------
        - mRS distributions for
          - not treated,
          - t=0 treatment, and
          - treatment at time of no effect
            (which also includes treatment-related excess deaths).
        - number of patients, for setting array sizes.

        Initialises:
        ------------
        - mRS distributions in logodds
        - Time of no effect for IVT
        - Time of no effect for MT
        - Weights for converting mRS score to utility

        Initialises arrays of patient data for:
          - Stroke type (code: 0 "other" stroke types, 1 nLVO, 2 LVO)
          - Time to IVT (minutes)
          - Treated with IVT (True/False)
          - Time to MT (minutes)
          - Treated with MT (True/False)
          - IVT had no effect (True/False)
          - MT had no effect (True/False)
        Each patient contributes one value to each of these arrays.
        Each array is initalised using a class so that the data is
        passed through a series of sanity checks (e.g. the time to
        IVT array will reject values of ['cat', 'dog']).
        """
        self.name = "Continuous clinical outcome model"

        sanity_check_input_mrs_dists(mrs_dists)
        # Store the input for the repr() string.
        self.mrs_dists_input = mrs_dists

        #
        # ##### Set up model parameters #####
        # Store modified Rankin Scale distributions as arrays in dictionary
        self.mrs_distribution_probs, self.mrs_distribution_logodds = \
            extract_mrs_probs_and_logodds(mrs_dists)

        # Set general model parameters
        self.ivt_time_no_effect_mins = ivt_time_no_effect_mins
        self.mt_time_no_effect_mins = mt_time_no_effect_mins

        # Store utility weightings for mRS 0-6
        if np.size(utility_weights) == 7:
            # Use ravel() to ensure array shape of (7, ).
            self.utility_weights = utility_weights.ravel()
        else:
            self.utility_weights = np.array(
                [0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])
            print(''.join([
                'Problem with the input utility weights. ',
                'Expected one weight per mRS score from 0 to 6. ',
                'Setting self.utility_weights to default values: ',
                f'{self.utility_weights}'
                ]))

        #
        # ##### Patient data setup #####
        # All arrays must contain this many values:
        self.number_of_patients = number_of_patients

        # Evaluated_array(
        #    number_of_patients, valid_dtypes_list, valid_min, valid_max)
        n = self.number_of_patients  # Defined to shorten the following.
        self.trial = dict(
            stroke_type_code=Evaluated_array(n, ['int'], 0, 2),
            onset_to_needle_mins=Evaluated_array(n, ['float'], 0.0, np.inf),
            ivt_chosen_bool=Evaluated_array(n, ['int', 'bool'], 0, 1),
            ivt_no_effect_bool=Evaluated_array(n, ['int', 'bool'], 0, 1),
            onset_to_puncture_mins=Evaluated_array(
                n, ['float'], 0.0, np.inf),
            mt_chosen_bool=Evaluated_array(n, ['int', 'bool'], 0, 1),
            mt_no_effect_bool=Evaluated_array(n, ['int', 'bool'], 0, 1),
        )

    def __str__(self):
        """Prints info when print(Instance) is called."""
        print_str = ''.join([
            f'There are {self.number_of_patients} patients ',
            'and the base mRS distributions are: ',
        ])
        for (key, val) in zip(
                self.mrs_distribution_probs.keys(),
                self.mrs_distribution_probs.values()
                ):
            print_str += '\n'
            print_str += f'{key} '
            print_str += f'{repr(val)}'

        print_str += '\n\n'
        print_str += ('The utility weights are: ' + f'{self.utility_weights}')

        print_str += '\n\n'
        print_str += ''.join([
            'The useful input data is stored in the trial dictionary. ',
            'The following arrays should be provided to `trial`:\n',
            '- stroke_type_code\n',
            '- onset_to_needle_mins\n',
            '- ivt_chosen_bool\n',
            '- onset_to_puncture_mins\n',
            '- mt_chosen_bool\n',
            ])

        print_str += ''.join([
            '\n',
            'The easiest way to create the results dictionaries is:\n',
            '  results_by_stroke_type, full_cohort_outcomes = ',
            'continuous_outcome.calculate_outcomes()'
            ])
        return print_str

    def __repr__(self):
        """Prints how to reproduce this instance of the class."""
        # This string prints without actual newlines, just the "\n"
        # characters, but it's the best way I can think of to display
        # the input dataframe in full.
        return ''.join([
            'Continous_outcome(',
            f'mrs_dists=DATAFRAME*, '
            f'number_of_patients={self.number_of_patients}',
            f'utility_weights={self.utility_weights}',
            f'ivt_time_no_effect_mins={self.ivt_time_no_effect_mins}',
            f'mt_time_no_effect_mins={self.mt_time_no_effect_mins})',
            '        \n\n        ',
            'The dataframe DATAFRAME* is created with: \n',
            f'  index: {self.mrs_dists_input.index}, \n',
            f'  columns: {self.mrs_dists_input.columns}, \n',
            f'  values: {repr(self.mrs_dists_input.values)}'
            ])

    def assign_ivt_no_effect(self):
        """
        Assign which patients receive IVT after the no effect time.
        """
        # From inputs, calculate which patients are treated too late
        # for any effect. Recalculate this on each run in case any
        # of the patient data arrays have changed since the last run.
        self.trial['ivt_no_effect_bool'].data = (
            (self.trial['ivt_chosen_bool'].data > 0) &
            (self.trial['onset_to_needle_mins'].data >=
             self.ivt_time_no_effect_mins)
            )

    def assign_mt_no_effect(self):
        """
        Assign which patients receive MT after the no effect time.
        """
        # From inputs, calculate which patients are treated too late
        # for any effect. Recalculate this on each run in case any
        # of the patient data arrays have changed since the last run.
        self.trial['mt_no_effect_bool'].data = (
            (self.trial['mt_chosen_bool'].data > 0) &
            (self.trial['onset_to_puncture_mins'].data >=
             self.mt_time_no_effect_mins)
            )

    """
    ################
    ##### MAIN #####
    ################

    This function runs everything important to find the final results.
    """
    def calculate_outcomes(self):
        """
        Calls methods to model mRS populations for:
        1) LVO not treated
        2) nLVO not treated
        3) LVO treated with IVT
        4) LVO treated with MT
        5) nLVO treated with IVT

        These are converted into cumulative probabilties, mean mRS, mRS shift,
        and proportion of patients with improved mRS.

        Returns:
        --------
        A results dictionary with entries for each of these three
        categories:
        - nLVO treated with IVT
        - LVO treated with IVT
        - LVO treated with MT
        Each category contains the following info:
        - each_patient_post_stroke_mrs_dist
        - not_treated_mean_mrs
        - no_effect_mean_mrs
        - each_patient_post_stroke_mean_mrs
        - each_patient_mean_mrs_shift
        - not_treated_mean_utility
        - no_effect_mean_utility
        - each_patient_post_stroke_mean_utility
        - each_patient_mean_added_utility
        - mean_valid_patients_mean_mrs_shift
        - mean_valid_patients_mean_added_utility'

        The full_cohort_outcomes results dictionary
        takes the results from the separate results dictionaries and
        pulls out the relevant parts for each patient category
        (nLVO+IVT, LVO+IVT, LVO+MT).
        The output arrays contain x values, one for each patient.
        Contents of returned dictionary:
        - each_patient_post_stroke_mrs_dist                 x by 7 grid
        - each_patient_post_stroke_mean_mrs                    x floats
        - each_patient_mean_mrs_shift                          x floats
        - each_patient_post_stroke_mean_utility                x floats
        - each_patient_mean_added_utility                      x floats
        - post_stroke_mean_mrs                                  1 float
        - mean_mrs_shift                                        1 float
        - mean_utility                                          1 float
        - mean_added_utility                                    1 float
        """
        # ##### Sanity checks #####

        # Do an extra check that the number of patients is matched
        # by all arrays in case the value was updated after
        # initialisation.
        patient_array_labels = [
            'stroke type (code)',
            'time to IVT (mins)',
            'received IVT (bool)',
            'time to MT (mins)',
            'received MT (bool)',
            'IVT no effect (bool)',
            'MT no effect (bool)'
        ]
        patient_array_vars = [
            self.trial['stroke_type_code'].data,
            self.trial['onset_to_needle_mins'].data,
            self.trial['ivt_chosen_bool'].data,
            self.trial['onset_to_puncture_mins'].data,
            self.trial['mt_chosen_bool'].data,
            self.trial['ivt_no_effect_bool'].data,
            self.trial['mt_no_effect_bool'].data
            ]
        length_warning_str = ''.join([
            'The following patient data arrays contain a different number ',
            'of patients than the instance value of ',
            f'{self.number_of_patients}:',
            '\n'
            ])
        for (val, key) in zip(patient_array_vars, patient_array_labels):
            if len(val) != self.number_of_patients:
                print(length_warning_str + '- ' + key +
                      f', length {len(val)}')
                length_warning_str = ''

        # Check if anyone has an nLVO and receives MT
        # (for which we don't have mRS probability distributions)
        number_of_patients_with_nLVO_and_MT = len((
            (self.trial['stroke_type_code'].data == 1) &
            (self.trial['mt_chosen_bool'].data > 0)
            ).nonzero()[0])
        if number_of_patients_with_nLVO_and_MT > 0:
            # Change the stroke type to LVO.
            # Assume that the initial diagnosis was incorrect.
            inds_nLVO_and_MT = np.where((
                (self.trial['stroke_type_code'].data == 1) &
                (self.trial['mt_chosen_bool'].data > 0)
                ))[0]
            new_stroke_types = self.trial['stroke_type_code'].data
            new_stroke_types[inds_nLVO_and_MT] = 2
            self.trial['stroke_type_code'].data = new_stroke_types

        # Determine who receives treatment too late for any effect:
        self.assign_ivt_no_effect()
        self.assign_mt_no_effect()

        # ##### Statistics #####
        # Function to create dictionaries of patient statistics
        # for each stroke type. The dataframes (_df) contain the same
        # information but organised into a table for easier reading.
        (self.nLVO_dict,
         self.LVO_dict,
         self.other_stroke_types_dict,
         self.nLVO_df,
         self.LVO_df,
         self.other_stroke_types_df
         ) = calculate_patient_population_stats(self.trial)

        # ##### Calculations #####
        # Get treatment results
        # These post-stroke mRS distributions are the same for any
        # measure of change of outcome...
        post_stroke_probs_lvo_ivt = \
            self.calculate_post_stroke_mrs_dists_for_lvo_ivt()
        post_stroke_probs_lvo_mt = \
            self.calculate_post_stroke_mrs_dists_for_lvo_mt()
        post_stroke_probs_nlvo_ivt = \
            self.calculate_post_stroke_mrs_dists_for_nlvo_ivt()
        # ... and these outcome dictionaries are specific to this
        # measure of change of outcome:
        lvo_ivt_outcomes = self.calculate_outcomes_dict_for_lvo_ivt(
            post_stroke_probs_lvo_ivt)
        lvo_mt_outcomes = self.calculate_outcomes_dict_for_lvo_mt(
            post_stroke_probs_lvo_mt)
        nlvo_ivt_outcomes = self.calculate_outcomes_dict_for_nlvo_ivt(
            post_stroke_probs_nlvo_ivt)

        # Gather results into one dictionary:
        outcomes_by_stroke_type_and_treatment = _merge_results_dicts(
            [lvo_ivt_outcomes, lvo_mt_outcomes, nlvo_ivt_outcomes],
            ['lvo_ivt', 'lvo_mt', 'nlvo_ivt'],
            )

        # Get the mean results for the actual patient cohort:
        full_cohort_outcomes = self._calculate_patient_outcomes(
            outcomes_by_stroke_type_and_treatment)
        return outcomes_by_stroke_type_and_treatment, full_cohort_outcomes

    """
    ####################
    ##### WRAPPERS #####
    ####################

    This block of functions contains wrappers. The functions here
    gather variables and pass them to other functions to do the
    actual calculations.

    Scroll down if you're looking for the calculations!

    The wrappers are for:
    + _calculate_probs_at_treatment_time()
    + _calculate_outcomes_dict()
      + ... which is itself a wrapper(!!) for
        _create_mrs_utility_dict().
    """
    def calculate_post_stroke_mrs_dists_for_lvo_ivt(self):
        """
        Wrapper for _calculate_probs_at_treatment_time() for LVO+IVT.

        Calculate post-stroke mRS dists for LVO treated with IVT.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_lvo']
            no_effect_probs = \
                self.mrs_distribution_probs['no_effect_lvo_ivt_deaths']
            no_effect_logodds = \
                self.mrs_distribution_logodds[
                    'no_effect_lvo_ivt_deaths']
            t0_logodds = \
                self.mrs_distribution_logodds['t0_treatment_lvo_ivt']
        except KeyError:
            raise KeyError(
                'Need to create LVO mRS distributions first.')

        # Create an x by 7 grid of mRS distributions,
        # one row of 7 mRS values for each of x patients.
        mask_valid = (self.trial['stroke_type_code'].data == 2)
        post_stroke_probs = _calculate_probs_at_treatment_time(
            t0_logodds,
            no_effect_logodds,
            self.trial['onset_to_needle_mins'].data,
            self.ivt_time_no_effect_mins,
            self.trial['ivt_chosen_bool'].data,
            self.trial['ivt_no_effect_bool'].data,
            mask_valid,
            not_treated_probs,
            no_effect_probs
            )
        return post_stroke_probs

    def calculate_post_stroke_mrs_dists_for_lvo_mt(self):
        """
        Wrapper for _calculate_probs_at_treatment_time() for LVO+MT.

        Calculate post-stroke mRS dists for LVO treated with MT.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_lvo']
            no_effect_probs = \
                self.mrs_distribution_probs['no_effect_lvo_mt_deaths']
            no_effect_logodds = \
                self.mrs_distribution_logodds[
                    'no_effect_lvo_mt_deaths']
            t0_logodds = \
                self.mrs_distribution_logodds['t0_treatment_lvo_mt']
        except KeyError:
            raise KeyError(
                'Need to create LVO mRS distributions first.')

        # Create an x by 7 grid of mRS distributions,
        # one row of 7 mRS values for each of x patients.
        mask_valid = (self.trial['stroke_type_code'].data == 2)
        post_stroke_probs = _calculate_probs_at_treatment_time(
            t0_logodds,
            no_effect_logodds,
            self.trial['onset_to_puncture_mins'].data,
            self.mt_time_no_effect_mins,
            self.trial['mt_chosen_bool'].data,
            self.trial['mt_no_effect_bool'].data,
            mask_valid,
            not_treated_probs,
            no_effect_probs
            )
        return post_stroke_probs

    def calculate_post_stroke_mrs_dists_for_nlvo_ivt(self):
        """
        Wrapper for _calculate_probs_at_treatment_time() for nLVO+IVT.

        Calculate post-stroke mRS dists for nLVO treated with IVT.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_nlvo']
            no_effect_probs = \
                self.mrs_distribution_probs['no_effect_nlvo_ivt_deaths']
            no_effect_logodds = \
                self.mrs_distribution_logodds[
                    'no_effect_nlvo_ivt_deaths']
            t0_logodds = \
                self.mrs_distribution_logodds['t0_treatment_nlvo_ivt']
        except KeyError:
            raise KeyError(
                'Need to create nLVO mRS distributions first.')

        # Create an x by 7 grid of mRS distributions,
        # one row of 7 mRS values for each of x patients.
        mask_valid = (self.trial['stroke_type_code'].data == 1)
        post_stroke_probs = _calculate_probs_at_treatment_time(
            t0_logodds,
            no_effect_logodds,
            self.trial['onset_to_needle_mins'].data,
            self.ivt_time_no_effect_mins,
            self.trial['ivt_chosen_bool'].data,
            self.trial['ivt_no_effect_bool'].data,
            mask_valid,
            not_treated_probs,
            no_effect_probs
            )
        return post_stroke_probs

    def calculate_outcomes_dict_for_lvo_ivt(self, post_stroke_probs):
        """
        Wrapper for _calculate_outcomes_dict() for LVO with IVT.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_lvo']
            no_effect_probs = \
                self.mrs_distribution_probs['no_effect_lvo_ivt_deaths']
        except KeyError:
            raise KeyError(
                'Need to create LVO mRS distributions first.')

        treatment_chosen_bool = self.trial['ivt_chosen_bool'].data == 2

        outcomes_dict = self.calculate_outcomes_dict(
            post_stroke_probs,
            not_treated_probs,
            no_effect_probs,
            treatment_chosen_bool
            )
        return outcomes_dict

    def calculate_outcomes_dict_for_lvo_mt(self, post_stroke_probs):
        """
        Wrapper for _calculate_outcomes_dict() for LVO with MT.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_lvo']
            no_effect_probs = \
                self.mrs_distribution_probs['no_effect_lvo_mt_deaths']
        except KeyError:
            raise KeyError(
                'Need to create LVO mRS distributions first.')

        treatment_chosen_bool = self.trial['mt_chosen_bool'].data == 2

        outcomes_dict = self.calculate_outcomes_dict(
            post_stroke_probs,
            not_treated_probs,
            no_effect_probs,
            treatment_chosen_bool
            )
        return outcomes_dict

    def calculate_outcomes_dict_for_nlvo_ivt(self, post_stroke_probs):
        """
        Wrapper for _calculate_outcomes_dict() for nLVO with IVT.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_nlvo']
            no_effect_probs = \
                self.mrs_distribution_probs['no_effect_nlvo_ivt_deaths']
        except KeyError:
            raise KeyError(
                'Need to create nLVO mRS distributions first.')

        treatment_chosen_bool = self.trial['ivt_chosen_bool'].data == 1

        outcomes_dict = self.calculate_outcomes_dict(
            post_stroke_probs,
            not_treated_probs,
            no_effect_probs,
            treatment_chosen_bool
            )
        return outcomes_dict

    def calculate_outcomes_dict(
            self,
            post_stroke_probs,
            not_treated_probs,
            no_effect_probs,
            treatment_chosen_bool
            ):
        """
        Calculate continuous outcomes dictionaries.

        This runs the _create_mrs_utility_dict() function twice,
        once with all valid patients and once with only the treated
        valid patients, and then creates one combined dictionary
        with the important bits.

        Inputs:
        -------
        post_stroke_probs     - x by 7 array. One post-stroke mRS
                                distribution per patient.
        not_treated_probs     - array of 7 floats. mRS probability
                                distribution for patients not receiving this
                                treatment.
        no_effect_probs       - array of 7 floats. mRS probability
                                distribution for patients who receive this
                                treatment after the time of no effect.
        treatment_chosen_bool - array of x bools. Whether each patient
                                received this treatment.

        Outputs:
        A dictionary containing the following:
        - each_patient_post_stroke_mrs_dist
        - not_treated_mean_mrs
        - no_effect_mean_mrs
        - each_patient_post_stroke_mean_mrs
        - each_patient_mean_mrs_shift
        - not_treated_mean_utility
        - no_effect_mean_utility
        - each_patient_post_stroke_mean_utility
        - each_patient_mean_added_utility
        - proportion_of_whole_cohort_improved
        - mean_valid_patients_mean_mrs_shift
        - mean_valid_patients_mean_added_utility
        - treated_population_mean_valid_patients_mean_mrs_shift
        - treated_population_mean_valid_patients_mean_added_utility
        """
        # Find mean mRS and utility values in these results dictionary.
        # The results for all patients...
        results_dict = self._create_mrs_utility_dict(
            post_stroke_probs,
            not_treated_probs,
            no_effect_probs
            )
        # ... and for only the patients who were treated:
        results_treated_dict = self._create_mrs_utility_dict(
            post_stroke_probs[treatment_chosen_bool, :],
            not_treated_probs,
            no_effect_probs
            )
        # Merge the two dictionaries:
        keys_to_merge = [
            'mean_valid_patients_mean_mrs_shift',
            'mean_valid_patients_mean_added_utility'
            ]
        for key in keys_to_merge:
            results_dict['treated_population_' + key] = (
                results_treated_dict[key])

        return results_dict

    """
    ##############################
    ##### CALCULATE OUTCOMES #####
    ##############################

    The following functions do most of the legwork in actually
    calculating the outcomes.

    If a function is specific to this outcome measurement method,
    it will appear below. If it is shared between multiple outcome
    measurement methods, it will probably be in the outcome_utilities
    module.
    """
    def _create_mrs_utility_dict(
            self,
            post_stroke_probs,
            not_treated_probs,
            no_effect_probs
            ):
        """
        Create a dictionary of useful mRS dist and utility values.

        Inputs:
        -------
        post_stroke_probs - x by 7 ndarray.
                            Previously-calculated mRS dists for all
                            patients in the array post-stroke.
                            The mRS dist of the nth patient is
                            post_stroke_probs[n, :].
        not_treated_probs - 1 by 7 array. mRS dist for
                            the patients who receive no treatment.
        no_effect_probs   - 1 by 7 array. mRS dist for
                            the patients who are treated too late
                            for any positive effect.

        Returns:
        --------
        results - dict. Contains various mRS and utility values.
        """
        # Convert cumulative mRS to non-cumulative:
        not_treated_noncum_dist = np.diff(np.append(0.0, not_treated_probs))
        no_effect_noncum_dist = np.diff(np.append(0.0, no_effect_probs))
        post_stroke_noncum_dist = np.diff(np.concatenate(
            (np.zeros((post_stroke_probs.shape[0], 1)), post_stroke_probs),
            axis=1))

        # Convert mRS distributions to utility-weighted mRS:
        not_treated_util = not_treated_noncum_dist * self.utility_weights
        no_effect_util = no_effect_noncum_dist * self.utility_weights
        post_stroke_util = post_stroke_noncum_dist * self.utility_weights

        # Put results in dictionary
        results = dict()

        # mRS distributions:
        results['each_patient_post_stroke_mrs_dist'] = (
            post_stroke_probs)                                   # x by 7 grid
        # mean values:
        results['not_treated_mean_mrs'] = np.sum(
            not_treated_noncum_dist * np.arange(7))                  # 1 float
        results['no_effect_mean_mrs'] = np.sum(
            no_effect_noncum_dist * np.arange(7))                    # 1 float
        results['each_patient_post_stroke_mean_mrs'] = np.sum(
            post_stroke_noncum_dist * np.arange(7), axis=1)         # x floats
        # Change from not-treated distribution:
        results['each_patient_mean_mrs_shift'] = (
            np.sum(post_stroke_noncum_dist * np.arange(7), axis=1)
            - results['not_treated_mean_mrs']
            )                                                       # x floats

        # Utility-weighted mRS distributions:
        # mean values:
        results['not_treated_mean_utility'] = np.sum(
            not_treated_util)                                        # 1 float
        results['no_effect_mean_utility'] = np.sum(no_effect_util)   # 1 float
        results['each_patient_post_stroke_mean_utility'] = np.sum(
            post_stroke_util, axis=1)                               # x floats
        # Change from not-treated distribution:
        results['each_patient_mean_added_utility'] = (
            np.sum(post_stroke_util, axis=1) - np.sum(not_treated_util)
            )                                                       # x floats

        # Calculate the overall changes.
        # Use nanmean here because invalid patient data is set to NaN,
        # e.g. patients who have nLVO when we're calculating
        # results for patients with LVOs.
        results['mean_valid_patients_mean_mrs_shift'] = (
            np.nanmean(results['each_patient_mean_mrs_shift'])       # 1 float
            if len(np.where(np.isnan(
                results['each_patient_mean_mrs_shift']) == False)[0]) > 0
            else np.NaN
            )
        results['mean_valid_patients_mean_added_utility'] = (
            np.nanmean(results['each_patient_mean_added_utility'])   # 1 float
            if len(np.where(np.isnan(
                results['each_patient_mean_added_utility']) == False)[0]) > 0
            else np.NaN
            )

        return results

    def _calculate_patient_outcomes(self, dict_results_by_category):
        """
        Find the outcomes for the full cohort from existing results.

        Takes the results from the separate results dictionaries and
        pulls out the relevant parts for each patient category
        (nLVO+IVT, LVO+IVT, LVO+MT).
        The output arrays contain x values, one for each patient.

        Run _merge_results_dicts() first with all three categories
        as input (nLVO+IVT, LVO+IVT, LVO+MT).

        Contents of returned dictionary:
        - each_patient_post_stroke_mrs_dist                 x by 7 grid
        - each_patient_post_stroke_mean_mrs                    x floats
        - each_patient_mean_mrs_shift                          x floats
        - each_patient_post_stroke_mean_utility                x floats
        - each_patient_mean_added_utility                      x floats
        - post_stroke_mean_mrs                                  1 float
        - mean_mrs_shift                                        1 float
        - mean_utility                                          1 float
        - mean_added_utility                                    1 float
        - proportion_improved                                   1 float

        Inputs:
        -------
        dict_results_by_category - dict. Contains outcome data for
                                   nLVO+IVT, LVO+IVT, LVO+MT groups
                                   where each group has x entries.

        Returns:
        --------
        full_cohort_outcomes - dict. Outcome data for the patient
                                 array, containing x entries.
        """
        # Find which indices belong to each category of stroke type
        # and treatment combination.
        # For LVO patients, the pre-stroke and no-treatment
        # distributions are the same for IVT and MT. So use the
        # LVO+IVT results for all LVO patients except those
        # who received MT.
        inds_lvo_not_mt = (
            (self.trial['stroke_type_code'].data == 2) &
            (self.trial['mt_chosen_bool'].data < 1)
            )
        inds_lvo_mt_only = (
            (self.trial['stroke_type_code'].data == 2) &
            (self.trial['ivt_chosen_bool'].data < 1) &
            (self.trial['mt_chosen_bool'].data > 0)
            )
        inds_nlvo_ivt = (
            (self.trial['stroke_type_code'].data == 1)
            )

        # When patients receive both IVT and MT, pick out which had the
        # bigger effect.
        # np.where() returns False where one of the values is NaN,
        # so this comparison only returns True where both values are valid.
        inds_lvo_ivt_better_than_mt = np.where(
            dict_results_by_category['lvo_ivt_each_patient_mean_mrs_shift'] <
            dict_results_by_category['lvo_mt_each_patient_mean_mrs_shift']
            )[0]
        inds_lvo_mt_better_than_ivt = np.where(
            dict_results_by_category['lvo_mt_each_patient_mean_mrs_shift'] <=
            dict_results_by_category['lvo_ivt_each_patient_mean_mrs_shift']
            )[0]

        inds = [inds_lvo_not_mt, inds_lvo_mt_only, inds_nlvo_ivt,
                inds_lvo_ivt_better_than_mt, inds_lvo_mt_better_than_ivt]

        # The categories have these labels in the combo dictionary:
        labels = ['lvo_ivt', 'lvo_mt', 'nlvo_ivt', 'lvo_ivt', 'lvo_mt']

        # Define new empty arrays that will be filled with results
        # from the existing results dictionaries.
        each_patient_post_stroke_mrs_dist = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_post_stroke_mrs_dist'].shape,
            np.NaN
            )
        each_patient_post_stroke_mean_mrs = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_post_stroke_mean_mrs'].shape,
            np.NaN
            )
        each_patient_mean_mrs_shift = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_mean_mrs_shift'].shape,
            np.NaN
            )
        each_patient_post_stroke_mean_utility = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_post_stroke_mean_utility'].shape,
            np.NaN
            )
        each_patient_mean_added_utility = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_mean_added_utility'].shape,
            np.NaN
            )

        for i, label in enumerate(labels):
            inds_here = inds[i]

            # mRS distributions:
            each_patient_post_stroke_mrs_dist[inds_here, :] = (
                dict_results_by_category
                [label + '_each_patient_post_stroke_mrs_dist']
                [inds_here, :]
                )                                                # x by 7 grid
            each_patient_post_stroke_mean_mrs[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_post_stroke_mean_mrs']
                [inds_here]
                )                                                   # x floats
            # Change from not-treated distribution:
            each_patient_mean_mrs_shift[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_mean_mrs_shift']
                [inds_here]
                )                                                   # x floats

            # Utility-weighted mRS distributions:
            # mean values:
            each_patient_post_stroke_mean_utility[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_post_stroke_mean_utility']
                [inds_here]
                )                                                   # x floats
            # Change from not-treated distribution:
            each_patient_mean_added_utility[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_mean_added_utility']
                [inds_here]
                )                                                   # x floats

        # Average these results over all patients:
        post_stroke_mean_mrs = \
            np.nanmean(each_patient_post_stroke_mean_mrs)            # 1 float
        mean_mrs_shift = \
            np.nanmean(each_patient_mean_mrs_shift)                  # 1 float
        mean_utility = \
            np.nanmean(each_patient_post_stroke_mean_utility)        # 1 float
        mean_added_utility = \
            np.nanmean(each_patient_mean_added_utility)              # 1 float

        # Create dictionary for combined full cohort outcomes:
        full_cohort_outcomes = dict(
            each_patient_post_stroke_mrs_dist=(
                each_patient_post_stroke_mrs_dist),              # x by 7 grid
            each_patient_post_stroke_mean_mrs=(
                each_patient_post_stroke_mean_mrs),                 # x floats
            each_patient_mean_mrs_shift=(
                each_patient_mean_mrs_shift),                       # x floats
            each_patient_post_stroke_mean_utility=(
                each_patient_post_stroke_mean_utility),             # x floats
            each_patient_mean_added_utility=(
                each_patient_mean_added_utility),                   # x floats
            post_stroke_mean_mrs=post_stroke_mean_mrs,               # 1 float
            mean_mrs_shift=mean_mrs_shift,                           # 1 float
            mean_utility=mean_utility,                               # 1 float
            mean_added_utility=mean_added_utility,                   # 1 float
            )

        # Save to instance:
        self.full_cohort_outcomes = full_cohort_outcomes

        return full_cohort_outcomes
