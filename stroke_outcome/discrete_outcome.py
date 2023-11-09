import numpy as np
import pandas as pd
import numpy.typing as npt  # For type hinting.
import warnings

from .evaluated_array import Evaluated_array
import stroke_outcome.outcome_utilities as ou


class Discrete_outcome:
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
    - each_patient_mrs_dist_post_stroke
    - each_patient_mrs_not_treated
    - each_patient_mrs_post_stroke
    - each_patient_utility_not_treated
    - each_patient_utility_post_stroke
    - each_patient_mrs_shift
    - each_patient_utility_shift
    - valid_patients_mean_mrs_post_stroke
    - valid_patients_mean_mrs_not_treated
    - valid_patients_mean_mrs_shift
    - valid_patients_mean_utility_post_stroke
    - valid_patients_mean_utility_not_treated
    - valid_patients_mean_utility_shift
    - treated_patients_mean_mrs_post_stroke
    - treated_patients_mean_mrs_shift
    - treated_patients_mean_utility_post_stroke
    - treated_patients_mean_utility_shift
    - improved_patients_mean_mrs_post_stroke
    - improved_patients_mean_mrs_shift
    - improved_patients_mean_utility_post_stroke
    - improved_patients_mean_utility_shift
    - proportion_of_valid_patients_who_improved
    - proportion_of_treated_patients_who_improved

    The full_cohort_outcomes results dictionary
    takes the results from the separate results dictionaries and
    pulls out the relevant parts for each patient category
    (nLVO+IVT, LVO+IVT, LVO+MT).
    The output arrays contain x values, one for each patient.
    Contents of returned dictionary:
    - each_patient_mrs_dist_post_stroke                 x by 7 grid
    - each_patient_mrs_post_stroke                         x floats
    - each_patient_mrs_not_treated                         x floats
    - each_patient_mrs_shift                               x floats
    - each_patient_utility_post_stroke                     x floats
    - each_patient_utility_not_treated                     x floats
    - each_patient_utility_shift                           x floats
    - mean_mrs_post_stroke                                  1 float
    - mean_mrs_shift                                        1 float
    - mean_utility                                          1 float
    - mean_utility_shift                                    1 float
    - proportion_improved                                   1 float


    Utility-weighted mRS
    --------------------

    In addition to mRS we may calculate utility-weighted mRS. Utility is an
    estimated quality of life (0=dead, 1=full quality of life, negative numbers
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
        discrete_outcome = Discrete_outcome(mrs_dists={pandas dataframe})
        # Import patient data:
        discrete_outcome.trial['onset_to_needle_mins'].data = {array 1}
        discrete_outcome.trial['ivt_chosen_bool'].data = {array 2}
        discrete_outcome.trial['stroke_type_code'].data = {array 3}
        discrete_outcome.trial['mrs_pre_stroke'].data = {array 3}
        # Calculate outcomes:
        results_by_stroke_type, full_cohort_outcomes = (
            discrete_outcome.calculate_outcomes())


    Limitations and notes:
    ----------------------
    When fixed probabilities are not provided, they are generated
    using np.random.uniform. This random element means that running the
    same data twice will produce different results.

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
            mrs_dists: pd.DataFrame = pd.DataFrame(),
            utility_weights: npt.ArrayLike = np.array([]),
            ivt_time_no_effect_mins: float = 378.0,
            mt_time_no_effect_mins: float = 480.0
            ):
        """
        Constructor for discrete clinical outcome model.

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
        self.name = "Discrete clinical outcome model"

        # Use either the user input "mrs_dists" or load them from
        # file if none were provided. Check that it all makes sense:
        mrs_dists = ou.sanity_check_input_mrs_dists(mrs_dists)
        # Store the input for the repr() string.
        self.mrs_dists_input = mrs_dists

        #
        # ##### Set up model parameters #####
        # Store modified Rankin Scale distributions as arrays in dictionary
        self.mrs_distribution_probs, self.mrs_distribution_logodds = \
            ou.extract_mrs_probs_and_logodds(mrs_dists)

        # Set general model parameters
        self.ivt_time_no_effect_mins = ivt_time_no_effect_mins
        self.mt_time_no_effect_mins = mt_time_no_effect_mins

        # Store utility weightings for mRS 0-6
        self.utility_weights = ou.sanity_check_utility_weights(utility_weights)

    def __str__(self):
        """Prints info when print(Instance) is called."""
        print_str = 'The base mRS distributions are: '
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
            '... using the syntax: \n',
            '    trial[{key}].data = {array}\n',
            'In addition, either of the following arrays should be ',
            'provided. If both are provided, the fixed probability array ',
            'is used.\n',
            '- mrs_pre_stroke\n',
            '- fixed_prob_pre_stroke\n',
            ])

        print_str += ''.join([
            '\n',
            'The easiest way to create the results dictionaries is:\n',
            '  results_by_stroke_type, full_cohort_outcomes = ',
            'discrete_outcome.calculate_outcomes()'
            ])
        return print_str

    def __repr__(self):
        """Prints how to reproduce this instance of the class."""
        # This string prints without actual newlines, just the "\n"
        # characters, but it's the best way I can think of to display
        # the input dataframe in full.
        return ''.join([
            'Discrete_outcome(',
            f'mrs_dists=DATAFRAME*, '
            f'utility_weights={self.utility_weights}',
            f'ivt_time_no_effect_mins={self.ivt_time_no_effect_mins}',
            f'mt_time_no_effect_mins={self.mt_time_no_effect_mins})',
            '        \n\n        ',
            'The dataframe DATAFRAME* is created with: \n',
            f'  index: {self.mrs_dists_input.index}, \n',
            f'  columns: {self.mrs_dists_input.columns}, \n',
            f'  values: {repr(self.mrs_dists_input.values)}'
            ])

    """
    ########################
    ##### DATA STORAGE #####
    ########################

    These functions set up where the data will be stored.
    """
    def _create_new_blank_trial_dict(self, n: int):
        """
        Create a new trial dictionary for this model type.

        By default, the dictionary contents are arrays of length "n"
        and all placeholder values of zero in the required data type.
        The dictionary contents are defined here but the method
        to update them is assign_patient_data() in outcome_utilities.

        The syntax `trial[key].data` makes the patient data pass
        through a series of sanity checks in the class. The checks make
        sure that the data is of the expected type and sits in the
        expected range.

        Inputs:
        -------
        n - int. number of patients.

        Returns:
        --------
        trial - dict. The dictionary with placeholder data.
        """
        # Each placeholder array has the format:
        # Evaluated_array(number_of_patients, valid_dtypes_list,
        #                 valid_min, valid_max, name)
        trial = dict(
            stroke_type_code=Evaluated_array(
                n, ['int'], 0, 2, 'stroke type code'),
            stroke_type_code_on_input=Evaluated_array(
                n, ['int'], 0, 2, 'stroke type code on input'),
            onset_to_needle_mins=Evaluated_array(
                n, ['float'], 0.0, np.inf, 'onset to needle (mins)'),
            ivt_chosen_bool=Evaluated_array(
                n, ['int', 'bool'], 0, 1, 'ivt chosen'),
            ivt_no_effect_bool=Evaluated_array(
                n, ['int', 'bool'], 0, 1, 'ivt no effect'),
            onset_to_puncture_mins=Evaluated_array(
                n, ['float'], 0.0, np.inf, 'onset to puncture (mins)'),
            mt_chosen_bool=Evaluated_array(
                n, ['int', 'bool'], 0, 1, 'mt chosen'),
            mt_no_effect_bool=Evaluated_array(
                n, ['int', 'bool'], 0, 1, 'mt no effect'),
            mrs_pre_stroke=Evaluated_array(
                n, ['int', 'float'], 0, 6, 'mrs pre-stroke'),
            fixed_prob_pre_stroke=Evaluated_array(
                n, ['float'], 0.0, 1.0, 'x pre-stroke'),
        )

        # Immediately overwrite the values of fixed probability pre-stroke
        # so that later we can check whether fixed probability has been updated
        # by the user. Set all values to something unlikely to happen
        # by chance or by manual user input:
        trial['fixed_prob_pre_stroke'].data = np.full(n, np.e / np.pi)
        return trial

    def _check_trial_dict_for_new_data(self, data_df: pd.DataFrame):
        """
        Updates self.trial dictionary with patient data.

        If there is an existing self.trial dictionary and each entry
        contains data for the same number of patients as in the input
        data_df, then the existing dictionary is updated. Otherwise
        a new trial dictionary is created.

        Inputs:
        -------
        data_df - pd.DataFrame. Dataframe containing the patient data.
                  This should have some columns with names matching the
                  keys of self.trial dictionary. To see those keys,
                  look at _create_new_blank_trial_dict().
        """

        # Number of patients in this new data:
        n = len(data_df)

        try:
            trial = self.trial
            make_new_trial_dict = False

            # Sanity check:
            # Number of patients in existing trial dictionary:
            nt = len(list(trial.values())[0].data)
            # If the numbers of patients do not match...
            if n != nt:
                # ... create a new trial dictionary for the new
                # number of patients.
                make_new_trial_dict = True
                # Print a warning in case this was accidental.
                warning_message = ''.join([
                    'Previous patient data has been deleted because this ',
                    'new data has a different number of patients.'
                    ])
                warnings.warn(warning_message)
        except AttributeError:
            # self.trial does not exist, so make a new trial dict:
            make_new_trial_dict = True

        if make_new_trial_dict:
            trial = self._create_new_blank_trial_dict(n)
            # Save to self:
            self.trial = trial

    def assign_patients_to_trial(self, data_df: pd.DataFrame):
        """
        Prepare input data for use in the model.

        Inputs:
        -------
        data_df - pd.DataFrame. Contains the patient data. Any column
                  with a name that matches a key in the trial dict
                  will be used to update the trial dictionary.
        """
        # Check if an existing self.trial object is compatible with the
        # input data, and if not then create a new self.trial dict.
        self._check_trial_dict_for_new_data(data_df)

        # Copy over applicable parts of the patient data to update
        # the trial dictionary:
        trial = ou.assign_patient_data(data_df, self.trial)

        # Store the updated trial dictionary in self:
        self.trial = trial

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

        These are converted into cumulative probabilties,
        mean mRS, and mRS shift.

        Returns:
        --------
        A results dictionary with entries for each of these three
        categories:
        - nLVO treated with IVT
        - LVO treated with IVT
        - LVO treated with MT

        The full_cohort_outcomes results dictionary
        takes the results from the separate results dictionaries and
        pulls out the relevant parts for each patient category
        (nLVO+IVT, LVO+IVT, LVO+MT).
        The output arrays contain x values, one for each patient.
        Contents of returned dictionary:
        - each_patient_mrs_dist_post_stroke                 x by 7 grid
        - each_patient_mrs_post_stroke                         x floats
        - each_patient_mrs_not_treated                         x floats
        - each_patient_mrs_shift                               x floats
        - each_patient_utility_post_stroke                     x floats
        - each_patient_utility_not_treated                     x floats
        - each_patient_utility_shift                           x floats
        - mean_mrs_post_stroke                                  1 float
        - mean_mrs_shift                                        1 float
        - mean_utility                                          1 float
        - mean_utility_shift                                    1 float
        - proportion_improved                                   1 float
        """
        # ##### Sanity checks #####
        # Check if anyone has an nLVO and receives MT
        # (for which we don't have mRS probability distributions)
        self.trial['stroke_type_code'].data = (
            ou.assign_nlvo_with_mt_as_lvo(
                self.trial['stroke_type_code'].data,
                self.trial['mt_chosen_bool'].data))

        # Sanity check the mRS distributions based on the stroke types:
        # if a stroke type exists in the stroke_type_code data,
        # check that the matching mRS distributions have been given.
        # Also check by treatment type.
        ou.sanity_check_mrs_dists_for_stroke_type(
                self.mrs_distribution_probs,
                self.trial['stroke_type_code'].data,
                self.trial['ivt_chosen_bool'].data,
                self.trial['mt_chosen_bool'].data
                )

        # Determine who receives treatment too late for any effect:
        self.trial['ivt_no_effect_bool'].data = (
            ou.assign_treatment_no_effect(
                self.trial['ivt_chosen_bool'].data,
                self.trial['onset_to_needle_mins'].data,
                self.ivt_time_no_effect_mins
                ))
        self.trial['mt_no_effect_bool'].data = (
            ou.assign_treatment_no_effect(
                self.trial['mt_chosen_bool'].data,
                self.trial['onset_to_puncture_mins'].data,
                self.mt_time_no_effect_mins
                ))

        # Pre-stroke mRS score
        # Check whether the fixed probabilities have been given by the user.
        # If so, calculate the pre-stroke mRS scores.
        # If not, generate some fixed probabilities from the given pre-stroke
        # mRS scores.
        # n.b. on init, all fixed_prob_pre_stroke was set to this fixed value.
        if ~np.all(self.trial['fixed_prob_pre_stroke'].data == (np.e / np.pi)):
            # Fixed probability has been given,
            # so calculate pre-stroke mRS:
            self.trial['mrs_pre_stroke'].data = (
                self.generate_cohort_mrs_scores_from_fixed_probs())
        else:
            # Fixed probability has not been given,
            # so generate it from pre-stroke mRS:
            self.trial['fixed_prob_pre_stroke'].data = (
                self.generate_cohort_fixed_prob_from_mrs_scores())

        # ##### Statistics #####
        # Function to create a dataframe of patient statistics
        # for each stroke type.
        self.stroke_type_stats_df = (
            ou.calculate_patient_population_stats(self.trial))

        # ##### Calculations #####
        # Get treatment results
        # These post-stroke mRS distributions are the same for any
        # measure of change of outcome...
        post_stroke_probs_lvo_ivt = (
            ou.calculate_post_stroke_mrs_dists_for_lvo_ivt(
                self.mrs_distribution_probs,
                self.mrs_distribution_logodds,
                self.trial,
                self.ivt_time_no_effect_mins))
        post_stroke_probs_lvo_mt = (
            ou.calculate_post_stroke_mrs_dists_for_lvo_mt(
                self.mrs_distribution_probs,
                self.mrs_distribution_logodds,
                self.trial,
                self.mt_time_no_effect_mins))
        post_stroke_probs_nlvo_ivt = (
            ou.calculate_post_stroke_mrs_dists_for_nlvo_ivt(
                self.mrs_distribution_probs,
                self.mrs_distribution_logodds,
                self.trial,
                self.ivt_time_no_effect_mins))
        # ... and these outcome dictionaries are specific to this
        # measure of change of outcome:
        lvo_ivt_outcomes = self.calculate_outcomes_dict_for_lvo_ivt(
            post_stroke_probs_lvo_ivt)
        lvo_mt_outcomes = self.calculate_outcomes_dict_for_lvo_mt(
            post_stroke_probs_lvo_mt)
        nlvo_ivt_outcomes = self.calculate_outcomes_dict_for_nlvo_ivt(
            post_stroke_probs_nlvo_ivt)

        # Gather results into one dictionary:
        outcomes_by_stroke_type_and_treatment = ou._merge_results_dicts(
            [lvo_ivt_outcomes, lvo_mt_outcomes, nlvo_ivt_outcomes],
            ['lvo_ivt', 'lvo_mt', 'nlvo_ivt'],
            )

        # Get the mean results for the full cohort:
        full_cohort_outcomes = self._calculate_patient_outcomes(
            outcomes_by_stroke_type_and_treatment)

        # ##### Data storage #####
        # Convert the patient data used by the model to a normal
        # dictionary and return this too.
        patient_data_dict = ou._convert_eval_dict_to_dict(self.trial)

        return (patient_data_dict,
                outcomes_by_stroke_type_and_treatment,
                full_cohort_outcomes
                )

    """
    ####################
    ##### WRAPPERS #####
    ####################

    This block of functions contains wrappers. The functions here
    gather variables and pass them to other functions to do the
    actual calculations.

    Scroll down if you're looking for the calculations!

    The wrappers are for _create_mrs_utility_dict().
    """
    def calculate_outcomes_dict_for_lvo_ivt(
            self, post_stroke_probs: npt.ArrayLike):
        """
        Wrapper for _create_mrs_utility_dict() for LVO with IVT.

        Inputs:
        -------
        post_stroke_probs - x by 7 ndarray.
                            Previously-calculated mRS dists for all
                            patients in the array post-stroke.
                            The mRS dist of the nth patient is
                            post_stroke_probs[n, :].

        Returns:
        --------
        outcomes_dict - dict. Various useful mRS and utility scores
                        for all patients in the input data.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_lvo']
        except KeyError:
            raise KeyError(
                'Need to create LVO mRS distributions first.')

        treatment_chosen_bool = self.trial['ivt_chosen_bool'].data == 1
        fixed_prob_each_patient = self.trial['fixed_prob_pre_stroke'].data

        outcomes_dict = self._create_mrs_utility_dict(
            post_stroke_probs,
            not_treated_probs,
            fixed_prob_each_patient,
            treatment_chosen_bool
            )
        return outcomes_dict

    def calculate_outcomes_dict_for_lvo_mt(
            self, post_stroke_probs: npt.ArrayLike):
        """
        Wrapper for _create_mrs_utility_dict() for LVO with MT.

        Inputs:
        -------
        post_stroke_probs - x by 7 ndarray.
                            Previously-calculated mRS dists for all
                            patients in the array post-stroke.
                            The mRS dist of the nth patient is
                            post_stroke_probs[n, :].

        Returns:
        --------
        outcomes_dict - dict. Various useful mRS and utility scores
                        for all patients in the input data.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_lvo']
        except KeyError:
            raise KeyError(
                'Need to create LVO mRS distributions first.')

        treatment_chosen_bool = self.trial['mt_chosen_bool'].data == 1
        fixed_prob_each_patient = self.trial['fixed_prob_pre_stroke'].data

        outcomes_dict = self._create_mrs_utility_dict(
            post_stroke_probs,
            not_treated_probs,
            fixed_prob_each_patient,
            treatment_chosen_bool
            )
        return outcomes_dict

    def calculate_outcomes_dict_for_nlvo_ivt(
            self, post_stroke_probs: npt.ArrayLike):
        """
        Wrapper for _create_mrs_utility_dict() for nLVO with IVT.

        Inputs:
        -------
        post_stroke_probs - x by 7 ndarray.
                            Previously-calculated mRS dists for all
                            patients in the array post-stroke.
                            The mRS dist of the nth patient is
                            post_stroke_probs[n, :].

        Returns:
        --------
        outcomes_dict - dict. Various useful mRS and utility scores
                        for all patients in the input data.
        """
        try:
            # Get relevant distributions
            not_treated_probs = \
                self.mrs_distribution_probs['no_treatment_nlvo']
        except KeyError:
            raise KeyError(
                'Need to create nLVO mRS distributions first.')

        treatment_chosen_bool = self.trial['ivt_chosen_bool'].data == 1
        fixed_prob_each_patient = self.trial['fixed_prob_pre_stroke'].data

        outcomes_dict = self._create_mrs_utility_dict(
            post_stroke_probs,
            not_treated_probs,
            fixed_prob_each_patient,
            treatment_chosen_bool
            )
        return outcomes_dict

    """
    ###############################################
    ##### LINK mRS SCORE TO FIXED PROBABILITY #####
    ###############################################

    The following functions link a patient's mRS score with the
    value of fixed probability used in the calculations.
    """

    def generate_cohort_fixed_prob_from_mrs_scores(self):
        """
        Generate fixed probabilities for a set of pre-stroke mRS scores.

        Treat the different stroke types separately as they have
        different pre-stroke mRS distributions. For each pre-stroke mRS,
        find the range of allowed values of fixed probability and
        generate a fixed probability score in this range for
        each relevant patient.
        """
        # Calculate these separately for nLVO, LVO, and "other"
        # patient subgroups.
        inds_nlvo = np.where(self.trial['stroke_type_code'].data == 1)[0]
        inds_lvo = np.where(self.trial['stroke_type_code'].data == 2)[0]
        # inds_other = np.where(self.trial['stroke_type_code'].data == 0)[0]

        inds = [inds_nlvo, inds_lvo]
        fixed_prob_full_cohort = np.full(
            self.trial['stroke_type_code'].data.shape,
            0.0
            )
        for s, stroke_type in enumerate(['nlvo', 'lvo']):
            # Which patients have this stroke type?
            inds_here = inds[s]
            mrs_scores_here = self.trial['mrs_pre_stroke'].data[inds_here]
            # Pull out the mRS distribution:
            mrs_dist = self.mrs_distribution_probs[
                f'pre_stroke_{stroke_type}']
            # Whack a zero at the front:
            mrs_dist = np.append(0.0, mrs_dist)
            # Generate fixed probability scores:
            fixed_prob_scores = self.assign_fixed_prob_by_mrs_score_uniform(
                mrs_scores_here, mrs_dist)
            # Place these scores in the full cohort array:
            fixed_prob_full_cohort[inds_here] = fixed_prob_scores
        return fixed_prob_full_cohort

    def assign_fixed_prob_by_mrs_score_uniform(
            self, mrs_scores: npt.ArrayLike, mrs_dist: npt.ArrayLike):
        """
        Convert mRS scores to fixed probabilities by random uniform sampling.

        Inputs:
        mrs_scores - np.array. One mRS score per patient.
        mrs_dist   - np.array. The mRS distribution to sample from.
                     The dist should contain cumulative probabilities
                     and begin with a leading 0.0 for mRS < 0.

        Returns:
        fixed_prob_scores   - np.array. One fixed probability per patient.
        """
        # Initially set all fixed probability to Not A Number.
        fixed_prob_scores = np.full(mrs_scores.size, np.NaN)
        for mRS in range(7):
            # Which patients have this mRS score?
            inds_here = np.where(mrs_scores == mRS)[0]
            # What are the fixed probabilities allowed to be?
            lower_bound = mrs_dist[mRS]
            upper_bound = mrs_dist[mRS + 1]
            # Randomly select some fixed probabilities in this range:
            fixed_prob_here = np.random.uniform(
                low=lower_bound, high=upper_bound, size=len(inds_here))
            # Store in the full group array:
            fixed_prob_scores[inds_here] = fixed_prob_here
        return fixed_prob_scores

    def generate_cohort_mrs_scores_from_fixed_probs(self):
        """
        Convert the fixed probabilities to pre-stroke mRS scores.

        This is used when fixed probabilities are not provided by the user.
        """
        # Calculate these separately for nLVO, LVO, and "other"
        # patient subgroups.
        inds_nlvo = np.where(self.trial['stroke_type_code'].data == 1)[0]
        inds_lvo = np.where(self.trial['stroke_type_code'].data == 2)[0]
        # inds_other = np.where(self.trial['stroke_type_code'].data == 0)[0]

        # Initially set everyone to mRS 0:
        mRS_scores_full_cohort = np.full(
            len(self.trial['stroke_type_code'].data), 0)
        # Convert each subgroup's fixed probabilities to mRS...
        mRS_nlvo = self.find_mrs_score_from_fixed_prob(
            self.trial['fixed_prob_pre_stroke'].data[inds_nlvo],
            self.mrs_distribution_probs['pre_stroke_nlvo']
            )
        mRS_lvo = self.find_mrs_score_from_fixed_prob(
            self.trial['fixed_prob_pre_stroke'].data[inds_lvo],
            self.mrs_distribution_probs['pre_stroke_lvo']
            )
        # ... and update these values in the full array:
        mRS_scores_full_cohort[inds_nlvo] = mRS_nlvo
        mRS_scores_full_cohort[inds_lvo] = mRS_lvo
        # Any patients who were not updated here will keep the
        # initial mRS value set above (0).
        return mRS_scores_full_cohort

    def find_mrs_score_from_fixed_prob(
            self,
            fixed_prob: npt.ArrayLike or float,
            mrs_dist: npt.ArrayLike or float
            ):
        """
        Convert fixed probability to mRS score using an mRS distribution.

        Inputs:
        -------
        fixed_prob - float or array. One fixed probability
                     per patient.
        mrs_dist   - array. mRS cumulative probability distribution to
                     sample from. It should begin with a leading zero.

        Returns:
        float or array. One mRS score per patient.
        """
        return np.digitize(fixed_prob, mrs_dist).astype(float)

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
            post_stroke_probs: npt.ArrayLike,
            not_treated_probs: npt.ArrayLike,
            fixed_prob_each_patient: npt.ArrayLike,
            mask_treated: npt.ArrayLike
            ):
        """
        Create a dictionary of useful mRS dist and utility values.

        Inputs:
        -------
        post_stroke_probs       - x by 7 ndarray.
                                  Previously-calculated mRS dists for all
                                  patients in the array post-stroke.
                                  The mRS dist of the nth patient is
                                  post_stroke_probs[n, :].
        not_treated_probs       - 1 by 7 array. mRS dist for
                                  the patients who receive no treatment.
        fixed_prob_each_patient - array of x floats. The fixed
                                  probabilities for all patients.
        mask_treated            - array of x bools. Whether each patient
                                  received this treatment.

        Returns:
        --------
        results - dict. Contains various mRS and utility values.
        The results dictionary contains the following:
        - each_patient_mrs_dist_post_stroke
        - each_patient_mrs_not_treated
        - each_patient_mrs_post_stroke
        - each_patient_utility_not_treated
        - each_patient_utility_post_stroke
        - each_patient_mrs_shift
        - each_patient_utility_shift
        - valid_patients_mean_mrs_post_stroke
        - valid_patients_mean_mrs_not_treated
        - valid_patients_mean_mrs_shift
        - valid_patients_mean_utility_post_stroke
        - valid_patients_mean_utility_not_treated
        - valid_patients_mean_utility_shift
        - treated_patients_mean_mrs_post_stroke
        - treated_patients_mean_mrs_shift
        - treated_patients_mean_utility_post_stroke
        - treated_patients_mean_utility_shift
        - improved_patients_mean_mrs_post_stroke
        - improved_patients_mean_mrs_shift
        - improved_patients_mean_utility_post_stroke
        - improved_patients_mean_utility_shift
        - proportion_of_valid_patients_who_improved
        - proportion_of_treated_patients_who_improved
        """
        # Pick out which patients are valid (i.e. no NaN):
        inds_valid = np.where(~np.isnan(post_stroke_probs[:, 0]))[0]
        inds_invalid = np.where(np.isnan(post_stroke_probs[:, 0]))[0]
        # Pick out which patients were treated:
        inds_treated = np.where((
            (~np.isnan(post_stroke_probs[:, 0])) &
            (mask_treated == 1)
            ))[0]

        # Put results in dictionary
        results = dict()

        # mRS distributions:
        results['each_patient_mrs_dist_post_stroke'] = (
            post_stroke_probs)                                   # x by 7 grid

        # For each patient:
        # For each of these mRS probability distributions:
        # + recorded post-stroke dist
        # + not treated
        # calculate the mRS bin that each patient would fall into.
        results['each_patient_mrs_not_treated'] = (                 # x floats
            self.find_mrs_score_from_fixed_prob(
                fixed_prob_each_patient, not_treated_probs))
        # Reset invalid patient values to NaN:
        results['each_patient_mrs_not_treated'][inds_invalid] = np.NaN
        # Loop over each patient to get a different set of post-stroke
        # mRS bins in each case.
        each_patient_mrs_post_stroke = []
        for i, fixed_prob in enumerate(fixed_prob_each_patient):
            mRS_dist = post_stroke_probs[i, :]
            if np.any(np.isnan(mRS_dist)):
                mRS_here = np.NaN
            else:
                mRS_here = self.find_mrs_score_from_fixed_prob(
                    fixed_prob, mRS_dist)
            each_patient_mrs_post_stroke.append(mRS_here)
        results['each_patient_mrs_post_stroke'] = (
            np.array(each_patient_mrs_post_stroke))                 # x floats

        # Convert to utility.
        # Leave the invalid patients as np.NaN and update the values
        # of the valid patients.
        # Use the mRS scores to index the utility array.
        each_patient_utility_not_treated = np.full(
            results['each_patient_mrs_not_treated'].shape, np.NaN)
        each_patient_utility_not_treated[inds_valid] = self.utility_weights[
            results['each_patient_mrs_not_treated'][inds_valid].astype(int)]
        results['each_patient_utility_not_treated'] = (
            each_patient_utility_not_treated)                       # x floats

        each_patient_utility_post_stroke = np.full(
            results['each_patient_mrs_post_stroke'].shape, np.NaN)
        each_patient_utility_post_stroke[inds_valid] = self.utility_weights[
            results['each_patient_mrs_post_stroke'][inds_valid].astype(int)]
        results['each_patient_utility_post_stroke'] = (
            each_patient_utility_post_stroke)                       # x floats

        # For each patient:
        # Calculate the shift in mRS and in utility from the
        # no-treatment distribution to the post-stroke distribution.
        results['each_patient_mrs_shift'] = (
            results['each_patient_mrs_post_stroke'] -
            results['each_patient_mrs_not_treated']
            )                                                       # x floats
        results['each_patient_utility_shift'] = (
            results['each_patient_utility_post_stroke'] -
            results['each_patient_utility_not_treated']
            )                                                       # x floats

        # Find the patients who improved on treatment.
        # "improved" means improvement in mRS score, i.e. negative shift.
        mask_improved = (
            (mask_treated == 1) &
            (results['each_patient_mrs_shift'] < 0.0)
            )
        inds_improved = np.where(mask_improved == 1)[0]

        # For each of these categories of patients:
        # + all patients
        # + all treated patients
        # + all treated patients who improved in mRS
        # calculate the mean mRS and utility, and mean shifts
        # from the no-treatment values.
        # Not expecting any NaN in these valid patients' data.

        # Valid patients:
        results['valid_patients_mean_mrs_post_stroke'] = (           # 1 float
            np.mean(results['each_patient_mrs_post_stroke'][inds_valid])
            )
        results['valid_patients_mean_mrs_not_treated'] = (           # 1 float
            np.mean(results['each_patient_mrs_not_treated'][inds_valid])
            )
        results['valid_patients_mean_mrs_shift'] = (                 # 1 float
            np.mean(results['each_patient_mrs_shift'][inds_valid])
            )
        results['valid_patients_mean_utility_post_stroke'] = (       # 1 float
            np.mean(results['each_patient_utility_post_stroke'][inds_valid])
            )
        results['valid_patients_mean_utility_not_treated'] = (       # 1 float
            np.mean(results['each_patient_utility_not_treated'][inds_valid])
            )
        results['valid_patients_mean_utility_shift'] = (             # 1 float
            np.mean(results['each_patient_utility_shift'][inds_valid])
            )

        # Treated patients:
        results['treated_patients_mean_mrs_post_stroke'] = (         # 1 float
            np.mean(results['each_patient_mrs_post_stroke'][inds_treated])
            )
        results['treated_patients_mean_mrs_shift'] = (               # 1 float
            np.mean(results['each_patient_mrs_shift'][inds_treated])
            )
        results['treated_patients_mean_utility_post_stroke'] = (     # 1 float
            np.mean(results['each_patient_utility_post_stroke'][inds_treated])
            )
        results['treated_patients_mean_utility_shift'] = (           # 1 float
            np.mean(results['each_patient_utility_shift'][inds_treated])
            )

        # Treated and improved patients:
        results['improved_patients_mean_mrs_post_stroke'] = (        # 1 float
            np.mean(results['each_patient_mrs_post_stroke'][inds_improved])
            )
        results['improved_patients_mean_mrs_shift'] = (              # 1 float
            np.mean(results['each_patient_mrs_shift'][inds_improved])
            )
        results['improved_patients_mean_utility_post_stroke'] = (    # 1 float
            np.mean(results['each_patient_utility_post_stroke'][inds_improved])
            )
        results['improved_patients_mean_utility_shift'] = (          # 1 float
            np.mean(results['each_patient_utility_shift'][inds_improved])
            )

        # Find proportion of improved patients:
        results['proportion_of_valid_patients_who_improved'] = (     # 1 float
            len(inds_improved) / len(inds_valid) if len(inds_valid) > 0
            else np.NaN
        )
        results['proportion_of_treated_patients_who_improved'] = (   # 1 float
            len(inds_improved) / len(inds_treated) if len(inds_treated) > 0
            else np.NaN
        )

        return results

    def _calculate_patient_outcomes(self, dict_results_by_category: dict):
        """
        Find the outcomes for the full cohort from existing results.

        Takes the results from the separate results dictionaries and
        pulls out the relevant parts for each patient category
        (nLVO+IVT, LVO+IVT, LVO+MT).
        The output arrays contain x values, one for each patient.

        Run _merge_results_dicts() first with all three categories
        as input (nLVO+IVT, LVO+IVT, LVO+MT).

        Contents of returned dictionary:
        - each_patient_mrs_dist_post_stroke                 x by 7 grid
        - each_patient_mrs_post_stroke                         x floats
        - each_patient_mrs_not_treated                         x floats
        - each_patient_mrs_shift                               x floats
        - each_patient_utility_post_stroke                     x floats
        - each_patient_utility_not_treated                     x floats
        - each_patient_utility_shift                           x floats
        - mean_mrs_post_stroke                                  1 float
        - mean_mrs_shift                                        1 float
        - mean_utility                                          1 float
        - mean_utility_shift                                    1 float
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
        # When both treatments give the same shift in mRS,
        # arbitrarily prioritise the MT data over IVT.
        inds_lvo_ivt_better_than_mt = np.where(
            (dict_results_by_category['lvo_ivt_each_patient_mrs_shift'] <
             dict_results_by_category['lvo_mt_each_patient_mrs_shift']) &
            (self.trial['stroke_type_code'].data == 2) &
            (self.trial['ivt_chosen_bool'].data > 0) &
            (self.trial['mt_chosen_bool'].data > 0)
            )[0]
        inds_lvo_mt_better_than_ivt = np.where(
            (dict_results_by_category['lvo_mt_each_patient_mrs_shift'] <=
             dict_results_by_category['lvo_ivt_each_patient_mrs_shift']) &
            (self.trial['stroke_type_code'].data == 2) &
            (self.trial['ivt_chosen_bool'].data > 0) &
            (self.trial['mt_chosen_bool'].data > 0)
            )[0]

        inds = [inds_lvo_not_mt, inds_lvo_mt_only, inds_nlvo_ivt,
                inds_lvo_ivt_better_than_mt, inds_lvo_mt_better_than_ivt]

        # The categories have these labels in the combo dictionary:
        labels = ['lvo_ivt', 'lvo_mt', 'nlvo_ivt', 'lvo_ivt', 'lvo_mt']

        # Define new empty arrays that will be filled with results
        # from the existing results dictionaries.
        each_patient_mrs_dist_post_stroke = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_mrs_dist_post_stroke'].shape,
            np.NaN
            )
        each_patient_mrs_post_stroke = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_mrs_post_stroke'].shape,
            np.NaN
            )
        each_patient_mrs_not_treated = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_mrs_not_treated'].shape,
            np.NaN
            )
        each_patient_mrs_shift = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_mrs_shift'].shape,
            np.NaN
            )
        each_patient_utility_post_stroke = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_utility_post_stroke'].shape,
            np.NaN
            )
        each_patient_utility_not_treated = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_utility_not_treated'].shape,
            np.NaN
            )
        each_patient_utility_shift = np.full(
            dict_results_by_category[
                labels[0] + '_each_patient_utility_shift'].shape,
            np.NaN
            )

        for i, label in enumerate(labels):
            inds_here = inds[i]

            # mRS distributions:
            each_patient_mrs_dist_post_stroke[inds_here, :] = (
                dict_results_by_category
                [label + '_each_patient_mrs_dist_post_stroke']
                [inds_here, :]
                )                                                # x by 7 grid
            each_patient_mrs_post_stroke[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_mrs_post_stroke']
                [inds_here]
                )                                                   # x floats
            each_patient_mrs_not_treated[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_mrs_not_treated']
                [inds_here]
                )                                                   # x floats
            # Change from not-treated distribution:
            each_patient_mrs_shift[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_mrs_shift']
                [inds_here]
                )                                                   # x floats

            # Utility-weighted mRS distributions:
            # mean values:
            each_patient_utility_post_stroke[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_utility_post_stroke']
                [inds_here]
                )                                                   # x floats
            each_patient_utility_not_treated[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_utility_not_treated']
                [inds_here]
                )                                                   # x floats
            # Change from not-treated distribution:
            each_patient_utility_shift[inds_here] = (
                dict_results_by_category
                [label + '_each_patient_utility_shift']
                [inds_here]
                )                                                   # x floats

        # Average these results over all patients:
        mean_mrs_post_stroke = \
            np.nanmean(each_patient_mrs_post_stroke)                 # 1 float
        mean_mrs_not_treated = \
            np.nanmean(each_patient_mrs_not_treated)                 # 1 float
        mean_mrs_shift = \
            np.nanmean(each_patient_mrs_shift)                       # 1 float
        mean_utility = \
            np.nanmean(each_patient_utility_post_stroke)             # 1 float
        mean_utility_not_treated = \
            np.nanmean(each_patient_utility_not_treated)             # 1 float
        mean_utility_shift = \
            np.nanmean(each_patient_utility_shift)                   # 1 float

        # Create dictionary for combined full cohort outcomes:
        full_cohort_outcomes = dict(
            each_patient_mrs_dist_post_stroke=(
                each_patient_mrs_dist_post_stroke),              # x by 7 grid
            each_patient_mrs_post_stroke=(
                each_patient_mrs_post_stroke),                      # x floats
            each_patient_mrs_not_treated=(
                each_patient_mrs_not_treated),                      # x floats
            each_patient_mrs_shift=(
                each_patient_mrs_shift),                            # x floats
            each_patient_utility_post_stroke=(
                each_patient_utility_post_stroke),                  # x floats
            each_patient_utility_not_treated=(
                each_patient_utility_not_treated),                  # x floats
            each_patient_utility_shift=(
                each_patient_utility_shift),                        # x floats
            mean_mrs_post_stroke=mean_mrs_post_stroke,               # 1 float
            mean_mrs_not_treated=mean_mrs_not_treated,               # 1 float
            mean_mrs_shift=mean_mrs_shift,                           # 1 float
            mean_utility=mean_utility,                               # 1 float
            mean_utility_not_treated=mean_utility_not_treated,       # 1 float
            mean_utility_shift=mean_utility_shift,                   # 1 float
            )

        # Save to instance:
        self.full_cohort_outcomes = full_cohort_outcomes

        return full_cohort_outcomes
