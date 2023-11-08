import numpy as np
import numpy.typing as npt  # For type hinting.


class Evaluated_array:
    """
    A class for sanity-checking data given some restrictions.

    The user can call this class with a given number of data entries,
    valid data types (e.g. int, string), and
    optional valid upper and lower bounds for the values.
    When setting the data array, the data is passed through a series of
    checks that the data is within the given restrictions.

    The resulting arrays are one-dimensional. If higher-dimensional
    data is passed in, it will be flattened.

    To use this class, call e.g.:
      eval_array = Evaluated_array(
          length=10,
          valid_min=0,
          valid_max=2,
          valid_dtypes=['int']
      )
    And then set data using e.g.:
      eval_array.data = np.full(10, 1, dtype=int)

    # ###########################################################
    TO DO - allow "int" classes to be input as float if conversion possible
    """
    def __init__(
            self,
            length: int,
            valid_dtypes: list,
            valid_min: float = np.NaN,
            valid_max: float = np.NaN,
            name_str: str = ''
            ):
        """
        Set up the evaluated array.

        Inputs:
        -------
        length       - int. Number of values for the data array.
        valid_dtypes - list of str. Allowed dtypes for the data
                       array. e.g. 'int' encompasses 'int64' and
                       similar more detailed type names.
        valid_min    - float or int. Minimum allowed value of
                       data in the array. If not applicable,
                       set valid_min to NaN.
        valid_max    - float or int. As valid_min for maximum.
        name_str     - str. Optional name for this data array.

        Initialises:
        ------------
        data - np.array. An array of zeroes in the first
               dtype in the list of valid_dtypes. This default array
               can be overwritten by the user and will then pass
               through the sanity check function.
        """
        self._length = length
        self._valid_dtypes = valid_dtypes

        # Optional min and max allowed values of the array:
        self._valid_min = valid_min
        self._valid_max = valid_max

        # Optional name for an instance of this class:
        self._name = name_str

        # Initially create a data array with dummy data.
        val = 0 if np.isnan(valid_min) else valid_min
        data = np.full(length, val, dtype=valid_dtypes[0])
        self.data = data

    def __str__(self):
        """
        Prints info when print(Instance) is called.
        """
        print_str = '\n'.join([
            'Evaluated array:',
            f'  _length = {self._length}',
            f'  _valid_dtypes = {self._valid_dtypes}',
            f'  _valid_min = {self._valid_min}',
            f'  _valid_max = {self._valid_max}',
            f'  _name = {self._name}',
            f'  data = {self.data}'
            ])
        return print_str

    def __repr__(self):
        """
        Prints how to reproduce this instance of the class.
        """
        return ''.join([
            'Evaluated_array(',
            f'length={self._length}, ',
            f'valid_dtypes={self._valid_dtypes}, ',
            f'valid_min={self._valid_min}, ',
            f'valid_max={self._valid_max}, ',
            f'name_str={self._name}',
            ')'
            ])

    def __setattr__(self, key: str, value):
        """
        Set attribute to the given value.

        Inputs:
        -------
        key   - str. Name of the attribute to be set.
        value - anything! Value to set the attribute to.
        """
        if key[0] != '_':
            # Expect this value to be a data array so run the sanity
            # checks. If they fail, an exception is raised and the
            # value will not be set.
            self.run_sanity_checks(value)
        self.__dict__[key] = value

    def __delattr__(self, key: str):
        """
        Set attribute to None (setup attrs) or default array (result).

        Inputs:
        -------
        key - str. Name of the attribute to be deleted or reset.
        """
        if key[0] != '_':
            # Return array to default values.
            # Select default value:
            val = 0 if np.isnan(self._valid_min) else self._valid_min
            self.__dict__[key] = np.full(
                self._length,
                val,
                dtype=np.dtype(self._valid_dtypes[0])
                )
        else:
            # Change setup value to None.
            self.__dict__[key] = None

    def run_sanity_checks(self, arr: npt.ArrayLike):
        """
        Check consistency of input data array with the setup values.

        Inputs:
        -------
        arr - array. The data array to be checked.
        """
        # Sanity checks flag. Change this to False if any checks fail:
        sanity_checks_passed = True
        # Don't raise exceptions as this function goes along to ensure
        # that all of the error messages are flagged up on the first
        # run through. Place error messages in here:
        failed_str = (
            f'{self._name}: Sanity checks failed. Values not updated.'
            )

        # Are all values the right dtype?
        if arr.dtype not in [
                np.dtype(valid_dtype) for valid_dtype in self._valid_dtypes
                ]:
            if ('int' in self._valid_dtypes) & (arr.dtype == float):
                # Check specifically whether floats have been given
                # instead of integers.
                if np.all(arr == arr.astype(int)):
                    # Quietly allow this case by converting float to
                    # integers.
                    arr = arr.astype(int)
                else:
                    sanity_checks_passed = False
            elif ('float' in self._valid_dtypes) & (arr.dtype == int):
                # Check whether integers have been given instead of floats.
                if np.all(arr == arr.astype(float)):
                    # Quietly allow this case by converting float to
                    # integers.
                    arr = arr.astype(float)
                else:
                    sanity_checks_passed = False
            else:
                sanity_checks_passed = False
        if sanity_checks_passed is False:
            failed_str += ''.join([
                # ' All values in the array must be the same dtype. ',
                f' Available data types are: {self._valid_dtypes}.'
                ])

        # Are all values within the allowed range?
        if self._valid_min is not None and self._valid_max is not None:
            if np.all(
                    (arr >= self._valid_min) & (arr <= self._valid_max)
                    ) is False:
                failed_str += ' Some values are outside the allowed range.'
                sanity_checks_passed = False

        # Is the array one-dimensional?
        if len(arr.shape) > 1:
            failed_str += ''.join([
                f' Flattening the input array from shape {arr.shape} to ',
                f'shape {arr.ravel().shape}.'
                ])
            arr = arr.ravel()
            # This doesn't fail the sanity check.

        # Does the array contain the right number of entries?
        if len(arr) != self._length:
            failed_str += ''.join([
                f' This array contains {len(arr)} values ',
                'but the expected length is ',
                f'{self._length}. ',
                'Please update the arrays to be the same length.'
                ])
            sanity_checks_passed = False

        # If any of the checks failed, raise exception now.
        if sanity_checks_passed is False:
            raise ValueError(failed_str) from None
            # ^ this syntax prevents longer message printing.
