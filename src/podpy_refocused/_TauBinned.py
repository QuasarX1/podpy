# SPDX-FileCopyrightText: 2016 Monica Turner <turnerm@mit.edu>
# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
"""
podpy is an implementatin of the pixel optical depth method as described in 
Turner et al. 2014, MNRAS, 445, 794, and Aguirre et al. 2002, ApJ, 576, 1. 
Please contact the author (Monica Turner) at turnerm@mit.edu if you have 
any questions, comment or issues. 
"""

import numpy as np
import numpy.random as rd
from QuasarCode import Console

from . import _universe as un

from typing import TYPE_CHECKING, Union, Tuple, Dict
if TYPE_CHECKING:
    from ._Spectrum import UserInput

class TauBinned:
    """
    Creates a spectrum object.

    Parameters
    ----------
    tau_x_object: instance of the Pod class
        Instance of the Pod class that you would like to use for the x axis (i.e., the 
        optical depths used for binning)
    tau_y_object: instance of the Pod class
        Instance of the Pod class that you would like to use for the y axis (i.e., compute
        the optical depth percentile in each bin of x)
    tau_x_min, tau_x_max: float, optional
        The minimum and maximum optical depths to use x (in log). Defaults are -5.75 and 4.0. 
    bin_size: float, optional
        The bin size to use when binning in x (in log). Default is 0.25. 
    chunk_size: float, optional
        The size of the chunks for bootstrap resampling, in Angstroms. Defaults is 50 A. 
    percentile_value: float, optional
        The percentile to compute in y. Default is 50.

    Instance attributes
    ----------
    tau_binned_x: The medians of the binned optical depths in x
    tau_binned_y: The percentile of the binned optical depths in y
    tau_binned_err: The error on the percentile in y     
    """


    MIN_BIN_PIXELS = 25
    MIN_CHUNKS_PER_BIN = 5
    MIN_PIX_ABOVE_PERCENTILE = 5
    FLAG = -10


    def __init__(self,
                tau_x_object,
                tau_y_object,
                tau_x_min = -5.75,
                tau_x_max = 4.0,
                bin_size = 0.25,
                chunk_size = 5.0, # Angstroms
                percentile_value = 50,
                bsrs = 1000,
                seed = 12345,
                use_legacy_code = False):
        self.seed = seed
        self.percentile_value = percentile_value
        self.chunk_size = chunk_size
        self.ion_x = tau_x_object.ion
        self.ion_y = tau_y_object.ion
        print("*** Binning optical depths ***")
        # Create the optical depth arrays
        lambda_x, tau_x, tau_y = TauBinned._find_pixel_pairs(tau_x_object, tau_y_object)
        self.pix_pairs_wavelength = lambda_x
        self.pix_pairs_x = tau_x
        self.pix_pairs_y = tau_y
        # Get the flat level, tau_min
        self._get_tau_min(tau_x, tau_y)
        print("Binning", self.ion_x, "vs.", self.ion_y)
        print("Number of good pixels:", len(lambda_x))
        if use_legacy_code:
            tau_chunks_x, tau_chunks_y = self._calc_percentiles_legacy(lambda_x, tau_x, tau_y,
                                                                       tau_x_min, tau_x_max, bin_size)
        else:
            tau_chunks_x, tau_chunks_y = self._calc_percentiles(lambda_x, tau_x, tau_y,
                                                                tau_x_min, tau_x_max, bin_size)
        if bsrs:
            if use_legacy_code:
                self._calc_errors_legacy(bsrs, tau_chunks_x, tau_chunks_y)
            else:
                self._calc_errors(bsrs, tau_chunks_x, tau_chunks_y)

    @staticmethod
    def _find_pixel_pairs(tau_x_object, tau_y_object):
        index_x, index_y = TauBinned._find_same_range(tau_x_object.z, tau_y_object.z)
        bad_pixel = ((tau_x_object.flag[index_x] % 2 == 1) | (tau_y_object.flag[index_y] % 2 == 1))
        index_x = index_x[~bad_pixel]
        index_y = index_y[~bad_pixel]
        tau_x, tau_y = tau_x_object.tau_rec[index_x], tau_y_object.tau_rec[index_y] 
        lambda_x = tau_x_object.lambdaa[index_x]
        return lambda_x, tau_x, tau_y

    @staticmethod
    def _find_same_range(array_1, array_2):
        # This is to be used for cases when oVI z range is less than that of
        # LyA and they need to be made the same. So the input array is either z
        # or lambda, and should be consecutive.
        error = 1E-10
        index_1 = np.where((array_1 >= array_2[0] - error) & (array_1 <= array_2[-1] + error))[0]
        index_2 = np.where((array_2 >= array_1[0] - error) & (array_2 <= array_1[-1] + error))[0]
        return index_1, index_2



    def _get_tau_min(self, tau_x, tau_y):
        self._get_tau_c()
        index_min = np.where(tau_x < self.tau_c) 
        self.tau_min = np.percentile(tau_y[index_min], self.percentile_value)

    def _get_tau_c(self):
        if self.ion_x in ["h1"]:
            self.tau_c = 0.1
        elif self.ion_x in ["c4", "si4"]:
            self.tau_c = 0.01

    def _calc_percentiles(self, lambda_x: np.ndarray, tau_x: np.ndarray, tau_y: np.ndarray, tau_x_min: float, tau_x_max: float, bin_size: float):
        # prepare chunks for bootstrapping
        edge_chunks = np.arange(lambda_x[0], lambda_x[-1] + self.chunk_size, self.chunk_size)
        num_chunks = len(edge_chunks) - 1
        # need to use list since elements will be different sizes
        chunk_pixels = [0] * (num_chunks)
        tau_chunks_x = [0] * (num_chunks)
        tau_chunks_y = [0] * (num_chunks)
        for i in range(num_chunks):
#            chunk_pixels[i] = np.where((lambda_x > edge_chunks[i]) & (lambda_x < edge_chunks[i+1]))[0]
            chunk_pixels[i] = np.where((lambda_x >= edge_chunks[i]) & (lambda_x < edge_chunks[i+1]))[0]
            tau_chunks_x[i] = tau_x[chunk_pixels[i]]
            tau_chunks_y[i] = tau_y[chunk_pixels[i]]
        # re-worked chunks to speedup bin crosschecking
        pixel_chunk_numbers = np.digitize(lambda_x, edge_chunks)
        out_of_bounds_chunk_numbers = np.array([0, num_chunks + 1], dtype = int)
        for chunk_index in range(num_chunks):
            # make sure this is producing the same results
            assert (np.where(pixel_chunk_numbers == chunk_index + 1)[0] == chunk_pixels[chunk_index]).all()# the original code used incomplete bin edge checks so has been modified to match
        # create the left and right bin edges
        self.edge_bins = np.arange(tau_x_min - bin_size / 2.0, tau_x_max + bin_size/2.0, bin_size)
        num_bins = len(self.edge_bins)-1
        # set up the empty arrays
        tau_binned_x, tau_binned_y = np.empty(num_bins), np.empty(num_bins)
        num_chunks_per_bin = np.zeros(num_bins)
        # bin the pixels
        pixel_bin_numbers = np.digitize(tau_x, self.edge_bins)
        # calculate the percentile
        print("Calculating", self.percentile_value, "th percentiles")
        for i in range(num_bins):
            Console.print_debug(f"Doing bin index {i}.")
            # Identify which pixels are in this bin
            bin_pixel_filter = pixel_bin_numbers == i + 1
            n_bin_px = bin_pixel_filter.sum()
            # Identify which unique chunks contribute to the bin
            num_chunks_per_bin[i] = len(np.unique(pixel_chunk_numbers[bin_pixel_filter][~np.isin(pixel_chunk_numbers[bin_pixel_filter], out_of_bounds_chunk_numbers)]))
            if n_bin_px > 0:
                tau_y_in_bin = tau_y[bin_pixel_filter] 
                bin_value = np.percentile(tau_y_in_bin, self.percentile_value)
                n_above_bin_value = (tau_y_in_bin > bin_value).sum()
            else:
                n_above_bin_value = 0
            if ((n_bin_px < TauBinned.MIN_BIN_PIXELS) or (num_chunks_per_bin[i] < TauBinned.MIN_CHUNKS_PER_BIN) or (n_above_bin_value < TauBinned.MIN_PIX_ABOVE_PERCENTILE)):
                tau_binned_x[i] = np.nan
                tau_binned_y[i] = np.nan
            else:
                # Calculate the percentiles
                tau_binned_x[i] = np.median(tau_x[bin_pixel_filter])
                tau_binned_y[i] = bin_value
        # get rid of emtpy bins
        self.nan = np.where(np.isnan(tau_binned_x))
        self.tau_binned_x = np.delete(tau_binned_x, self.nan)
        self.tau_binned_y = np.delete(tau_binned_y, self.nan)
        self.num_chunks_per_bin = np.delete(num_chunks_per_bin, self.nan)
        self.num_chunks = num_chunks
        return tau_chunks_x, tau_chunks_y

    def _calc_percentiles_legacy(self, lambda_x, tau_x, tau_y, tau_x_min, tau_x_max, bin_size):
        # prepare chunks for bootstrapping
        edge_chunks = np.arange(lambda_x[0], lambda_x[-1] + self.chunk_size, self.chunk_size)
        num_chunks = len(edge_chunks) - 1
        # need to use list since elements will be different sizes
        chunk_pixels = [0] * (num_chunks)
        tau_chunks_x = [0] * (num_chunks)
        tau_chunks_y = [0] * (num_chunks)
        for i in range(num_chunks):
            chunk_pixels[i] = np.where((lambda_x > edge_chunks[i]) & (lambda_x < edge_chunks[i+1]))[0]
            tau_chunks_x[i] = tau_x[chunk_pixels[i]]
            tau_chunks_y[i] = tau_y[chunk_pixels[i]]
        # create the left and right bin edges
        self.edge_bins = np.arange(tau_x_min - bin_size / 2.0, tau_x_max + bin_size/2.0, bin_size)
        num_bins = len(self.edge_bins)-1
        # set up the empty arrays
        tau_binned_x, tau_binned_y = np.empty(num_bins), np.empty(num_bins)
        num_chunks_per_bin = np.zeros(num_bins)
        # calculate the percentile
        print("Calculating", self.percentile_value, "th percentiles")
        for i in range(num_bins):
            Console.print_debug(f"Doing bin index {i}.")
            bin_pixels = np.where((tau_x > self.edge_bins[i]) & (tau_x < self.edge_bins[i+1]))[0]
            # Remove any bins that have less than 5 chunks
            # Also remove any bins that have less than 25 pixels
            tmp = [filter(lambda x: x in bin_pixels, sublist) for sublist in chunk_pixels]
            tmp = [len(list(sublist)) for sublist in tmp]
            tmp = np.array(tmp)
            num_chunks_per_bin[i] = len(np.where(tmp > 0)[0])
            if len(bin_pixels) > 0:
                ypix = tau_y[bin_pixels] 
                yval = np.percentile(ypix, self.percentile_value)
                ngt_yval = len(np.where(ypix > yval)[0])
            else:
                ngt_yval = 0
            if ((len(bin_pixels) < TauBinned.MIN_BIN_PIXELS) or
                (num_chunks_per_bin[i] < TauBinned.MIN_CHUNKS_PER_BIN) or
                (ngt_yval < TauBinned.MIN_PIX_ABOVE_PERCENTILE)):
                bin_pixels = np.empty(0, dtype=int)
            # Calculate the percentiles
            tau_binned_x[i] = np.median(tau_x[bin_pixels])
            if (len(bin_pixels) > 0):
                tau_binned_y[i] = yval
            else:
                tau_binned_y[i] = float('nan')
        # get rid of emtpy bins
        self.nan = np.where(np.isnan(tau_binned_x))
        self.tau_binned_x = np.delete(tau_binned_x, self.nan)
        self.tau_binned_y = np.delete(tau_binned_y, self.nan)
        self.num_chunks_per_bin = np.delete(num_chunks_per_bin, self.nan)
        self.num_chunks = num_chunks
        return tau_chunks_x, tau_chunks_y

    def _calc_errors(self, bsrs, tau_chunks_x, tau_chunks_y):
        # bootstrap resampling
        print("Bootstrap resampling")
        rs = rd.RandomState(self.seed)
        num_bins = len(self.edge_bins) - 1
        val_matrix = np.empty((bsrs, num_bins)) 
        flag_no_pixels = -100
        tau_fake_min = np.empty(bsrs)
        for i in range(bsrs):            
            if (i + 1) % 100 == 0:
                print("...", i + 1)
            # Make the fake spectrum of length num_chunks
            tau_fake_x = np.empty(0) 
            tau_fake_y = np.empty(0) 
            for j in range(self.num_chunks):
                index = rs.randint(self.num_chunks)
                tau_fake_x = np.append(tau_fake_x, tau_chunks_x[index])
                tau_fake_y = np.append(tau_fake_y, tau_chunks_y[index])
            # Find the percentile of the optical depth in each bin
            for j in range(num_bins):
                bin_pixels = np.where((tau_fake_x > self.edge_bins[j]) & 
                    (tau_fake_x < self.edge_bins[j+1]))[0]
                val_matrix[i][j] = (np.percentile(tau_fake_y[bin_pixels], 
                    self.percentile_value) if (len(bin_pixels) > 0) else flag_no_pixels)
            # Also get the errors on tau_min    
            index_min = np.where(tau_fake_x < self.tau_c) 
            tau_fake_min[i] = np.percentile(tau_fake_y[index_min], self.percentile_value)    
        val_matrix = val_matrix.transpose()    
        el = np.empty(num_bins)
        eu = np.empty(num_bins)    
        for i in range(num_bins):    
            tmp = val_matrix[i]
            tmp = tmp[tmp != flag_no_pixels] 
            if len(tmp) > 0:
                el[i] = np.percentile(tmp, un.one_sigma_below)
                eu[i] = np.percentile(tmp, un.one_sigma_above)
        el = np.delete(el, self.nan)
        eu = np.delete(eu, self.nan)
        self.tau_binned_err = (self.tau_binned_y - el, -self.tau_binned_y + eu) 
        self.tau_min_err = (self.tau_min - np.percentile(tau_fake_min, un.one_sigma_below),
                            -self.tau_min + np.percentile(tau_fake_min, un.one_sigma_above))

    def _calc_errors_legacy(self, bsrs, tau_chunks_x, tau_chunks_y):
        # bootstrap resampling
        print("Bootstrap resampling")
        rs = rd.RandomState(self.seed)
        num_bins = len(self.edge_bins) - 1
        val_matrix = np.empty((bsrs, num_bins)) 
        flag_no_pixels = -100
        tau_fake_min = np.empty(bsrs)
        for i in range(bsrs):            
            if (i + 1) % 100 == 0:
                print("...", i + 1)
            # Make the fake spectrum of length num_chunks
            tau_fake_x = np.empty(0) 
            tau_fake_y = np.empty(0) 
            for j in range(self.num_chunks):
                index = rs.randint(self.num_chunks)
                tau_fake_x = np.append(tau_fake_x, tau_chunks_x[index])
                tau_fake_y = np.append(tau_fake_y, tau_chunks_y[index])
            # Find the percentile of the optical depth in each bin
            for j in range(num_bins):
                bin_pixels = np.where((tau_fake_x > self.edge_bins[j]) & 
                    (tau_fake_x < self.edge_bins[j+1]))[0]
                val_matrix[i][j] = (np.percentile(tau_fake_y[bin_pixels], 
                    self.percentile_value) if (len(bin_pixels) > 0) else flag_no_pixels)
            # Also get the errors on tau_min    
            index_min = np.where(tau_fake_x < self.tau_c) 
            tau_fake_min[i] = np.percentile(tau_fake_y[index_min], self.percentile_value)    
        val_matrix = val_matrix.transpose()    
        el = np.empty(num_bins)
        eu = np.empty(num_bins)    
        for i in range(num_bins):    
            tmp = val_matrix[i]
            tmp = tmp[tmp != flag_no_pixels] 
            if len(tmp) > 0:
                el[i] = np.percentile(tmp, un.one_sigma_below)
                eu[i] = np.percentile(tmp, un.one_sigma_above)
        el = np.delete(el, self.nan)
        eu = np.delete(eu, self.nan)
        self.tau_binned_err = (self.tau_binned_y - el, -self.tau_binned_y + eu) 
        self.tau_min_err = (self.tau_min - np.percentile(tau_fake_min, un.one_sigma_below),
                            -self.tau_min + np.percentile(tau_fake_min, un.one_sigma_above))



def bin_pixels_from_SpecWizard(ion_x: str, ion_y: str, *spectra: "UserInput",
                               x_limits: Tuple[float, float] = (-5.75, 4.0),
                               bin_log_space_width: float = 0.25,
                               chunk_width: float = 5.0,
                               percentile: float = 50.0,
                               n_bootstrap_resamples: int = 1000,
                               random_seed: int = 12345,
                               legacy: bool = False
    ) -> Tuple[TauBinned, ...]:
    """
    Create bin objects from individual spectra (compatible with the output of "from_SpecWizard()").

    Parameters:
        str                              ion_x -> UserInput object attribute name containing the ion optical depths to bin by.
        str                              ion_y -> UserInput object attribute name containing the ion optical depths to populate each bin with.
        (args) UserInput               spectra -> All specttrum objects (passed as individual params).
        (float, float)                x_limits -> Interval to bin between as a tuple. Default is (-5.75, 4.0).
        float              bin_log_space_width -> Width of a bin in log10 space alon ghe X-axis. Default is 0.25.
        float                      chunk_width -> Width of a single chunk of pixels in Angstroms. Defualt is 5.
        float                       percentile -> Percentile to calculate as a percentage. Defualt is the median (50%).
        int              n_bootstrap_resamples -> The number of resamples used when calculating errors. Set to 0 to disable error calculation. Default is 1,000.
        int                        random_seed -> Seed for the random number generator used by the error calculation. Default is 12345.
        bool                            legacy -> Use the legacy code for binning and calculating errors (default is False).

    Returns:
        Tuple[TauBinned] -> An instance of TauBinned - one for each input spectrum.
    """
    
    binned_objects = []
    for spectrum in spectra:
        binned_objects.append(TauBinned(
                tau_x_object = getattr(spectrum, ion_x),
                tau_y_object = getattr(spectrum, ion_y),
                   tau_x_min = x_limits[0],
                   tau_x_max = x_limits[1],
                    bin_size = bin_log_space_width,
                  chunk_size = chunk_width,
            percentile_value = percentile,
                        bsrs = n_bootstrap_resamples,
                        seed = random_seed,
             use_legacy_code = legacy
        ))
    return tuple(binned_objects)



class BinnedOpticalDepthResults(object):
    def __init__(self, ion_x, ion_y, pixel_wavelength, pixel_x, pixel_y, percentile, tau_binned_x, tau_binned_y, tau_binned_err, tau_min):
        self.ion_x: str = ion_x
        self.ion_y: str = ion_y
        self.percentile: float = percentile
        self.tau_binned_x: np.ndarray = tau_binned_x
        self.tau_binned_y: np.ndarray = tau_binned_y
        self.tau_binned_err: Union[np.ndarray, None] = tau_binned_err
        self.tau_min: float = tau_min
        self.pixel_wavelength: np.ndarray = pixel_wavelength
        self.pixel_x: np.ndarray = pixel_x
        self.pixel_y: np.ndarray = pixel_y

    @staticmethod
    def from_TauBinned(data: TauBinned):
        return BinnedOpticalDepthResults(ion_x = data.ion_x,
                                         ion_y = data.ion_y,
                                         pixel_wavelength = data.pix_pairs_wavelength,
                                         pixel_x = data.pix_pairs_x,
                                         pixel_y = data.pix_pairs_x,
                                         percentile = data.percentile_value,
                                         tau_binned_x = data.tau_binned_x,
                                         tau_binned_y = data.tau_binned_y,
                                         tau_binned_err = data.tau_binned_err if "tau_binned_err" in vars(data) else None,
                                         tau_min = data.tau_min)

    @property
    def has_errors(self):
        return self.tau_binned_err is not None

    @property
    def is_median(self):
        return self.percentile == 50.0

def bin_combined_pixels_from_SpecWizard(ion_x: str, ion_y: str, *spectra: "UserInput",
                                        x_limits: Tuple[float, float] = (-5.75, 4.0),
                                        bin_log_space_width: float = 0.25,
                                        chunk_width: float = 5.0,
                                        percentile: float = 50.0,
                                        n_bootstrap_resamples: int = 1000,
                                        random_seed: int = 12345,
                                        legacy: bool = False
    ) -> Dict[str, Union[np.ndarray, float, None]]:
    """
    Combines and bins multiple spectra (compatible with the output of "from_SpecWizard()").

    Parameters:
        str                              ion_x -> UserInput object attribute name containing the ion optical depths to bin by.
        str                              ion_y -> UserInput object attribute name containing the ion optical depths to populate each bin with.
        (args) UserInput               spectra -> All specttrum objects (passed as individual params).
        (float, float)                x_limits -> Interval to bin between as a tuple. Default is (-5.75, 4.0).
        float              bin_log_space_width -> Width of a bin in log10 space alon ghe X-axis. Default is 0.25.
        float                      chunk_width -> Width of a single chunk of pixels in Angstroms. Defualt is 5.
        float                       percentile -> Percentile to calculate as a percentage. Defualt is the median (50%).
        int              n_bootstrap_resamples -> The number of resamples used when calculating errors. Set to 0 to disable error calculation. Default is 1,000.
        int                        random_seed -> Seed for the random number generator used by the error calculation. Default is 12345.
        bool                            legacy -> Use the legacy code for binning and calculating errors (default is False).

    Returns:
        BinnedOpticalDepthResults(
            str                          ion_x -> Ion of the X-axis data.
            str                          ion_y -> Ion of the Y-axis data.
            float                   percentile -> Percentile calculated in each bin.
            np.ndarray            tau_binned_x -> Bin locations.
            np.ndarray            tau_binned_y -> Percentile value in each corresponding bin.
            (np.ndarray | None) tau_binned_err -> Bootstrapped Y-error in each corresponding bin. If binning is disabled, becomes None.
            float                      tau_min -> Value of tau_min, calculated as median of all individual spectrum tau_min values.
        )
    """

    single_instance = TauBinned(
            tau_x_object = getattr(spectra[0], ion_x),
            tau_y_object = getattr(spectra[0], ion_y),
               tau_x_min = x_limits[0],
               tau_x_max = x_limits[1],
                bin_size = bin_log_space_width,
              chunk_size = chunk_width,
        percentile_value = percentile,
                    bsrs = n_bootstrap_resamples,
                    seed = random_seed,
         use_legacy_code = False#legacy
    )

    spectra_wavelengths = []
    spectra_tau_x = []
    spectra_tau_y = []
    tau_min_values = []
    for spectrum in spectra:
        spectrum_lambda_x, spectrum_tau_x, spectrum_tau_y = TauBinned._find_pixel_pairs(tau_x_object = getattr(spectrum, ion_x),
                                                                                        tau_y_object = getattr(spectrum, ion_y))

        single_instance._get_tau_min(spectrum_tau_x, spectrum_tau_y)
        spectrum_tau_min = single_instance.tau_min

        spectra_wavelengths.append(spectrum_lambda_x)
        spectra_tau_x.append(spectrum_tau_x)
        spectra_tau_y.append(spectrum_tau_y)
        tau_min_values.append(spectrum_tau_min)

    tau_min = np.median(tau_min_values)

    all_wavelengths = np.concatenate(spectra_wavelengths)
    all_tau_x = np.concatenate(spectra_tau_x)
    all_tau_y = np.concatenate(spectra_tau_y)

    sort_order = np.argsort(all_wavelengths)
    all_wavelengths = all_wavelengths[sort_order]
    all_tau_x = all_tau_x[sort_order]
    all_tau_y = all_tau_y[sort_order]

    if legacy:
        tau_chunks_x, tau_chunks_y = single_instance._calc_percentiles_legacy(all_wavelengths, all_tau_x, all_tau_y,
                                                                              tau_x_min = x_limits[0], tau_x_max = x_limits[1], bin_size = bin_log_space_width)
    else:
        tau_chunks_x, tau_chunks_y = single_instance._calc_percentiles(all_wavelengths, all_tau_x, all_tau_y,
                                                                       tau_x_min = x_limits[0], tau_x_max = x_limits[1], bin_size = bin_log_space_width)

    tau_binned_x = single_instance.tau_binned_x
    tau_binned_y = single_instance.tau_binned_y
    
    if n_bootstrap_resamples > 0:
        if legacy:
            single_instance._calc_errors_legacy(n_bootstrap_resamples, tau_chunks_x, tau_chunks_y)
        else:
            single_instance._calc_errors(n_bootstrap_resamples, tau_chunks_x, tau_chunks_y)

    #tau_binned_err = single_instance.tau_binned_err

    #return {
    #    "tau_binned_x": tau_binned_x,
    #    "tau_binned_y": tau_binned_y,
    #    "tau_binned_err": tau_binned_err if n_bootstrap_resamples > 0 else None,
    #    "tau_min": tau_min
    #}
    return BinnedOpticalDepthResults(
        ion_x = ion_x,
        ion_y = ion_y,
        pixel_wavelength = all_wavelengths,
        pixel_x = all_tau_x,
        pixel_y = all_tau_y,
        percentile = percentile,
        tau_binned_x = tau_binned_x,
        tau_binned_y = tau_binned_y,
        tau_binned_err = single_instance.tau_binned_err if n_bootstrap_resamples > 0 else None,
        tau_min = tau_min
    )
