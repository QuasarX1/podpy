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

from . import _universe as un
from ._Pod import Pod, LymanAlpha, Metal
from ._Ions import Ion
from ._SpecWizard import SpecWizard_Data

import os
from typing import cast, Union, Tuple, List, Callable

import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py as h5
from QuasarCode import Console


class Spectrum:
    """
    Creates a spectrum object.

    Parameters
    ----------
    z_qso: float
        Redshift of the quasar
    wavelength_px: np.ndarray[float]
        Wavelengths of each pixel.
    flux_px: np.ndarray[float]
        Raw flux in each pixel.
    noise_sigma_px: np.ndarray[float]
        Noise standard deviation associated with each pixel.
    object_name: str
        Name of the object, important for locating relevant files
    filepath: str, optional
        The directory tree leading to the flux, error array and mask files
    mask_badreg: boolean, optional
        Whether bad regions should be manually masked using a file provided by the user. 
        The file should have the format filepath + object_name + "_badreg.mask", and
        contain two tab-separated text columns, where each row contains the start and end
        wavelength of a region to mask.  
    mask_dla: boolean, optional
        Same as mask_badreg, except for masking DLAs, and the file should be named
        filepath + objecta_name + '_dla.mask'.
    
    Instance attributes
    ----------
    lambdaa: wavelength in angstroms
    flux: normalized flux array
    sigma_noise: error array 				
    npix: length of the above arrays 
    """
    BAD_NOISE = -1 # Value to set bad pixels to

    def __init__(
                    self,
                    z_qso:          float,
                    wavelength_px:  np.ndarray,
                    flux_px:        np.ndarray,
                    noise_sigma_px: np.ndarray,
                    object_name:    Union[str, None] = None,
                    filepath:       Union[str, None] = None,
                    mask_badreg:    bool             = False,
                    mask_dla:       bool             = False
                ) -> None:

        self.z_qso:       float            = z_qso
        self.object_name: Union[str, None] = object_name
        self.filepath:    str              = filepath if filepath is not None else "."
        self.mask_badreg: bool             = mask_badreg
        self.mask_dla:    bool             = mask_dla

        self.npix:        int        = len(wavelength_px)
        self.lambdaa:     np.ndarray = wavelength_px
        self.flux:        np.ndarray = flux_px
        self.sigma_noise: np.ndarray = noise_sigma_px
        self.flag:        np.ndarray = None

        # Mask any bad regions or dlas
        self.mask_spectrum()

        # Prep spectrum
        self.prep_spectrum()

    def mask_spectrum(self) -> None:
        """
        Search for mask files and apply the mask if they exist   
        """

        if self.mask_badreg or self.mask_dla:
            mask_type_list = []
            if self.mask_badreg:
                mask_type_list.append("badreg")
            if self.mask_dla:
                mask_type_list.append("dla")

            Console.print_verbose_info("Looking for mask files:")

            for mask_type in mask_type_list:
                mask_file = os.path.join(self.filepath, f"{self.object_name}_{mask_type}.mask")
                if os.path.isfile(mask_file):
                    Console.print_verbose_info(f"    found {mask_type} file")
                    data = np.loadtxt(mask_file)
                    if len(data.shape) == 1:
                        data = [data]
                    for row in data:
                        mask_region_idx = np.where((self.lambdaa >= row[0]) & (self.lambdaa <= row[1]))
                        self.sigma_noise[mask_region_idx] = Spectrum.BAD_NOISE
                else:
                    Console.print_warning(f"No {mask_type} file found.")


    def prep_spectrum(self) -> None:
        """
        Prepares the spectrum before implementing POD 
        by setting up the flag array that keeps track of bad / corrected pixels,
        and setting up the optical depth arrays and functions. 
        """

        # Flag regions where spectrum is no good (noise <= 0)
        # and set noise to -1 exactly for consistency in interpolation
        self.flag = np.zeros(self.npix, dtype = int)
        bad_pix = np.where(self.sigma_noise <= 0)
        self.sigma_noise[bad_pix] = Spectrum.BAD_NOISE
        self.flag[bad_pix] = Pod.FLAG_DISCARD
        # Set the flux to tau_min since they are bad regions, should not have signal
        self.flux[bad_pix] = 1.0
        # Prepare the functions for interpolating
        self.flux_function = self.interp_f_lambda(self.lambdaa, self.flux)
        self.sigma_noise_function = self.interp_f_lambda(self.lambdaa,self.sigma_noise)
        negflux = self.flux <= 0
        self.tau = np.where(negflux, Pod.TAU_MAX, -np.log(self.flux))
        self.tau_function = self.interp_f_lambda(self.lambdaa, self.tau)

    @staticmethod
    def interp_f_lambda(lambdaa, f_lambda) -> intp.interp1d:
        """
        Linearly interpolates a function of lambda

        Parameters
        ----------
        lambdaa: array_like
            Wavelength array.
        f_lambda: array_like	
            Whichever quantity is a function of the wavelength. 

        Returns
        -------
        f: function of wavelength  

        """
        return intp.interp1d(lambdaa, f_lambda, kind='linear')

    def get_tau_rec_h1(self, **kwargs) -> LymanAlpha:
        """
        Get the optical depth of HI, returned as an object named self.h1.    

        Parameters
        ----------
        label: str, optional
            Adds a label to the object that is attached to the spectrum. e.g., label = "_A" 
            would result in an object named "h1_A". 	

        See Pod class for remaining input parameters.

        """
        name = "h1" + kwargs.pop("label", "")
        vars(self)[name] = LymanAlpha(self, **kwargs)
        return vars(self)[name]

    def get_tau_rec_ion(self, ion: str, *args, **kwargs) -> Metal:
        """
        Get the optical depth of ion $ion, returned as an object named self.$ion

        Parameters
        ----------
        ion: str
            Currently supported: c2, c3, c4, n5, o1, o6, si2, si3, si4 
        label: str, optional
            Adds a label to the object that is attached to the spectrum. e.g., label = "_A" 
            would result in an object named "$ion_A". 	

        See Pod class for remaining input parameters.

        """
        name = ion + kwargs.pop("label", "")
        vars(self)[name] = Metal(self, ion, *args, **kwargs)
        return vars(self)[name]

    def get_tau_rec_c2(self, **kwargs) -> Metal:
        """
        Get the optical depth of CII using the fiducial recovery parameters: 
            correct_h1 = False
            correct_self = False
            take_min_doublet = False	
        Returns an object named self.c2.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "c2",
            correct_h1       = False,
            correct_self     = False,
            take_min_doublet = False,
            **kwargs
        )

    def get_tau_rec_c3(self, **kwargs) -> Metal:
        """
        Get the optical depth of CIII using the fiducial recovery parameters: 
            correct_h1 = True 
            correct_self = False
            take_min_doublet = False	
        Returns an object named self.c3.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "c3",
            correct_h1       = True,
            correct_self     = False,
            take_min_doublet = False,
            **kwargs
        )

    def get_tau_rec_c4(self, **kwargs) -> Metal:
        """
        Get the optical depth of CIV using the fiducial recovery parameters: 
            correct_h1 = False 
            correct_self = True 
            take_min_doublet = False	
        Returns an object named self.c4.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "c4",
            correct_h1       = False,
            correct_self     = True,
            take_min_doublet = False,
            **kwargs
        )

    def get_tau_rec_n5(self, **kwargs) -> Metal:
        """
        Get the optical depth of NV using the fiducial recovery parameters: 
            correct_h1 = False 
            correct_self = False 
            take_min_doublet = True	
        Returns an object named self.n5.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "n5",
            correct_h1       = False,
            correct_self     = False,
            take_min_doublet = True,
            **kwargs
        )

    def get_tau_rec_o1(self, **kwargs) -> Metal:
        """
        Get the optical depth of OI using the fiducial recovery parameters: 
            correct_h1 = False 
            correct_self = False 
            take_min_doublet = False 
        Returns an object named self.o1.

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "o1",
            correct_h1       = False,
            correct_self     = False,
            take_min_doublet = False,
            **kwargs
        )

    def get_tau_rec_o6(self, **kwargs) -> Metal:
        """
        Get the optical depth of OVI using the fiducial recovery parameters: 
            correct_h1 = True 
            correct_self = False 
            take_min_doublet = True	
        Returns an object named self.o6.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "o6",
            correct_h1       = True,
            correct_self     = False,
            take_min_doublet = True,
            **kwargs
        )

    def get_tau_rec_si2(self, **kwargs) -> Metal:
        """
        Get the optical depth of SiII using the fiducial recovery parameters: 
            correct_h1 = False 
            correct_self = False 
            take_min_doublet = False 
        Returns an object named self.si2.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "si2",
            correct_h1       = False,
            correct_self     = False,
            take_min_doublet = False,
            **kwargs
        )

    def get_tau_rec_si3(self,  **kwargs) -> Metal:
        """
        Get the optical depth of SiIII using the fiducial recovery parameters: 
            correct_h1 = False 
            correct_self = False 
            take_min_doublet = False	
        Returns an object named self.si3.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "si3",
            correct_h1       = False,
            correct_self     = False,
            take_min_doublet = False,
            **kwargs
        )

    def get_tau_rec_si4(self, **kwargs) -> Metal:
        """
        Get the optical depth of SiIV using the fiducial recovery parameters: 
            correct_h1 = False 
            correct_self = False 
            take_min_doublet = True	
        Returns an object named self.si4.  

        See Pod class for optional input parameters. 
        """
        return self.get_tau_rec_ion(
            ion              = "si4",
            correct_h1       = False,
            correct_self     = False,
            take_min_doublet = True,
            **kwargs
        )

    def _get_ion_data(self, ion_name):
        if ion_name not in vars(self):
            getattr(self, f"get_tau_rec_{ion_name}")()
        return vars(self)[ion_name]
    @property
    def h1_data(self)  -> LymanAlpha: return self._get_ion_data("h1")
    @property
    def c2_data(self)  -> Metal:      return self._get_ion_data("c2")
    @property
    def c3_data(self)  -> Metal:      return self._get_ion_data("c3")
    @property
    def c4_data(self)  -> Metal:      return self._get_ion_data("c4")
    @property
    def n5_data(self)  -> Metal:      return self._get_ion_data("n5")
    @property
    def o1_data(self)  -> Metal:      return self._get_ion_data("o1")
    @property
    def o6_data(self)  -> Metal:      return self._get_ion_data("o6")
    @property
    def si2_data(self) -> Metal:      return self._get_ion_data("si2")
    @property
    def si3_data(self) -> Metal:      return self._get_ion_data("si3")
    @property
    def si4_data(self) -> Metal:      return self._get_ion_data("si4")

    def fit_continuum(
                        self,
                        bin_size:       float = 20.0,
                        n_cf_sigma:     float = 2.0,
                        max_iterations: int   = 20
                    ) -> None:
        """
        Automatically re-fit the continuum redward of the QSO Lya emission 
        in order to homogenize continuum fitting errors between different quasars
        """

        Console.print_verbose_info("Fitting continuum...")

        # Set up the area to fit
        lambda_min = (1.0 + self.z_qso) * un.lambda_h1[0]
        index_correction = np.where(self.lambdaa > lambda_min)[0]
        if len(index_correction) == 0:
            raise RuntimeError("Unable to fit continuum redwards of Ly alpha: no pixels redward of LyA!")
        lambdaa = self.lambdaa[index_correction]
        flux = self.flux[index_correction].copy()
        sigma_noise = self.sigma_noise[index_correction].copy()
        flag = self.flag[index_correction].copy()

        # Get the bins
        bin_size = bin_size * (1. + self.z_qso)
        lambda_max = lambdaa[-1]
        nbins = np.floor((lambda_max - lambda_min) / bin_size)
        nbins = int(nbins)
        bin_size = (lambda_max - lambda_min) / nbins
        Console.print_verbose_info("...using", nbins, "bins of", bin_size, "A")
        bin_edges = lambda_min + np.arange(nbins + 1) * bin_size
        pixels_list = [None] * nbins
        lambda_centre = np.empty(nbins)
        for i in range(nbins):
            pixels_list[i] = np.where((lambdaa > bin_edges[i]) & (lambdaa <= bin_edges[i+1]))[0]
            lambda_centre[i] = bin_edges[i] + bin_size / 2.

        # Throw this all into a while loop until convergence
        flag_interp = flag.copy()
        flux_interp = flux
        converged = 0
        medians = np.empty(nbins)
        medians_flag = np.zeros(nbins, dtype=int)
        niter = 0
        lambda_interp = lambda_centre 
        medians_interp = np.empty(nbins)
        while converged == 0 and niter < max_iterations:
            niter += 1
            for i in range(nbins):
                pixels = pixels_list[i]
                lambda_bin = lambdaa[pixels]
                flux_bin = flux[pixels]
                medians[i] = np.median(flux_bin[flag_interp[pixels] % 2 == 0])
                if np.isnan(medians[i]):
                    medians_flag[i] = 1
            for i in range(nbins):
                if medians_flag[i] == 1:
                    medians[i] = medians[(i-1) % nbins]
            medians_interp = medians
            # Interpolate
            flux_function = intp.splrep(lambda_interp, medians_interp, k = 3)
            flux_interp = intp.splev(lambdaa, flux_function)
            flag_interp_new = flag.copy()
            bad_pix = np.where((flux_interp - flux) > n_cf_sigma * sigma_noise)[0]
            flag_interp_new[bad_pix] = 1
            # Test for convergence 
            if (flag_interp_new == flag_interp).all():
                converged = 1
            flag_interp = flag_interp_new

        # Divide out the spectrum in the fitted part
        self.flux_old = self.flux.copy()
        self.sigma_noise_old = self.sigma_noise.copy()
        bad_pix = np.where(flag % 2 == 1)
        flux_interp[bad_pix] = flux[bad_pix] 
        self.flux[index_correction] = flux / flux_interp 
        self.sigma_noise[index_correction] = sigma_noise / flux_interp 
        self.flux_function = self.interp_f_lambda(self.lambdaa, self.flux)
        self.sigma_noise_function = self.interp_f_lambda(self.lambdaa, self.sigma_noise)

        Console.print_verbose_info("...done\n")


    def plot_spectrum(self):
        """	
        Quick view plot of QSO spectrum
        """
        fig, ax = plt.subplots(figsize = (12, 3))
        ax.plot(self.lambdaa, self.flux, lw = 0.5, c = 'b')
        ax.plot(self.lambdaa, self.sigma_noise, lw = 0.5, c = 'r')
        ax.axhline(y = 0, ls = ':', c = 'k', lw = 0.5)
        ax.axhline(y = 1, ls = ':', c = 'k', lw = 0.5)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("$\lambda$ [\AA]")
        ax.set_ylabel("Normalized flux")
        ax.minorticks_on()
        fig.show()

class KodiaqFits_Spectrum(Spectrum):
    """
    Createa spectrum from KODIAQ fits files. The flux file and error array, 
    respectively, should be located in the files:
    filepath + objectname + _f.fits 
    filepath + objectname + _e.fits 
    """

    def __init__(
                    self,
                    z_qso:       float,
                    object_name: str,
                    filepath:    str    = ".",
                  **kwargs
                ):

        # Read in flux file
        flux_filepath = os.path.join(filepath, f"{object_name}_f.fits")
        flux_file = fits.open(flux_filepath)[0]
        flux = flux_file.data

        # Get wavelength array
        crpix1 = flux_file.header['CRPIX1'] # reference pixel
        crval1 = flux_file.header['CRVAL1'] # coordinate at reference pixel
        cdelt1 = flux_file.header['CDELT1'] # coordinate increment per pixel
        length = flux_file.header['NAXIS1'] 
        wavelengths = 10**((np.arange(length) + 1.0 - crpix1) * cdelt1 + crval1)

        # Read in error file
        err_filepath = os.path.join(filepath, f"{object_name}_e.fits")
        err_file = fits.open(err_filepath)[0]
        sigma_noise = err_file.data

        Spectrum.__init__(
                            self,
                            z_qso = z_qso,
                            wavelength_px = wavelengths,
                            flux_px = flux,
                            noise_sigma_px = sigma_noise,
                            object_name = object_name,
                            filepath = filepath
                          **kwargs
                          )



def from_SpecWizard(
                    filepath:                  str   = "./LongSpectrum.hdf5",
                    object_name:               Union[str, None]                   = None,
                    mask_directory:            Union[str, None]                   = None,
                    sightline_filter:          Union[List[bool], List[int], None] = None,
                    mask_bad_regions:          bool  = True,
                    mask_dla:                  bool  = True,
                    identify_h1_contamination: bool  = True,
                    correct_h1_contamination:  bool  = True,
                    n_higher_order_lyman:      int   = 16,
                    saturationn_sigma_limit:   float = None
                   ) -> Tuple[Spectrum, ...]:
    """
    Loads data from a .hdf5 SpecWizard output file.

    Set 'mask_dla' to False if recovering O VI (see example script).
    """

    data = SpecWizard_Data(filepath)
    n_spectra = len(data)

    # Read data

    quasar_redshift = float(data.header["Z_qso"])

    wavelengths = data.wavelengths
    fluxes = data.get_flux(sightline_filter)
    flux_error_sigmas = data.get_flux_noise_sigma(sightline_filter)
    n_spectra = len(fluxes)

#    data = h5.File(filepath)
#
#    # Identify what spectra are avalible
#
#    first_spec_num = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["first_specnum"])
#    n_spectra = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["NumberOfSpectra"])
#
#    # Handle the sightline filter to ensure a common format
#
#    if sightline_filter is None:
#        sightline_filter = np.full(n_spectra, True, dtype = bool)
#    elif len(sightline_filter) != n_spectra:
#        if len(sightline_filter) == 0:
#            raise ValueError("Provided filter has length 0.")
#        elif isinstance(sightline_filter[0], bool):
#            raise ValueError(f"SpecWizard output file contained {n_spectra} spectra but provided filter expected {len(sightline_filter)}.")
#
#    if isinstance(sightline_filter[0], bool) or isinstance(sightline_filter[0], np.bool_):
#        pass
#    elif isinstance(sightline_filter[0], int):
#        sightline_filter = [i in sightline_filter for i in range(n_spectra)]
#    else:
#        raise TypeError(f"Sightline filter type should be integers or booleans, not {type(sightline_filter[0])}.")
#
#    # Generate the spectrum numbers to be read
#
#    spec_nums = tuple([v for v in range(first_spec_num, first_spec_num + n_spectra) if sightline_filter[v - 1]])
#    n_spectra = len(spec_nums)
#
#    # Read data
#
#    quasar_redshift = float(data["Header"].attrs["Z_qso"])
#
#    wavelengths = data["Wavelength_Ang"][:]
#    fluxes = np.array([data[f"Spectrum{n}/Flux"][:] for n in spec_nums])
#    flux_error_sigmas = np.array([(data[f"Spectrum{n}/Noise_Sigma"][:] if "Noise_Sigma" in data[f"Spectrum{n}"] else np.zeros_like(fluxes, dtype = float)) for n in spec_nums])

    # Set options for H I recovery

    lyman_alpha_kwargs = {}
    lyman_alpha_kwargs["n_higher_order"] = n_higher_order_lyman
    lyman_alpha_kwargs["correct_contam"] = 0 if not identify_h1_contamination else 1 if not correct_h1_contamination else 2
    if saturationn_sigma_limit is not None:
        lyman_alpha_kwargs["nsigma_sat"] = saturationn_sigma_limit

    # Generate Spectrum objects and recover H I

    spectrum_objects = []
    for i in range(n_spectra):
        spectrum_objects.append(Spectrum(
            z_qso = quasar_redshift,
            wavelength_px = wavelengths,
            flux_px = fluxes[i],
            noise_sigma_px = flux_error_sigmas[i],
            object_name = object_name,
            filepath = mask_directory,
            mask_badreg = mask_bad_regions,
            mask_dla = mask_dla
        ))
        spectrum_objects[-1].get_tau_rec_h1(**lyman_alpha_kwargs)

    return tuple(spectrum_objects)

def fit_continuum(*spectra: Spectrum) -> None:
    for spectrum in spectra:
        spectrum.fit_continuum()

def recover_c4(
               *spectra:                      Spectrum,
                observed_log10_flat_level:    Union[float, None] = None,
                apply_recomended_corrections: bool               = True,
                saturationn_sigma_limit:      float              = None
              ) -> None:
    offset = None if observed_log10_flat_level is None else 10**(observed_log10_flat_level)
    for spectrum in spectra:
        if apply_recomended_corrections:
            kwargs = {}
            if saturationn_sigma_limit is not None:
                kwargs["nsigma_sat"]    = saturationn_sigma_limit
                kwargs["nsigma_dm"]     = saturationn_sigma_limit
                kwargs["nsigma_contam"] = saturationn_sigma_limit
            spectrum.get_tau_rec_c4(**kwargs)
        else:
            # Calculate optical depths without any corrections
            spectrum.get_tau_rec_ion("c4", False, False, False)
        if observed_log10_flat_level is not None:
            spectrum.c4.tau_rec = np.log10(10**(spectrum.c4.tau_rec) + offset)

    return tuple(spectrum.c4_data for spectrum in spectra)

def recover_si4(
               *spectra:                      Spectrum,
                observed_log10_flat_level:    Union[float, None] = None,
                apply_recomended_corrections: bool = True,
                n_higher_order_lyman:         int = 16,
                saturationn_sigma_limit:      float = None
              ) -> None:
    offset = None if observed_log10_flat_level is None else 10**(observed_log10_flat_level)
    for spectrum in spectra:
        if apply_recomended_corrections:
            kwargs = {}
            if saturationn_sigma_limit is not None:
                kwargs["nsigma_sat"]    = saturationn_sigma_limit
                kwargs["nsigma_dm"]     = saturationn_sigma_limit
                kwargs["nsigma_contam"] = saturationn_sigma_limit
            spectrum.get_tau_rec_si4(n_higher_order = n_higher_order_lyman, **kwargs)
        else:
            # Calculate optical depths without any corrections
            spectrum.get_tau_rec_ion("si4", False, False, False)
        if observed_log10_flat_level is not None:
            spectrum.o6.tau_rec = np.log10(10**(spectrum.o6.tau_rec) + 10**(offset))

    return tuple(spectrum.o6_data for spectrum in spectra)

def recover_o6(
               *spectra:                      Spectrum,
                observed_log10_flat_level:    Union[float, None] = None,
                apply_recomended_corrections: bool = True,
                n_higher_order_lyman:         int = 16,
                saturationn_sigma_limit:      float = None
              ) -> None:
    offset = None if observed_log10_flat_level is None else 10**(observed_log10_flat_level)
    for spectrum in spectra:
        if apply_recomended_corrections:
            kwargs = {}
            if saturationn_sigma_limit is not None:
                kwargs["nsigma_sat"]    = saturationn_sigma_limit
                kwargs["nsigma_dm"]     = saturationn_sigma_limit
                kwargs["nsigma_contam"] = saturationn_sigma_limit
            spectrum.get_tau_rec_o6(n_higher_order = n_higher_order_lyman, **kwargs)
        else:
            # Calculate optical depths without any corrections
            spectrum.get_tau_rec_ion("o6", False, False, False)
        if observed_log10_flat_level is not None:
            spectrum.o6.tau_rec = np.log10(10**(spectrum.o6.tau_rec) + 10**(offset))

    return tuple(spectrum.o6_data for spectrum in spectra)
