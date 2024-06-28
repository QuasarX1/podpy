# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from ._Ions import Ion

from functools import singledispatchmethod
from typing import Any, Union, List, Tuple, Iterable, Collection
import os

import numpy as np
import h5py as h5
from unyt import unyt_quantity, unyt_array, angstrom
from scipy.interpolate import RegularGridInterpolator
from QuasarCode import Console



class SpecWizard_NoiseProfile(object):
    """
    Noise profile interpolation table used by SpecWizard.

    To read from an existing file, use:
        SpecWizard_NoiseProfile.read(filepath)

    To retrive the value of sigma for a given pixel, call the instance passing the wavelength(s) and normalised flux(es).

    Call signitures are:

        wavelengths (unyt.unyt_array[unyt.angstrom]),    normalised_fluxes (numpy.ndarray)

        wavelengths (numpy.ndarray),                     normalised_fluxes (numpy.ndarray)

        wavelength  (unyt.unyt_quantity[unyt.angstrom]), normalised_flux   (float)

        wavelength  (float),                             normalised_flux   (float)
    """

    def __init__(self, wavelengths: unyt_array, normalised_fluxes: np.ndarray, sigma_table: np.ndarray):
        self.__wavelengths = wavelengths.to(angstrom)
        self.__normalised_fluxes = normalised_fluxes
        self.__sigma_table = sigma_table

        # Create a 2D interpolator to replace the functionality from add_noise in SpecWizard (F90 edition)
        self.__interpolator = RegularGridInterpolator((self.__wavelengths.value, self.__normalised_fluxes), self.__sigma_table)

    @property
    def wavelengths(self) -> unyt_array:
        """
        Wavelengths (in Angstroms) of the interpolation table wavelength axis.
        """
        return self.__wavelengths.copy()

    @property
    def normalised_fluxes(self) -> np.ndarray:
        """
        Normalised fluxes of the interpolation table flux axis.
        """
        return self.__normalised_fluxes.copy()

    @property
    def sigma_table(self) -> np.ndarray:
        """
        Values of noise sigma for 2D coordinates (wavelength, flux).
        """
        return self.__sigma_table.copy()

    @property
    def interpolator(self) -> RegularGridInterpolator:
        """
        scipy.interpolate.RegularGridInterpolator instance for selecting values of sigma.
        """
        return self.__interpolator

    @singledispatchmethod
    def __call__(self, wavelengths: Any, normalised_fluxes: Any) -> Union[np.ndarray, float]:
        raise TypeError(f"Unexpected wavelength type: {type(wavelengths)}")
    @__call__.register(np.ndarray)
    def _(self, wavelengths: np.ndarray, normalised_fluxes: np.ndarray) -> np.ndarray:
        result = np.empty_like(normalised_fluxes)
        in_bounds = (self.__wavelengths[0] <= wavelengths) & (wavelengths <= self.__wavelengths[-1])
        result[in_bounds] = self.__interpolator(np.column_stack((wavelengths[in_bounds], normalised_fluxes[in_bounds])))
        result[~in_bounds] = result[in_bounds].max()
        return result
    @__call__.register(unyt_array)
    def _(self, wavelengths: unyt_array, normalised_fluxes: np.ndarray) -> np.ndarray:
        return self(wavelengths.to(angstrom).value, normalised_fluxes)
    @__call__.register(float)
    def _(self, wavelength: float, normalised_flux: float) -> float:
        return self(np.array([wavelength], dtype = float), np.array([normalised_flux], dtype = float))[0]
    @__call__.register(unyt_quantity)#TODO: check if this is an issue due to inheritance
    def _(self, wavelength: unyt_quantity, normalised_flux: float) -> float:
        return self(wavelength.to(angstrom).value, normalised_flux)
    
    def write(self, filepath: str, allow_overwrite: bool = False) -> None:
        """
        Write a SpecWizard noise profile file.
        """

        if not allow_overwrite and os.path.exists(filepath):
            raise FileExistsError(f"File already exists at \"{filepath}\" and overwriting was not enabled.")
        with h5.File(filepath, "w") as file:
            file.create_dataset(name = "NormalizedFlux",      data = np.array(self.__normalised_fluxes, dtype = np.float32))
            file.create_dataset(name = "Wavelength_Angstrom", data = np.array(self.__wavelengths, dtype = np.float32))
            file.create_dataset(name = "NormalizedNoise",     data = np.array(self.__sigma_table.T, dtype = np.float32))

    @staticmethod
    def read(filepath: str) -> "SpecWizard_NoiseProfile":
        """
        Read from an existing SpecWizard noise profile file.
        """

        with h5.File(filepath, "r") as file:
            return SpecWizard_NoiseProfile(
                wavelengths = unyt_array(file["Wavelength_Angstrom"][:], units = angstrom),
                normalised_fluxes = file["NormalizedFlux"][:],
                sigma_table = file["NormalizedNoise"][:].T
            )



class SpecWizard_Data(object):
    """
    Data loader for SpecWizard output files.

    Provides convenience methods for accessing data without needing to know the specifics of spectrum numbers.
    """

    def __init__(self, filepath: str):
        self.__file = h5.File(filepath)
        self.__wavelengths = self.__file["Wavelength_Ang"][:]
        self.__first_spectrum_number = int(self.__file["Parameters/SpecWizardRuntimeParameters"].attrs["first_specnum"])
        self.__len = int(self.__file["Parameters/SpecWizardRuntimeParameters"].attrs["NumberOfSpectra"])
        self.__has_noise_data = "Noise_Sigma" in self[0]

    @property
    def hdf5(self) -> h5.File:
        return self.__file

    @property
    def header(self) -> h5.AttributeManager:
        return self.__file["Header"].attrs

    @property
    def parameters_modify_metallicity(self) -> h5.AttributeManager:
        return self.__file["Header/ModifyMetallicityParameters"].attrs

    @property
    def parameters_chemical_elements(self) -> h5.AttributeManager:
        return self.__file["Parameters/ChemicalElements"].attrs

    @property
    def parameters(self) -> h5.AttributeManager:
        return self.__file["Parameters/SpecWizardRuntimeParameters"].attrs

    @property
    def wavelengths(self) -> np.ndarray:#TODO: update to be a unyt_array then fix everything that breaks!
        return self.__wavelengths

    @property
    def min_wavelength(self) -> float:
        return float(self.__wavelengths[0])

    @property
    def max_wavelength(self) -> float:
        return float(self.__wavelengths[-1])

    @property
    def noise_avalible(self) -> bool:
        return self.__has_noise_data
    
    def __len__(self):
        return self.__len
    
    def __getitem__(self, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None]):
        if index is None:
            index = slice(len(self))

        if isinstance(index, (int, str)):
            return self[[index]][0]
        
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        elif isinstance(index, Iterable):
            if isinstance(index, np.ndarray) and index.dtype == np.bool_:
                if len(index.shape) != 1 or index.shape[0] != len(self):
                    raise IndexError(f"Mismached boolean filter shape: {len(self)} spectra are avalible (1D) but the provided filter had shape {index.shape}.")
                index = [i for i in range(len(self)) if index[i]]
            elif len(index) > 0 and isinstance(index[0], bool):
                if len(index) < len(self):
                    index = list(index)
                    index.extend([False] * len(self) - len(index))

            results = []
            for i in index:
                if isinstance(i, int):
                    results.append(self.__file["Spectrum{}".format(self.__first_spectrum_number + i)])
                elif isinstance(i, str):
                    results.append(self.__file[f"Spectrum{i}"])
                else:
                    raise TypeError(f"Indexes/Keys contained an invalid type: {type(i)}")
            return results
        else:
            raise TypeError(f"Index/Key type invalid: was {type(index)}")
        
    def get_spectrum_field(self, field: str, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None, unit: Union[str, None] = None) -> Union[np.ndarray, unyt_array, List[np.ndarray], List[unyt_array]]:
        if isinstance(index, (int, str)):
            return self.get_spectrum_field(field, [index], unit)[0]
        else:
            return [(spec[field][:] if unit is None else unyt_array(input_array = spec[field][:], units = unit)) for spec in self[index]]
    
    def get_spectrum_ion_field(self, ion: Union[Ion, str], field: str, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None, unit: Union[str, None] = None) -> Union[np.ndarray, unyt_array, List[np.ndarray], List[unyt_array]]:
        if isinstance(index, (int, str)):
            return self.get_spectrum_ion_field(ion, field, [index], unit)[0]
        else:
            return [(spec[f"{ion}/{field}"][:] if unit is None else unyt_array(input_array = spec[f"{ion}/{field}"][:], units = unit)) for spec in self[index]]
    
    def get_spectrum_ion_opticaldepth_weighted_field(self, ion: Union[Ion, str], field: str, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None, unit: Union[str, None] = None) -> Union[np.ndarray, unyt_array, List[np.ndarray], List[unyt_array]]:
        if field == "RedshiftSpaceOpticalDepthOfStrongestTransition":
            return self.get_RSODOST(ion, index)
        if isinstance(index, (int, str)):
            return self.get_spectrum_ion_opticaldepth_weighted_field(ion, field, [index], unit)[0]
        else:
            return [(spec[f"{ion}/RedshiftSpaceOpticalDepthWeighted/{field}"][:] if unit is None else unyt_array(input_array = spec[f"{ion}/RedshiftSpaceOpticalDepthWeighted/{field}"][:], units = unit)) for spec in self[index]]

    def get_flux(self, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        return self.get_spectrum_field("Flux", index)

    def get_flux_noise_sigma(self, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        if not self.__has_noise_data:
            raise KeyError("Target SpecWizard output file does not contain noise data.")
        return self.get_spectrum_field("Noise_Sigma", index)

    def get_flux_noise_random_sample(self, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        if not self.__has_noise_data:
            raise KeyError("Target SpecWizard output file does not contain noise data.")
        return self.get_spectrum_field("Gaussian_deviate", index)
        
    @staticmethod
    def _apply_noise(normalised_flux: Union[float, np.ndarray], noise: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return normalised_flux + noise
        
    @staticmethod
    def _apply_noise_sigma(normalised_flux: Union[float, np.ndarray], sigma: Union[float, np.ndarray], gaussian_sample: Union[float, np.ndarray, None] = None) -> Union[float, np.ndarray]:
        args_are_single_value = False
        if not isinstance(normalised_flux, np.ndarray):
            args_are_single_value = True
            normalised_flux = np.array([normalised_flux], dtype = float)
        if gaussian_sample is None:
            gaussian_sample = np.random.normal(0, 1, normalised_flux.shape)

        new_flux = SpecWizard_Data._apply_noise(normalised_flux, sigma * gaussian_sample)
        
        if args_are_single_value:
            return new_flux[0]
        else:
            return new_flux
        
    @staticmethod
    def _sigma_from_signal_to_noise(normalised_flux: Union[float, np.ndarray], signal_to_noise: Union[float, np.ndarray], min_noise: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Negitive value(s) for S/N will result in sigma = min_noise (where S/N < 0)
        """
        if not isinstance(signal_to_noise, np.ndarray):
            if signal_to_noise < 0:
                return min_noise if isinstance(min_noise, np.ndarray) or not isinstance(normalised_flux, np.ndarray) else np.full_like(normalised_flux, min_noise)
            else:
                return (normalised_flux * (1.0 - (min_noise * signal_to_noise)) / signal_to_noise) + min_noise
        else:
            result = np.empty_like(signal_to_noise)
            ignore_mask = signal_to_noise < 0
            result = (normalised_flux * (1.0 - (min_noise * signal_to_noise)) / signal_to_noise) + min_noise
            result[ignore_mask] = min_noise[ignore_mask] if isinstance(min_noise, np.ndarray) else min_noise
            return result

    def get_noisey_flux(self, index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        if not self.__has_noise_data:
            raise KeyError("Target SpecWizard output file does not contain noise data.")
        f = self.get_flux(index)
        s = self.get_flux_noise_sigma(index)
        d = self.get_flux_noise_random_sample(index)
        return SpecWizard_Data._apply_noise_sigma(f, s, d) if isinstance(f, np.ndarray) else [SpecWizard_Data._apply_noise_sigma(f[i], s[i], d[i]) for i in range(len(f))]
#        if isinstance(f, np.ndarray):
#            return f + (s * d)
#        else:
#            return [f[i] + (s[i] * d[i]) for i in range(len(f))]
        
    def get_flux_with_artificial_signal_to_noise(
        self,
        signal_to_noise: Union[float, Collection[float]],
        min_noise: float,
        index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None,
        wavelength_boundaries: Union[Collection[Union[float, None]], None] = None,
        use_existing_random: bool = False
    ) -> Union[Tuple[np.ndarray, Union[float, np.ndarray]], Tuple[List[np.ndarray], List[Union[float, np.ndarray]]]]:
        if isinstance(signal_to_noise, Collection):
            if wavelength_boundaries is None:
                raise TypeError("Multiple signal to noise values are only supported when specifying an argument for \"wavelength_boundaries\".")
            elif len(wavelength_boundaries) - len(signal_to_noise) != 1:
                raise IndexError("Mismatched number of signal to noise values with wavelength boundaries. The length of \"wavelength_boundaries\" must be one greater than \"signal_to_noise\".")
        elif wavelength_boundaries is not None:
            if len(wavelength_boundaries) == 2:
                signal_to_noise = [signal_to_noise]
            else:
                raise ValueError(f"Unexpected number of wavelength boundary values for a single signal to noise value. If not None, must be exactly 2, but got {len(wavelength_boundaries)}.")

        if wavelength_boundaries is not None:
            # Generate an array of S/N values according to the specification

            wavelength_boundaries = list(wavelength_boundaries) # Make the collection mutable
            signal_to_noise_array = np.empty_like(self.wavelengths)

            for i in range(1, len(wavelength_boundaries) - 1):
                if wavelength_boundaries[i] is None:
                    raise TypeError(f"None value in \"wavelength_boundaries\" at index {i}. None values may only appear at either end of the list.")
            if wavelength_boundaries[0] is None:
                wavelength_boundaries[0] = self.min_wavelength - 1 # Dosen't need to be specific, just smaller than or equal to the limit
            if wavelength_boundaries[-1] is None:
                wavelength_boundaries[-1] = self.max_wavelength + 1.0 # Dosen't need to be specific, just larger than the limit

            chunk_numbers = np.digitize(self.wavelengths, wavelength_boundaries)

            for chunk_number in np.unique(chunk_numbers):
                signal_to_noise_array[chunk_numbers == chunk_number] = -1.0 if chunk_number in (0, len(wavelength_boundaries)) else (signal_to_noise[chunk_number - 1] if signal_to_noise[chunk_number - 1] is not None else -1.0)

            selected_signal_to_noise: Union[float, np.ndarray] = signal_to_noise_array

        else:
            # Just the existing S/N value is needed - change of variable tor type checking purposes
            selected_signal_to_noise: Union[float, np.ndarray] = signal_to_noise
        
        f = self.get_flux(index)
        d = self.get_flux_noise_random_sample(index) if use_existing_random else None

        if isinstance(f, np.ndarray):
            # Just a single spectrum
            sigma = SpecWizard_Data._sigma_from_signal_to_noise(f, selected_signal_to_noise, min_noise)
            return SpecWizard_Data._apply_noise_sigma(f, sigma, d), sigma
        else:
            # Multiple spectra
            sigmas = [SpecWizard_Data._sigma_from_signal_to_noise(f[i], selected_signal_to_noise, min_noise) for i in range(len(f))]
            return (
                [
                    SpecWizard_Data._apply_noise_sigma(f[i], sigmas[i], d[i] if use_existing_random else None)
                    for i
                    in range(len(f))
                ],
                sigmas
            )
    def get_flux_using_noise_profile(
        self,
        noise_profile: SpecWizard_NoiseProfile,
        index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None,
        use_existing_random: bool = False
    ) -> Union[Tuple[np.ndarray, Union[float, np.ndarray]], Tuple[List[np.ndarray], List[Union[float, np.ndarray]]]]:

        f = self.get_flux(index)
        d = self.get_flux_noise_random_sample(index) if use_existing_random else None

        if isinstance(f, np.ndarray):
            # Just a single spectrum
#            sigma = noise_profile(np.column_stack((self.wavelengths, f)))
            sigma = noise_profile(self.wavelengths, f)
            return SpecWizard_Data._apply_noise_sigma(f, sigma, d), sigma
        else:
            # Multiple spectra
#            sigmas = [noise_profile(np.column_stack((self.wavelengths, f[i]))) for i in range(len(f))]
            sigmas = [noise_profile(self.wavelengths, f[i]) for i in range(len(f))]
            return (
                [
                    SpecWizard_Data._apply_noise_sigma(f[i], sigmas[i], d[i] if use_existing_random else None)
                    for i
                    in range(len(f))
                ],
                sigmas
            )

    def get_RSODOST(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        return self.get_spectrum_ion_field(ion, "RedshiftSpaceOpticalDepthOfStrongestTransition", index)

    def get_opticaldepth_weighted_peculiar_velocity(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[unyt_array, List[unyt_array]]:
        return self.get_spectrum_ion_opticaldepth_weighted_field(ion, "LOSPeculiarVelocity_KMpS", index, unit = "km/s")

    def get_opticaldepth_weighted_overdensity(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        return self.get_spectrum_ion_opticaldepth_weighted_field(ion, "OverDensity", index)

    def get_opticaldepth_weighted_temperature(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[unyt_array, List[unyt_array]]:
        return self.get_spectrum_ion_opticaldepth_weighted_field(ion, "Temperature_K", index, unit = "K")
