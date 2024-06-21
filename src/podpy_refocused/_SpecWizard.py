# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from ._Ions import Ion

from typing import Union, List, Iterable

import numpy as np
import h5py as h5
from unyt import unyt_array



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
    def wavelengths(self) -> np.ndarray:
        return self.__wavelengths
    
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
        return self.get_spectrum_field("Noise_Sigma", index)

    def get_RSODOST(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        return self.get_spectrum_ion_field(ion, "RedshiftSpaceOpticalDepthOfStrongestTransition", index)

    def get_opticaldepth_weighted_peculiar_velocity(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[unyt_array, List[unyt_array]]:
        return self.get_spectrum_ion_opticaldepth_weighted_field(ion, "LOSPeculiarVelocity_KMpS", index, unit = "km/s")

    def get_opticaldepth_weighted_overdensity(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[np.ndarray, List[np.ndarray]]:
        return self.get_spectrum_ion_opticaldepth_weighted_field(ion, "OverDensity", index)

    def get_opticaldepth_weighted_temperature(self, ion: Union[Ion, str], index: Union[int, str, slice, Iterable[bool], Iterable[int], Iterable[str], None] = None) -> Union[unyt_array, List[unyt_array]]:
        return self.get_spectrum_ion_opticaldepth_weighted_field(ion, "Temperature_K", index, unit = "K")
