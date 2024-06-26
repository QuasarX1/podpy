# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from .._SpecWizard import SpecWizard_Data, SpecWizard_NoiseProfile
from .._Ions import Ion
from .._Spectrum import SpectrumCollection # from_SpecWizard, fit_continuum, recover_c4, recover_si4, recover_o6
from .._TauBinned import bin_combined_pixels_from_SpecWizard, BinnedOpticalDepthResults
from .. import plotting
from .._universe import c as speed_of_light_kms

import numpy as np
from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper, ArrayVisuliser, cast_nullable_float, cast_nullable_list
from typing import Union, List
from matplotlib import pyplot as plt
import h5py as h5
import datetime
import os

DEFAULT_SPECWIZARD_FILENAME = "LongSpectrum.hdf5"


def _nullable_float_converter(s: str) -> Union[str, None]:
    if s == "":
        return None
    else:
        return float(s)



def main():
    ScriptWrapper(
        command = "pod-plot-specwizard-spectrum",
        authors = [ScriptWrapper.AuthorInfomation(given_name = "Christopher", family_name = "Rowe", email = "contact@cjrrowe.com", website_url = "cjrrowe.com")],
        version = "1.0.0",
        edit_date = datetime.date(2024, 6, 4),
        description = "Plots a spectrum from data in SpecWizard output files.",
        dependancies = ["podpy-refocused"],
        usage_paramiter_examples = [f"-i ./{DEFAULT_SPECWIZARD_FILENAME} --c4 --wavelength --x-min 6201 --x-max 6209 --y-min 0.97 --y-max 1.0005 --title \"S/N = 100\"'"],
        parameters = [
            ScriptWrapper.OptionalParam[List[str]](
                "datafiles", "i",
                description = f"SpecWizard output file(s).\nDefaults to a file in the current working directory named \"{DEFAULT_SPECWIZARD_FILENAME}\".\nSpecify as a semicolon seperated list to compare data between files.\nValid directories may be used, assuming the filename is the default.",
                default_value = [os.path.join(".", DEFAULT_SPECWIZARD_FILENAME)],
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.OptionalParam[List[str]](
                "datafile-names",
                description = "Display names for each input file.\nRedundant if only one data file is specified.\nSpecify a number of names as a semicolon seperated list.",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                requirements = ["datafiles"]
            ),
            ScriptWrapper.Flag(
                "raw",
                description = "Plot the raw flux.\nOnly supported for --wavelength",
                conflicts = ["raw-with-noise", "velocity", "redshift"],
                requirements = ["wavelength"]#TODO: fix QuasarCode bug (fixed?)
            ),
            ScriptWrapper.Flag(
                "raw-with-noise",
                description = "Plot the raw flux with noise added.\nOnly supported for --wavelength",
                conflicts = ["raw", "velocity", "redshift"],
                requirements = ["wavelength"]#TODO: fix QuasarCode bug (fixed?)
            ),
            ScriptWrapper.Flag(
                "h1",
                description = "Plot fluxes from H I."
            ),
            ScriptWrapper.Flag(
                "c4",
                description = "Plot fluxes from C IV."
            ),
            ScriptWrapper.Flag(
                "si4",
                description = "Plot fluxes from Si IV."
            ),
            ScriptWrapper.Flag(
                "o6",
                description = "Plot fluxes from O VI."
            ),
            ScriptWrapper.OptionalParam[List[int]](
                "target-spectra", "s",
                description = "Index(es) of the spectrum to plot.\nDefault is 0.\nSpecifying a single value for multiple data files will check for the same index in all files.\nSpecify as a semicolon seperated list if a different sightline is desired frome each file.",
                conversion_function = ScriptWrapper.make_list_converter(";", int),
                default_value = [0]
            ),
            ScriptWrapper.Flag(
                "wavelength", "w",
                description = "Use wavelength pixels.",
                conflicts = ["velocity", "redshift"]
            ),
            ScriptWrapper.Flag(
                "velocity",
                description = "Use velocity pixels.",
                conflicts = ["raw", "wavelength", "redshift"]
            ),
            ScriptWrapper.Flag(
                "redshift", "r",
                description = "Use redshift pixels.",
                conflicts = ["raw", "wavelength", "velocity"]
            ),
            ScriptWrapper.OptionalParam[float | None](
                "x-min",
                description = "Minimum X value to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "x-max",
                description = "Maximum X value to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "y-min",
                description = "Minimum Y value to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "y-max",
                description = "Maximum Y value to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[List[bool] | None](
                "ignore-noise",
                sets_param = "ignore_noise",
                description = "String containing Y or N for each input file indicating whether noise should be ignored.\nY = no noise\nN = use other nose prescription argument or fall back\non the noise prescription (if any) in the file.\nOnly applies to individual ions and will be disregarded for the with and without noise raw flux options, with the latter still being affected by the other noise alteration options.",
                conversion_function = ScriptWrapper.filter_converter,
            ),
            ScriptWrapper.OptionalParam[List[Union[float, None]] | None](
                "min-noise",
                sets_param = "min_noises",
                description = "Minimum noise value to use. One per input file as a semicolon seperated list.\nLeave elements blank to use other nose prescription argument or fall back\non the noise prescription (if any) in the file.\nElements must match populated signal to noise specifications.",
                conversion_function = ScriptWrapper.make_list_converter(";", _nullable_float_converter),
            ),
            ScriptWrapper.OptionalParam[List[Union[float, None]] | None](
                "global-signal-to-noise",
                sets_param = "signal_to_noises",
                description = "Values of signal to noise to use. One per input file as a semicolon seperated list.\nLeave elements blank to use other nose prescription argument or fall back\non the noise prescription (if any) in the file.",
                conversion_function = ScriptWrapper.make_list_converter(";", _nullable_float_converter),
                requirements = ["min-noise"]
            ),
            ScriptWrapper.OptionalParam[List[List[Union[float, None]]] | None](
                "partial-signal-to-noise",
                sets_param = "signal_to_noise_chunks",
                description = "Values of signal to noise to use - multiple values spread accross the spectrum.\nOne per input file as a semicolon seperated list, with a single file's list of values seperated by commas.\nLeave outer (;) elements blank to use other nose prescription argument or fall back\non the noise prescription (if any) for a single file.\nA given file's sub-list must contain one less element than the corresponding file-element in --partial-signal-to-noise-bin-edges.",
                conversion_function = ScriptWrapper.make_list_converter(";", ScriptWrapper.make_list_converter(",", _nullable_float_converter)),
                requirements = ["partial-signal-to-noise-bin-edges", "min-noise"]
            ),
            ScriptWrapper.OptionalParam[List[List[Union[float, None]]] | None](
                "partial-signal-to-noise-bin-edges",
                sets_param = "signal_to_noise_wavelength_bounds",
                description = "Edges of the wavelength regions (in Angstroms) within which to apply signal to noise from --partial-signal-to-noise.\nOne per input file as a semicolon seperated list, with a single file's list of values seperated by commas.\nLeave outer (;) elements blank to use other nose prescription argument or fall back\non the noise prescription (if any) for a single file.\nLeave inner (,) elements blank at only the start or end of a sub-list to indicate the chunk extends to the limit of the spectrum.\nA given file's sub-list must contain one more element than the corresponding file-element in --partial-signal-to-noise.",
                conversion_function = ScriptWrapper.make_list_converter(";", ScriptWrapper.make_list_converter(",", _nullable_float_converter)),
                requirements = ["partial-signal-to-noise", "min-noise"]
            ),
            ScriptWrapper.OptionalParam[List[str] | None](
                "noise-files",
                sets_param = "noise_files",
                description = "Use noise prescription from a SpecWizard noise prescription file - one per input file as a semicolon seperated list.\nLeave elements blank to use other nose prescription argument or fall back\non the noise prescription (if any) in the file.",
                conversion_function = ScriptWrapper.make_list_converter(";", str)
            ),
            ScriptWrapper.Flag(
                "no-h1-corrections",
                description = "Make no alterations to saturated H I pixels.",
                conflicts = ["compare-corrections", "use-RSODOST-field"]
            ),
            ScriptWrapper.Flag(
                "no-metal-corrections",
                description = "Make no alterations to saturated metal pixels.",
                conflicts = ["compare-corrections", "use-RSODOST-field"]
            ),
            ScriptWrapper.Flag(
                "compare-corrections",
                description = "Plot comparison lines for all selected ions to show their un-corrected values.",
                conflicts = ["no-h1-corrections", "no-metal-corrections", "use-RSODOST-field"]
            ),
            ScriptWrapper.Flag(
                "use-RSODOST-field",
                description = "Get ion flux from the 'RedshiftSpaceOpticalDepthOfStrongestTransition' field.",
                conflicts = ["no-h1-corrections", "no-metal-corrections", "compare-corrections", "ignore-noise", "min-noise", "global-signal-to-noise", "partial-signal-to-noise", "partial-signal-to-noise-bin-edges", "noise-files"]
            ),
            ScriptWrapper.OptionalParam[str | None](
                "title",
                description = "Optional plot title."
            ),
            ScriptWrapper.OptionalParam[str | None](
                "output-file", "o",
                description = "Name of the file to save plot as.\nLeave unset to show the plot in a matplotlib interactive window."
            ),
        ]
    ).run(__main)



def __main(
            datafiles: List[str],
            datafile_names: Union[List[str], None],
            raw: bool,
            raw_with_noise: bool,
            h1: bool,
            c4: bool,
            si4: bool,
            o6: bool,
            target_spectra: List[int],
            wavelength: bool,
            velocity: bool,
            redshift: bool,
            x_min: Union[float, None],
            x_max: Union[float, None],
            y_min: Union[float, None],
            y_max: Union[float, None],
            ignore_noise: Union[List[bool], None],
            min_noises: Union[List[float], None],
            signal_to_noises: Union[List[float], None],
            signal_to_noise_chunks: Union[List[List[float]], None],
            signal_to_noise_wavelength_bounds: Union[List[List[float]], None],
            noise_files: Union[List[str], None],
            no_h1_corrections: bool,
            no_metal_corrections: bool,
            compare_corrections: bool,
            use_RSODOST_field: bool,
            title: Union[str, None],
            output_file: Union[str, None]
        ) -> None:
    
    if not (raw or raw_with_noise or h1 or c4 or si4 or o6):
        Console.print_error("Target ion(s) must be specified!")
        Console.print_info("Terminating...")
        return

    if not (wavelength or velocity or redshift):
        Console.print_error("Spectrum X-axis type must be specified! (options are '--wavelength', '--velocity' or '--redshift')")
        Console.print_info("Terminating...")
        return
    
    if len(datafiles) > 1 and sum([(raw or raw_with_noise), h1, c4, si4, o6]) > 1:
        Console.print_error("Only one ion can be specified when comparing different files!")
        Console.print_info("Terminating...")
        return
    
    if len(datafiles) > 1 and len(target_spectra) != 1 and len(target_spectra) != len(datafiles):
        Console.print_error("Invalid number of spectrum indexes.\nEither just a single index should be specified, or a number of indexes matching the number of data files.")
        Console.print_info("Terminating...")
        return

    if len(datafiles) > 1 and datafile_names is not None and len(datafile_names) != len(datafiles):
        Console.print_error("Invalid number of data file names.\nIf specified, must be a number of elements matching the number of data files.")
        Console.print_info("Terminating...")
        return
    
    # Chect noise prescription settings
    custom_noise = ignore_noise is not None or signal_to_noises is not None or signal_to_noise_chunks is not None or signal_to_noise_wavelength_bounds is not None or noise_files is not None
    if custom_noise:
        noise_settings = [None] * len(datafiles)
        file_indexes = list(range(len(datafiles)))
        if ignore_noise is not None:
            for i in file_indexes:
                if ignore_noise[i] is not None:
                    noise_settings[i] = {
                        "use_flux_without_noise" : True
                    }
        if signal_to_noises is not None:
            for i in file_indexes:
                if signal_to_noises[i] is not None:
                    if noise_settings[i] is not None:
                        Console.print_error(f"Conflicting noise alterations for file number {i + 1}.")
                        Console.print_info("Terminating...")
                        return
                    noise_settings[i] = {
                        "noise_override_signal_to_noise" : signal_to_noises[i],
                        "noise_override_min_noise" : min_noises[i]
                    }
        if signal_to_noise_chunks is not None or signal_to_noise_wavelength_bounds is not None:
            if signal_to_noise_chunks is None or signal_to_noise_wavelength_bounds is None or min_noises is None:
                Console.print_error("Only one of --partial-signal-to-noise, --partial-signal-to-noise-bin-edges and --min-noise specified.\nAll are required if either of the first two are used.")
                Console.print_info("Terminating...")
                return
            for i in file_indexes:
                if signal_to_noise_chunks[i] is not None:
                    if noise_settings[i] is not None:
                        Console.print_error(f"Conflicting noise alterations for file number {i + 1}.")
                        Console.print_info("Terminating...")
                        return
                    noise_settings[i] = {
                        "noise_override_signal_to_noise" : signal_to_noise_chunks[i],
                        "noise_override_signal_to_noise_wavelength_limits" : signal_to_noise_wavelength_bounds[i],
                        "noise_override_min_noise" : min_noises[i]
                    }
        if noise_files is not None:
            for i in file_indexes:
                if noise_files[i] != "":
                    if noise_settings[i] is not None:
                        Console.print_error(f"Conflicting noise alterations for file number {i + 1}.")
                        Console.print_info("Terminating...")
                        return
                    noise_settings[i] = {
                        "noise_override_profile" : SpecWizard_NoiseProfile.read(noise_files[i])
                    }
        for i in file_indexes:
            if noise_settings[i] is None:
                noise_settings[i] = {}
    
    if Settings.verbose:
        Console.print_verbose_info("Reading data from the following files/directories:\n    {}".format("    \n".join(datafiles)))
        if raw: Console.print_verbose_info("Displaying raw flux pixels.")
        if raw_with_noise: Console.print_verbose_info("Displaying raw flux pixels with noise added.")
        if h1: Console.print_verbose_info("Displaying H I pixels.")
        if c4: Console.print_verbose_info("Displaying C IV pixels.")
        if si4: Console.print_verbose_info("Displaying Si IV pixels.")
        if o6: Console.print_verbose_info("Displaying O VI pixels.")
        if wavelength: Console.print_verbose_info("Pixels in wavelength space (Angstroms).")
        elif velocity: Console.print_verbose_info("Pixels in velocity space (km/s).")
        elif redshift: Console.print_verbose_info("Pixels in redshift space.")
        if not compare_corrections:
            Console.print_verbose_info("H I corrections: {}".format("no" if no_h1_corrections else "yes"))
            Console.print_verbose_info("Metal corrections: {}".format("no" if no_metal_corrections else "yes"))
        else:
            Console.print_verbose_info("Compairing corrections.")
        if use_RSODOST_field: Console.print_verbose_info("Reading optical depths from \"RedshiftSpaceOpticalDepthOfStrongestTransition\" field.")
        Console.print_verbose_info("Output: {}".format("interactive plot" if output_file is None else output_file))
    
    if datafile_names is None:
        datafile_names = [f"File #{n}" for n in range(1, 1 + len(datafiles))]
    
    # Load and process data

    x_data = []
    flux_data = []
    labels = []
    colours = []
    linestyles = []
    alphas = []

#pylint:disable-next=consider-using-enumerate
    for datafile_index in range(len(datafiles)):
        datafile = datafiles[datafile_index]

        if not os.path.isfile(datafile):
            if os.path.isdir(datafile):
                test_datafile = os.path.join(datafile, DEFAULT_SPECWIZARD_FILENAME)
                if os.path.isfile(test_datafile):
                    datafile = test_datafile
                else:
                    raise FileNotFoundError(f"Directory was provided, but contained no file named \"{DEFAULT_SPECWIZARD_FILENAME}\".\nDirectory was: {datafile}")
            else:
                raise FileNotFoundError(f"No file exists at: {datafile}")

        target_spectrum = target_spectra[0] if len(target_spectra) == 1 else target_spectra[datafile_index]

        x_data.append([])
        flux_data.append([])
        labels.append([])
        colours.append([])
        linestyles.append([])
        alphas.append([])

        Console.print_verbose_info(f"Loading data from: {datafile}")

        data = SpecWizard_Data(datafile)

        Console.print_verbose_info(f"Selecting spectrum at index: {target_spectrum}")

        if (target_spectrum >= 0 and target_spectrum >= len(data)) or (target_spectrum < 0 and -target_spectrum > len(data)):
            raise IndexError(f"Unable to select sightline index {target_spectrum} as there are only {len(data)} spectra avalible.")

        if raw or raw_with_noise:
            Console.print_verbose_info("Loading raw fluxes...", end = "")

            x_data[datafile_index].append(data.wavelengths)
            if raw or (custom_noise and "use_flux_without_noise" in noise_settings[datafile_index]) or not data.noise_avalible:
                flux_data[datafile_index].append(data.get_flux(target_spectrum))
                labels[datafile_index].append("Raw Flux")
            else:
                # raw_with_noise
                if not custom_noise or len(noise_settings[datafile_index]) == 0:
                    flux_data[datafile_index].append(
                        data.get_noisey_flux(target_spectrum)
                    )
                elif "noise_override_profile" in noise_settings[datafile_index]:
                    flux_data[datafile_index].append(
                        data.get_flux_using_noise_profile(
                            noise_settings[datafile_index]["noise_override_profile"],
                            target_spectrum,
                            use_existing_random = data.noise_avalible
                        )[0]
                    )
                elif "noise_override_signal_to_noise_wavelength_limits" in noise_settings[datafile_index]:
                    flux_data[datafile_index].append(
                        data.get_flux_with_artificial_signal_to_noise(
                            noise_settings[datafile_index]["noise_override_signal_to_noise"],
                            noise_settings[datafile_index]["noise_override_min_noise"],
                            target_spectrum,
                            noise_settings[datafile_index]["noise_override_signal_to_noise_wavelength_limits"],
                            use_existing_random = data.noise_avalible
                        )[0]
                    )
                else:
                    flux_data[datafile_index].append(
                        data.get_flux_with_artificial_signal_to_noise(
                            noise_settings[datafile_index]["noise_override_signal_to_noise"],
                            noise_settings[datafile_index]["noise_override_min_noise"],
                            target_spectrum,
                            use_existing_random = data.noise_avalible
                        )[0]
                    )
                labels[datafile_index].append("Noisey Flux")
            colours[datafile_index].append("grey" if len(datafiles) == 1 else ("black", "blue", "orange")[datafile_index % 3])
            linestyles[datafile_index].append("-")
            alphas[datafile_index].append(1.0)

            print("done")
            Console.print_verbose_info(f"Got {len(x_data[datafile_index][-1])} pixels.")
            Console.print_debug(("Raw Flux:\n" if raw else "Raw Flux (with noise):\n") + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

        if use_RSODOST_field:
            if h1:
                Console.print_verbose_info("Loading H I strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data.wavelengths)
                flux_data[datafile_index].append(np.exp(-data.get_RSODOST(Ion.H_I, target_spectrum)))
                labels[datafile_index].append("H I")
                colours[datafile_index].append("black")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"H I (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("H I (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
            if c4:
                Console.print_verbose_info("Loading C IV strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data.wavelengths)
                flux_data[datafile_index].append(np.exp(-data.get_RSODOST(Ion.C_IV, target_spectrum)))
                labels[datafile_index].append("C IV")
                colours[datafile_index].append("blue")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"C IV (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("C IV  (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
            if si4:
                Console.print_verbose_info("Loading Si IV strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data.wavelengths)
                flux_data[datafile_index].append(np.exp(-data.get_RSODOST(Ion.Si_IV, target_spectrum)))
                labels[datafile_index].append("Si IV")
                colours[datafile_index].append("yellow")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"Si IV (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("Si IV  (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
            if o6:
                Console.print_verbose_info("Loading O VI strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data.wavelengths)
                flux_data[datafile_index].append(np.exp(-data.get_RSODOST(Ion.O_VI, target_spectrum)))
                labels[datafile_index].append("O VI")
                colours[datafile_index].append("red")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"O VI (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("O VI  (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

        else:

            h1_wavelength = None
            c4_wavelength = None
            si4_wavelength = None
            o6_wavelength = None

            h1_redshift = None
            c4_redshift = None
            si4_redshift = None
            o6_redshift = None

            uncorrected_hydrogen_uncorrected_h1_tau = None
            uncorrected_hydrogen_uncorrected_c4_tau = None # H I corrections have no affect on the defualt corrections for C IV
            uncorrected_hydrogen_uncorrected_si4_tau = None
            uncorrected_hydrogen_uncorrected_o6_tau = None # Why does this have no effect?

            uncorrected_hydrogen_corrected_h1_tau = None
            uncorrected_hydrogen_corrected_c4_tau = None # H I corrections have no affect on the defualt corrections for C IV
            uncorrected_hydrogen_corrected_si4_tau = None
            uncorrected_hydrogen_corrected_o6_tau = None # Why does this have no effect?

            corrected_hydrogen_uncorrected_h1_tau = None
            corrected_hydrogen_uncorrected_c4_tau = None
            corrected_hydrogen_uncorrected_si4_tau = None
            corrected_hydrogen_uncorrected_o6_tau = None

            corrected_h1_tau = None
            corrected_c4_tau = None
            corrected_si4_tau = None
            corrected_o6_tau = None

            if h1 or c4 or si4:
                uncorrected_h1_dla_masked_spectra = SpectrumCollection.from_specwizard(
                    data = data,
                    sightline_filter = [target_spectrum],
                    mask_dla = True,
                    noise_override_default_to_existing_random = data.noise_avalible,
                    **(noise_settings[datafile_index] if custom_noise else {})
                )
                uncorrected_h1_dla_masked_spectra.recover_h1(
                    identify_h1_contamination = False,
                    correct_h1_contamination = False
                )
                uncorrected_h1_dla_masked_spectrum_object = uncorrected_h1_dla_masked_spectra[0]

                corrected_h1_dla_masked_spectra = SpectrumCollection.from_specwizard(
                    data = data,
                    sightline_filter = [target_spectrum],
                    mask_dla = True,
                    noise_override_default_to_existing_random = data.noise_avalible,
                    **(noise_settings[datafile_index] if custom_noise else {})
                )
                corrected_h1_dla_masked_spectra.recover_h1(
                    identify_h1_contamination = True,
                    correct_h1_contamination = True
                )
                corrected_h1_dla_masked_spectrum_object = corrected_h1_dla_masked_spectra[0]

                if c4:
                    #TODO:python -m debugpy --listen 5678 --wait-for-client $(which pod-plot-specwizard-spectrum) -i Turner16_QSO_Q1317-0507 --c4 --wavelength -v -d
                    #breakpoint()#2479 6202.713041638517
                    uncorrected_h1_dla_masked_spectra.recover_c4(apply_recomended_corrections = True)
                    corrected_h1_dla_masked_spectra.recover_c4(apply_recomended_corrections = True)

                    c4_wavelength = corrected_h1_dla_masked_spectrum_object.c4_data.lambdaa
                    c4_redshift = corrected_h1_dla_masked_spectrum_object.c4_data.z

                    assert (uncorrected_h1_dla_masked_spectrum_object.c4_data.lambdaa == corrected_h1_dla_masked_spectrum_object.c4_data.lambdaa).all()
                    assert (uncorrected_h1_dla_masked_spectrum_object.c4_data.z == corrected_h1_dla_masked_spectrum_object.c4_data.z).all()

                    uncorrected_hydrogen_uncorrected_c4_tau = uncorrected_h1_dla_masked_spectrum_object.c4_data.tau
                    uncorrected_hydrogen_corrected_c4_tau = uncorrected_h1_dla_masked_spectrum_object.c4_data.tau_rec
                    corrected_hydrogen_uncorrected_c4_tau = corrected_h1_dla_masked_spectrum_object.c4_data.tau
                    corrected_c4_tau = corrected_h1_dla_masked_spectrum_object.c4_data.tau_rec
                    
                    if Settings.debug:
                        Console.print_debug(ArrayVisuliser.arrange(2, [
                            uncorrected_hydrogen_uncorrected_c4_tau, uncorrected_hydrogen_corrected_c4_tau,
                            corrected_hydrogen_uncorrected_c4_tau,   corrected_c4_tau
                                                                      ],
                                                                      [
                            "No H I, No C IV",   "No H I, With C IV",
                            "With H I, No C IV", "With H I, With C IV"
                                                                      ]).render())
                        # Ensure that H I corrections have no affect on C IV raw data
                        assert (uncorrected_hydrogen_uncorrected_c4_tau == corrected_hydrogen_uncorrected_c4_tau).all(), "H I correction had an affect on the recovery of tau_(C IV). This should not be the case with PodPy default recovery settings!"
                        # Ensure that H I corrections have no affect on C IV corrected data
                        assert (uncorrected_hydrogen_corrected_c4_tau   == corrected_c4_tau).all(),                      "H I correction had an affect on the recovery of tau_(C IV). This should not be the case with PodPy default recovery settings!"

                if si4:
                    uncorrected_h1_dla_masked_spectra.recover_si4(apply_recomended_corrections = True)
                    corrected_h1_dla_masked_spectra.recover_si4(apply_recomended_corrections = True)

                    si4_wavelength = corrected_h1_dla_masked_spectrum_object.si4_data.lambdaa
                    si4_redshift = corrected_h1_dla_masked_spectrum_object.si4_data.z

                    assert (uncorrected_h1_dla_masked_spectrum_object.si4_data.lambdaa == corrected_h1_dla_masked_spectrum_object.si4_data.lambdaa).all()
                    assert (uncorrected_h1_dla_masked_spectrum_object.si4_data.z == corrected_h1_dla_masked_spectrum_object.si4_data.z).all()

                    uncorrected_hydrogen_uncorrected_si4_tau = uncorrected_h1_dla_masked_spectrum_object.si4_data.tau
                    uncorrected_hydrogen_corrected_si4_tau = uncorrected_h1_dla_masked_spectrum_object.si4_data.tau_rec
                    corrected_hydrogen_uncorrected_si4_tau = corrected_h1_dla_masked_spectrum_object.si4_data.tau
                    corrected_si4_tau = corrected_h1_dla_masked_spectrum_object.si4_data.tau_rec
                    
                    if Settings.debug:
                        Console.print_debug(ArrayVisuliser.arrange(2, [
                            uncorrected_hydrogen_uncorrected_si4_tau, uncorrected_hydrogen_corrected_si4_tau,
                            corrected_hydrogen_uncorrected_si4_tau,   corrected_si4_tau
                                                                      ],
                                                                      [
                            "No H I, No Si IV",   "No H I, With Si IV",
                            "With H I, No Si IV", "With H I, With Si IV"
                                                                      ]).render())
                        # Ensure that H I corrections have no affect on Si IV raw data
                        assert (uncorrected_hydrogen_uncorrected_si4_tau == corrected_hydrogen_uncorrected_si4_tau).all(), "H I correction had an affect on the recovery of tau_(C IV). This should not be the case with PodPy default recovery settings!"

                if h1:
                    h1_wavelength = corrected_h1_dla_masked_spectrum_object.h1_data.lambdaa
                    h1_redshift = corrected_h1_dla_masked_spectrum_object.h1_data.z

                    uncorrected_hydrogen_uncorrected_h1_tau = uncorrected_h1_dla_masked_spectrum_object.h1_data.tau
                    uncorrected_hydrogen_corrected_h1_tau = uncorrected_h1_dla_masked_spectrum_object.h1_data.tau_rec
                    corrected_hydrogen_uncorrected_h1_tau = corrected_h1_dla_masked_spectrum_object.h1_data.tau
                    corrected_h1_tau = corrected_h1_dla_masked_spectrum_object.h1_data.tau_rec

                    if Settings.debug:
                        Console.print_debug(ArrayVisuliser.arrange(2, [
                            uncorrected_hydrogen_uncorrected_h1_tau, uncorrected_hydrogen_corrected_h1_tau,
                            corrected_hydrogen_uncorrected_h1_tau,   corrected_h1_tau
                                                                      ],
                                                                      [
                            "No H I, No ?",   "No H I, With ?",
                            "With H I, No ?", "With H I, With ?"
                                                                      ]).render())
                        Console.print_debug((uncorrected_hydrogen_uncorrected_h1_tau != uncorrected_hydrogen_corrected_h1_tau).sum(), uncorrected_hydrogen_uncorrected_h1_tau.shape)
                        ArrayVisuliser([uncorrected_hydrogen_uncorrected_h1_tau[uncorrected_hydrogen_uncorrected_h1_tau != uncorrected_hydrogen_corrected_h1_tau], uncorrected_hydrogen_corrected_h1_tau[uncorrected_hydrogen_uncorrected_h1_tau != uncorrected_hydrogen_corrected_h1_tau]]).print()
                        # Ensure that enabling H I recovery dosen't alter the raw data
                        assert (uncorrected_hydrogen_uncorrected_h1_tau == corrected_hydrogen_uncorrected_h1_tau).all(), "Raw H I data altered by enabling H I recovery."
                    if Settings.verbose or Settings.debug:
                        # Ensure that the recovered H I data is unaltered from the raw data when H I recovery is disabled
                        if not (uncorrected_hydrogen_uncorrected_h1_tau == uncorrected_hydrogen_corrected_h1_tau).all():
                            Console.print_warning("H I recovery caused differences even though recovery is disabled. Is this due to limiting to values between tau min and max.")

            if o6:
                uncorrected_h1_non_dla_masked_spectra = SpectrumCollection.from_specwizard(
                    data = data,
                    sightline_filter = [target_spectrum],
                    mask_dla = False,
                    noise_override_default_to_existing_random = data.noise_avalible,
                    **(noise_settings[datafile_index] if custom_noise else {})
                )
                uncorrected_h1_non_dla_masked_spectra.recover_h1(
                    identify_h1_contamination = False,
                    correct_h1_contamination = False
                )
                uncorrected_h1_non_dla_masked_spectrum_object = uncorrected_h1_non_dla_masked_spectra[0]

                corrected_h1_non_dla_masked_spectra = SpectrumCollection.from_specwizard(
                    data = data,
                    sightline_filter = [target_spectrum],
                    mask_dla = False,
                    noise_override_default_to_existing_random = data.noise_avalible,
                    **(noise_settings[datafile_index] if custom_noise else {})
                )
                corrected_h1_non_dla_masked_spectra.recover_h1(
                    identify_h1_contamination = True,
                    correct_h1_contamination = True
                )
                corrected_h1_non_dla_masked_spectrum_object = corrected_h1_non_dla_masked_spectra[0]

                uncorrected_h1_non_dla_masked_spectra.recover_o6(apply_recomended_corrections = True)
                corrected_h1_non_dla_masked_spectra.recover_o6(apply_recomended_corrections = True)

                o6_wavelength = corrected_h1_non_dla_masked_spectrum_object.o6_data.lambdaa
                o6_redshift = corrected_h1_non_dla_masked_spectrum_object.o6_data.z

                uncorrected_hydrogen_uncorrected_o6_tau = uncorrected_h1_non_dla_masked_spectrum_object.o6_data.tau
                uncorrected_hydrogen_corrected_o6_tau = uncorrected_h1_non_dla_masked_spectrum_object.o6_data.tau_rec
                corrected_hydrogen_uncorrected_o6_tau = corrected_h1_non_dla_masked_spectrum_object.o6_data.tau
                corrected_o6_tau = corrected_h1_non_dla_masked_spectrum_object.o6_data.tau_rec
                
                if Settings.debug:
                    Console.print_debug(ArrayVisuliser.arrange(2, [
                        uncorrected_hydrogen_uncorrected_o6_tau, uncorrected_hydrogen_corrected_o6_tau,
                        corrected_hydrogen_uncorrected_o6_tau,   corrected_o6_tau
                                                                    ],
                                                                    [
                        "No H I, No O VI",   "No H I, With O VI",
                        "With H I, No O VI", "With H I, With O VI"
                                                                    ]).render())
                    assert (uncorrected_hydrogen_uncorrected_o6_tau == corrected_hydrogen_uncorrected_o6_tau).all(), "H I correction had an affect on the recovery of tau_(O VI)."
                    assert (uncorrected_hydrogen_corrected_o6_tau   == corrected_o6_tau).all(),                      "H I correction had an affect on the recovery of tau_(O VI)."

            if h1:
                if compare_corrections:
                        x_data[datafile_index].extend([h1_wavelength if wavelength else h1_redshift if redshift else h1_redshift * speed_of_light_kms] * 4)
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"])
                        alphas[datafile_index].extend([1.0] * 4)

                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_h1_tau))
                        labels[datafile_index].append("H I (raw)")
                        linestyles[datafile_index].append("-")

                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_h1_tau))
                        labels[datafile_index].append("H I (corrected, corrected disabled)")
                        linestyles[datafile_index].append("--")

                        flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_h1_tau))
                        labels[datafile_index].append("H I (raw, correction enabled)")
                        linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_h1_tau))
                        labels[datafile_index].append("H I (corrected)")
                        linestyles[datafile_index].append(":")

                else:
                    x_data[datafile_index].append(h1_wavelength if wavelength else h1_redshift if redshift else h1_redshift * speed_of_light_kms)
                    colours[datafile_index].append("black")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_h1_tau)) # same as corrected_hydrogen_uncorrected_h1_tau
                        labels[datafile_index].append("Ly $\\alpha$")
                        Console.print_verbose_info(f"H I (uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("H I (uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_h1_tau))
                        labels[datafile_index].append("H I")
                        Console.print_verbose_info(f"H I (corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("H I (corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

            if c4:
                if compare_corrections:
                        x_data[datafile_index].extend([c4_wavelength if wavelength else c4_redshift if redshift else c4_redshift * speed_of_light_kms] * (4 if Settings.debug else 2))
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"] if Settings.debug else ["black", "orange"])
                        alphas[datafile_index].extend([1.0] * (4 if Settings.debug else 2))

                        #x_data[datafile_index].append(c4_wavelength if wavelength else c4_redshift if redshift else c4_redshift * speed_of_light_kms)
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_c4_tau))
                        labels[datafile_index].append("C IV (raw, without corrected H I)")
                        linestyles[datafile_index].append("-")

                        if Settings.debug:
                            flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_c4_tau))
                            labels[datafile_index].append("C IV (corrected, without corrected H I)")
                            linestyles[datafile_index].append("--")

                            flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_c4_tau))
                            labels[datafile_index].append("C IV (raw, with corrected H I)")
                            linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_c4_tau))
                        labels[datafile_index].append("C IV (corrected, with corrected H I)")
                        linestyles[datafile_index].append(":"if Settings.debug else "-")

                else:
                    x_data[datafile_index].append(c4_wavelength if wavelength else c4_redshift if redshift else c4_redshift * speed_of_light_kms)
                    colours[datafile_index].append("blue")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_c4_tau))
                        labels[datafile_index].append("C IV (raw, without corrected H I)")
                        Console.print_verbose_info(f"C IV (uncorrected: raw): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (uncorrected: raw):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif no_h1_corrections and not no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_c4_tau))
                        labels[datafile_index].append("C IV (corrected, without corrected H I)")
                        Console.print_verbose_info(f"C IV (corrected: H I uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (corrected: H I uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif not no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_c4_tau))
                        labels[datafile_index].append("C IV (raw, with corrected H I)")
                        Console.print_verbose_info(f"C IV (uncorrected: raw, H I corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (uncorrected: raw, H I corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_c4_tau))
                        labels[datafile_index].append("C IV (corrected, with corrected H I)")
                        Console.print_verbose_info(f"C IV (corrected: default): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (corrected: default):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

            if si4:
                if compare_corrections:
                        x_data[datafile_index].extend([si4_wavelength if wavelength else si4_redshift if redshift else si4_redshift * speed_of_light_kms] * (4 if Settings.debug else 2))
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"] if Settings.debug else ["black", "orange"])
                        alphas[datafile_index].extend([1.0] * (4 if Settings.debug else 2))

                        #x_data[datafile_index].append(si4_wavelength if wavelength else si4_redshift if redshift else si4_redshift * speed_of_light_kms)
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_si4_tau))
                        labels[datafile_index].append("Si IV (raw, without corrected H I)")
                        linestyles[datafile_index].append("-")

                        if Settings.debug:
                            flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_si4_tau))
                            labels[datafile_index].append("Si IV (corrected, without corrected H I)")
                            linestyles[datafile_index].append("--")

                            flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_si4_tau))
                            labels[datafile_index].append("Si IV (raw, with corrected H I)")
                            linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_si4_tau))
                        labels[datafile_index].append("Si IV (corrected, with corrected H I)")
                        linestyles[datafile_index].append(":"if Settings.debug else "-")

                else:
                    x_data[datafile_index].append(si4_wavelength if wavelength else si4_redshift if redshift else si4_redshift * speed_of_light_kms)
                    colours[datafile_index].append("yellow")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_si4_tau))
                        labels[datafile_index].append("Si IV (raw, without corrected H I)")
                        Console.print_verbose_info(f"Si IV (uncorrected: raw): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (uncorrected: raw):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif no_h1_corrections and not no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_si4_tau))
                        labels[datafile_index].append("Si IV (corrected, without corrected H I)")
                        Console.print_verbose_info(f"Si IV (corrected: H I uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (corrected: H I uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif not no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_si4_tau))
                        labels[datafile_index].append("Si IV (raw, with corrected H I)")
                        Console.print_verbose_info(f"Si IV (uncorrected: raw, H I corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (uncorrected: raw, H I corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_si4_tau))
                        labels[datafile_index].append("Si IV (corrected, with corrected H I)")
                        Console.print_verbose_info(f"Si IV (corrected: default): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (corrected: default):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

            if o6:
                if compare_corrections:
                        x_data[datafile_index].extend([o6_wavelength if wavelength else o6_redshift if redshift else o6_redshift * speed_of_light_kms] * (4 if Settings.debug else 2))
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"] if Settings.debug else ["black", "orange"])
                        alphas[datafile_index].extend([1.0] * (4 if Settings.debug else 2))

                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_o6_tau))
                        labels[datafile_index].append("O VI (raw, without corrected H I)")
                        linestyles[datafile_index].append("-")

                        if Settings.debug:
                            flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_o6_tau))
                            labels[datafile_index].append("O VI (corrected, without corrected H I)")
                            linestyles[datafile_index].append("--")

                            flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_o6_tau))
                            labels[datafile_index].append("O VI (raw, with corrected H I)")
                            linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_o6_tau))
                        labels[datafile_index].append("O VI (corrected, with corrected H I)")
                        linestyles[datafile_index].append(":" if Settings.debug else "-")

                else:
                    x_data[datafile_index].append(o6_wavelength if wavelength else o6_redshift if redshift else o6_redshift * speed_of_light_kms)
                    colours[datafile_index].append("red")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_o6_tau))
                        labels[datafile_index].append("O VI (raw, without corrected H I)")
                        Console.print_verbose_info(f"O VI (uncorrected: raw): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (uncorrected: raw):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif no_h1_corrections and not no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_o6_tau))
                        labels[datafile_index].append("O VI (corrected, without corrected H I)")
                        Console.print_verbose_info(f"O VI (corrected: H I uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (corrected: H I uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif not no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_o6_tau))
                        labels[datafile_index].append("O VI (raw, with corrected H I)")
                        Console.print_verbose_info(f"O VI (uncorrected: raw, H I corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (uncorrected: raw, H I corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_o6_tau))
                        labels[datafile_index].append("O VI (corrected, with corrected H I)")
                        Console.print_verbose_info(f"O VI (corrected: default): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (corrected: default):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

        '''
        data = h5.File(datafile)
        first_spec_num = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["first_specnum"])
        n_spectra = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["NumberOfSpectra"])

        Console.print_verbose_info(f"Selecting spectrum at index: {target_spectrum}")

        if (target_spectrum >= 0 and target_spectrum >= n_spectra) or (target_spectrum < 0 and -target_spectrum > n_spectra):
            raise IndexError(f"Unable to select sightline index {target_spectrum} as there are only {n_spectra} spectra avalible.")
        selected_spectrum_data = data[f"Spectrum{first_spec_num + target_spectrum}"]

        if raw or raw_with_noise:
            Console.print_verbose_info("Loading raw fluxes...", end = "")

            x_data[datafile_index].append(data["Wavelength_Ang"][:])
            if raw:
                flux_data[datafile_index].append(selected_spectrum_data["Flux"][:])
            elif raw_with_noise:
                flux_data[datafile_index].append(
                    selected_spectrum_data["Flux"][:] + (selected_spectrum_data["Noise_Sigma"][:] * selected_spectrum_data["Gaussian_deviate"][:])
                )
            else: raise RuntimeError("This isn't possible! Please report!")
            labels[datafile_index].append("Raw Flux")
            colours[datafile_index].append("grey")
            linestyles[datafile_index].append("-")
            alphas[datafile_index].append(1.0)

            print("done")
            Console.print_verbose_info(f"Got {len(x_data[datafile_index][-1])} pixels.")
            Console.print_debug(("Raw Flux:\n" if raw else "Raw Flux (with noise):\n") + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

        if not use_RSODOST_field:

            h1_wavelength = None
            c4_wavelength = None
            si4_wavelength = None
            o6_wavelength = None

            h1_redshift = None
            c4_redshift = None
            si4_redshift = None
            o6_redshift = None

            uncorrected_hydrogen_uncorrected_h1_tau = None
            uncorrected_hydrogen_uncorrected_c4_tau = None # H I corrections have no affect on the defualt corrections for C IV
            uncorrected_hydrogen_uncorrected_si4_tau = None
            uncorrected_hydrogen_uncorrected_o6_tau = None # Why does this have no effect?

            uncorrected_hydrogen_corrected_h1_tau = None
            uncorrected_hydrogen_corrected_c4_tau = None # H I corrections have no affect on the defualt corrections for C IV
            uncorrected_hydrogen_corrected_si4_tau = None
            uncorrected_hydrogen_corrected_o6_tau = None # Why does this have no effect?

            corrected_hydrogen_uncorrected_h1_tau = None
            corrected_hydrogen_uncorrected_c4_tau = None
            corrected_hydrogen_uncorrected_si4_tau = None
            corrected_hydrogen_uncorrected_o6_tau = None

            corrected_h1_tau = None
            corrected_c4_tau = None
            corrected_si4_tau = None
            corrected_o6_tau = None

            if h1 or c4 or si4:
                uncorrected_h1_dla_masked_spectrum_object = from_SpecWizard(
                    filepath = datafile,
                    sightline_filter = [target_spectrum],
                    identify_h1_contamination = False,
                    correct_h1_contamination = False,
                    mask_dla = True
                )[0]
                corrected_h1_dla_masked_spectrum_object = from_SpecWizard(
                   filepath = datafile,
                   sightline_filter = [target_spectrum],
                   identify_h1_contamination = True,
                   correct_h1_contamination = True,
                   mask_dla = True
                )[0]

                if c4:
                    #TODO:python -m debugpy --listen 5678 --wait-for-client $(which pod-plot-specwizard-spectrum) -i Turner16_QSO_Q1317-0507 --c4 --wavelength -v -d
                    #breakpoint()#2479 6202.713041638517
                    recover_c4(uncorrected_h1_dla_masked_spectrum_object, apply_recomended_corrections = True)
                    recover_c4(corrected_h1_dla_masked_spectrum_object, apply_recomended_corrections = True)

                    c4_wavelength = corrected_h1_dla_masked_spectrum_object.c4_data.lambdaa
                    c4_redshift = corrected_h1_dla_masked_spectrum_object.c4_data.z

                    assert (uncorrected_h1_dla_masked_spectrum_object.c4_data.lambdaa == corrected_h1_dla_masked_spectrum_object.c4_data.lambdaa).all()
                    assert (uncorrected_h1_dla_masked_spectrum_object.c4_data.z == corrected_h1_dla_masked_spectrum_object.c4_data.z).all()

                    uncorrected_hydrogen_uncorrected_c4_tau = uncorrected_h1_dla_masked_spectrum_object.c4_data.tau
                    uncorrected_hydrogen_corrected_c4_tau = uncorrected_h1_dla_masked_spectrum_object.c4_data.tau_rec
                    corrected_hydrogen_uncorrected_c4_tau = corrected_h1_dla_masked_spectrum_object.c4_data.tau
                    corrected_c4_tau = corrected_h1_dla_masked_spectrum_object.c4_data.tau_rec
                    
                    if Settings.debug:
                        Console.print_debug(ArrayVisuliser.arrange(2, [
                            uncorrected_hydrogen_uncorrected_c4_tau, uncorrected_hydrogen_corrected_c4_tau,
                            corrected_hydrogen_uncorrected_c4_tau,   corrected_c4_tau
                                                                      ],
                                                                      [
                            "No H I, No C IV",   "No H I, With C IV",
                            "With H I, No C IV", "With H I, With C IV"
                                                                      ]).render())
                        # Ensure that H I corrections have no affect on C IV raw data
                        assert (uncorrected_hydrogen_uncorrected_c4_tau == corrected_hydrogen_uncorrected_c4_tau).all(), "H I correction had an affect on the recovery of tau_(C IV). This should not be the case with PodPy default recovery settings!"
                        # Ensure that H I corrections have no affect on C IV corrected data
                        assert (uncorrected_hydrogen_corrected_c4_tau   == corrected_c4_tau).all(),                      "H I correction had an affect on the recovery of tau_(C IV). This should not be the case with PodPy default recovery settings!"

                if si4:
                    recover_si4(uncorrected_h1_dla_masked_spectrum_object, apply_recomended_corrections = True)
                    recover_si4(corrected_h1_dla_masked_spectrum_object, apply_recomended_corrections = True)

                    si4_wavelength = corrected_h1_dla_masked_spectrum_object.si4_data.lambdaa
                    si4_redshift = corrected_h1_dla_masked_spectrum_object.si4_data.z

                    assert (uncorrected_h1_dla_masked_spectrum_object.si4_data.lambdaa == corrected_h1_dla_masked_spectrum_object.si4_data.lambdaa).all()
                    assert (uncorrected_h1_dla_masked_spectrum_object.si4_data.z == corrected_h1_dla_masked_spectrum_object.si4_data.z).all()

                    uncorrected_hydrogen_uncorrected_si4_tau = uncorrected_h1_dla_masked_spectrum_object.si4_data.tau
                    uncorrected_hydrogen_corrected_si4_tau = uncorrected_h1_dla_masked_spectrum_object.si4_data.tau_rec
                    corrected_hydrogen_uncorrected_si4_tau = corrected_h1_dla_masked_spectrum_object.si4_data.tau
                    corrected_si4_tau = corrected_h1_dla_masked_spectrum_object.si4_data.tau_rec
                    
                    if Settings.debug:
                        Console.print_debug(ArrayVisuliser.arrange(2, [
                            uncorrected_hydrogen_uncorrected_si4_tau, uncorrected_hydrogen_corrected_si4_tau,
                            corrected_hydrogen_uncorrected_si4_tau,   corrected_si4_tau
                                                                      ],
                                                                      [
                            "No H I, No Si IV",   "No H I, With Si IV",
                            "With H I, No Si IV", "With H I, With Si IV"
                                                                      ]).render())
                        # Ensure that H I corrections have no affect on Si IV raw data
                        assert (uncorrected_hydrogen_uncorrected_si4_tau == corrected_hydrogen_uncorrected_si4_tau).all(), "H I correction had an affect on the recovery of tau_(C IV). This should not be the case with PodPy default recovery settings!"

                if h1:
                    h1_wavelength = corrected_h1_dla_masked_spectrum_object.h1_data.lambdaa
                    h1_redshift = corrected_h1_dla_masked_spectrum_object.h1_data.z

                    uncorrected_hydrogen_uncorrected_h1_tau = uncorrected_h1_dla_masked_spectrum_object.h1_data.tau
                    uncorrected_hydrogen_corrected_h1_tau = uncorrected_h1_dla_masked_spectrum_object.h1_data.tau_rec
                    corrected_hydrogen_uncorrected_h1_tau = corrected_h1_dla_masked_spectrum_object.h1_data.tau
                    corrected_h1_tau = corrected_h1_dla_masked_spectrum_object.h1_data.tau_rec

                    if Settings.debug:
                        Console.print_debug(ArrayVisuliser.arrange(2, [
                            uncorrected_hydrogen_uncorrected_h1_tau, uncorrected_hydrogen_corrected_h1_tau,
                            corrected_hydrogen_uncorrected_h1_tau,   corrected_h1_tau
                                                                      ],
                                                                      [
                            "No H I, No ?",   "No H I, With ?",
                            "With H I, No ?", "With H I, With ?"
                                                                      ]).render())
                        Console.print_debug((uncorrected_hydrogen_uncorrected_h1_tau != uncorrected_hydrogen_corrected_h1_tau).sum(), uncorrected_hydrogen_uncorrected_h1_tau.shape)
                        ArrayVisuliser([uncorrected_hydrogen_uncorrected_h1_tau[uncorrected_hydrogen_uncorrected_h1_tau != uncorrected_hydrogen_corrected_h1_tau], uncorrected_hydrogen_corrected_h1_tau[uncorrected_hydrogen_uncorrected_h1_tau != uncorrected_hydrogen_corrected_h1_tau]]).print()
                        # Ensure that enabling H I recovery dosen't alter the raw data
                        assert (uncorrected_hydrogen_uncorrected_h1_tau == corrected_hydrogen_uncorrected_h1_tau).all(), "Raw H I data altered by enabling H I recovery."
                    if Settings.verbose or Settings.debug:
                        # Ensure that the recovered H I data is unaltered from the raw data when H I recovery is disabled
                        if not (uncorrected_hydrogen_uncorrected_h1_tau == uncorrected_hydrogen_corrected_h1_tau).all():
                            Console.print_warning("H I recovery caused differences even though recovery is disabled. Is this due to limiting to values between tau min and max.")

            if o6:
                uncorrected_h1_non_dla_masked_spectrum_object = from_SpecWizard(
                    filepath = datafile,
                    sightline_filter = [target_spectrum],
                    identify_h1_contamination = False,
                    correct_h1_contamination = False,
                    mask_dla = False
                )[0]
                corrected_h1_non_dla_masked_spectrum_object = from_SpecWizard(
                    filepath = datafile,
                    sightline_filter = [target_spectrum],
                    identify_h1_contamination = True,
                    correct_h1_contamination = True,
                    mask_dla = False
                )[0]

                recover_o6(uncorrected_h1_non_dla_masked_spectrum_object, apply_recomended_corrections = True)
                recover_o6(corrected_h1_non_dla_masked_spectrum_object, apply_recomended_corrections = True)

                o6_wavelength = corrected_h1_non_dla_masked_spectrum_object.o6_data.lambdaa
                o6_redshift = corrected_h1_non_dla_masked_spectrum_object.o6_data.z

                uncorrected_hydrogen_uncorrected_o6_tau = uncorrected_h1_non_dla_masked_spectrum_object.o6_data.tau
                uncorrected_hydrogen_corrected_o6_tau = uncorrected_h1_non_dla_masked_spectrum_object.o6_data.tau_rec
                corrected_hydrogen_uncorrected_o6_tau = corrected_h1_non_dla_masked_spectrum_object.o6_data.tau
                corrected_o6_tau = corrected_h1_non_dla_masked_spectrum_object.o6_data.tau_rec
                
                if Settings.debug:
                    Console.print_debug(ArrayVisuliser.arrange(2, [
                        uncorrected_hydrogen_uncorrected_o6_tau, uncorrected_hydrogen_corrected_o6_tau,
                        corrected_hydrogen_uncorrected_o6_tau,   corrected_o6_tau
                                                                    ],
                                                                    [
                        "No H I, No O VI",   "No H I, With O VI",
                        "With H I, No O VI", "With H I, With O VI"
                                                                    ]).render())
                    assert (uncorrected_hydrogen_uncorrected_o6_tau == corrected_hydrogen_uncorrected_o6_tau).all(), "H I correction had an affect on the recovery of tau_(O VI)."
                    assert (uncorrected_hydrogen_corrected_o6_tau   == corrected_o6_tau).all(),                      "H I correction had an affect on the recovery of tau_(O VI)."

            if h1:
                if compare_corrections:
                        x_data[datafile_index].extend([h1_wavelength if wavelength else h1_redshift if redshift else h1_redshift * speed_of_light_kms] * 4)
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"])
                        alphas[datafile_index].extend([1.0] * 4)

                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_h1_tau))
                        labels[datafile_index].append("H I (raw)")
                        linestyles[datafile_index].append("-")

                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_h1_tau))
                        labels[datafile_index].append("H I (corrected, corrected disabled)")
                        linestyles[datafile_index].append("--")

                        flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_h1_tau))
                        labels[datafile_index].append("H I (raw, correction enabled)")
                        linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_h1_tau))
                        labels[datafile_index].append("H I (corrected)")
                        linestyles[datafile_index].append(":")

                else:
                    x_data[datafile_index].append(h1_wavelength if wavelength else h1_redshift if redshift else h1_redshift * speed_of_light_kms)
                    colours[datafile_index].append("black")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_h1_tau)) # same as corrected_hydrogen_uncorrected_h1_tau
                        labels[datafile_index].append("Ly $\\alpha$")
                        Console.print_verbose_info(f"H I (uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("H I (uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_h1_tau))
                        labels[datafile_index].append("H I")
                        Console.print_verbose_info(f"H I (corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("H I (corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

            if c4:
                if compare_corrections:
                        x_data[datafile_index].extend([c4_wavelength if wavelength else c4_redshift if redshift else c4_redshift * speed_of_light_kms] * (4 if Settings.debug else 2))
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"] if Settings.debug else ["black", "orange"])
                        alphas[datafile_index].extend([1.0] * (4 if Settings.debug else 2))

                        #x_data[datafile_index].append(c4_wavelength if wavelength else c4_redshift if redshift else c4_redshift * speed_of_light_kms)
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_c4_tau))
                        labels[datafile_index].append("C IV (raw, without corrected H I)")
                        linestyles[datafile_index].append("-")

                        if Settings.debug:
                            flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_c4_tau))
                            labels[datafile_index].append("C IV (corrected, without corrected H I)")
                            linestyles[datafile_index].append("--")

                            flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_c4_tau))
                            labels[datafile_index].append("C IV (raw, with corrected H I)")
                            linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_c4_tau))
                        labels[datafile_index].append("C IV (corrected, with corrected H I)")
                        linestyles[datafile_index].append(":"if Settings.debug else "-")

                else:
                    x_data[datafile_index].append(c4_wavelength if wavelength else c4_redshift if redshift else c4_redshift * speed_of_light_kms)
                    colours[datafile_index].append("blue")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_c4_tau))
                        labels[datafile_index].append("C IV (raw, without corrected H I)")
                        Console.print_verbose_info(f"C IV (uncorrected: raw): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (uncorrected: raw):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif no_h1_corrections and not no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_c4_tau))
                        labels[datafile_index].append("C IV (corrected, without corrected H I)")
                        Console.print_verbose_info(f"C IV (corrected: H I uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (corrected: H I uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif not no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_c4_tau))
                        labels[datafile_index].append("C IV (raw, with corrected H I)")
                        Console.print_verbose_info(f"C IV (uncorrected: raw, H I corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (uncorrected: raw, H I corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_c4_tau))
                        labels[datafile_index].append("C IV (corrected, with corrected H I)")
                        Console.print_verbose_info(f"C IV (corrected: default): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("C IV (corrected: default):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

            if si4:
                if compare_corrections:
                        x_data[datafile_index].extend([si4_wavelength if wavelength else si4_redshift if redshift else si4_redshift * speed_of_light_kms] * (4 if Settings.debug else 2))
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"] if Settings.debug else ["black", "orange"])
                        alphas[datafile_index].extend([1.0] * (4 if Settings.debug else 2))

                        #x_data[datafile_index].append(si4_wavelength if wavelength else si4_redshift if redshift else si4_redshift * speed_of_light_kms)
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_si4_tau))
                        labels[datafile_index].append("Si IV (raw, without corrected H I)")
                        linestyles[datafile_index].append("-")

                        if Settings.debug:
                            flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_si4_tau))
                            labels[datafile_index].append("Si IV (corrected, without corrected H I)")
                            linestyles[datafile_index].append("--")

                            flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_si4_tau))
                            labels[datafile_index].append("Si IV (raw, with corrected H I)")
                            linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_si4_tau))
                        labels[datafile_index].append("Si IV (corrected, with corrected H I)")
                        linestyles[datafile_index].append(":"if Settings.debug else "-")

                else:
                    x_data[datafile_index].append(si4_wavelength if wavelength else si4_redshift if redshift else si4_redshift * speed_of_light_kms)
                    colours[datafile_index].append("yellow")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_si4_tau))
                        labels[datafile_index].append("Si IV (raw, without corrected H I)")
                        Console.print_verbose_info(f"Si IV (uncorrected: raw): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (uncorrected: raw):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif no_h1_corrections and not no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_si4_tau))
                        labels[datafile_index].append("Si IV (corrected, without corrected H I)")
                        Console.print_verbose_info(f"Si IV (corrected: H I uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (corrected: H I uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif not no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_si4_tau))
                        labels[datafile_index].append("Si IV (raw, with corrected H I)")
                        Console.print_verbose_info(f"Si IV (uncorrected: raw, H I corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (uncorrected: raw, H I corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_si4_tau))
                        labels[datafile_index].append("Si IV (corrected, with corrected H I)")
                        Console.print_verbose_info(f"Si IV (corrected: default): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("Si IV (corrected: default):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

            if o6:
                if compare_corrections:
                        x_data[datafile_index].extend([o6_wavelength if wavelength else o6_redshift if redshift else o6_redshift * speed_of_light_kms] * (4 if Settings.debug else 2))
                        colours[datafile_index].extend(["black", "blue", "yellow", "orange"] if Settings.debug else ["black", "orange"])
                        alphas[datafile_index].extend([1.0] * (4 if Settings.debug else 2))

                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_o6_tau))
                        labels[datafile_index].append("O VI (raw, without corrected H I)")
                        linestyles[datafile_index].append("-")

                        if Settings.debug:
                            flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_o6_tau))
                            labels[datafile_index].append("O VI (corrected, without corrected H I)")
                            linestyles[datafile_index].append("--")

                            flux_data[datafile_index].append(np.exp(-10**corrected_hydrogen_uncorrected_o6_tau))
                            labels[datafile_index].append("O VI (raw, with corrected H I)")
                            linestyles[datafile_index].append("-.")

                        flux_data[datafile_index].append(np.exp(-10**corrected_o6_tau))
                        labels[datafile_index].append("O VI (corrected, with corrected H I)")
                        linestyles[datafile_index].append(":" if Settings.debug else "-")

                else:
                    x_data[datafile_index].append(o6_wavelength if wavelength else o6_redshift if redshift else o6_redshift * speed_of_light_kms)
                    colours[datafile_index].append("red")
                    linestyles[datafile_index].append("-")
                    alphas[datafile_index].append(1.0)
                    
                    if no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_uncorrected_o6_tau))
                        labels[datafile_index].append("O VI (raw, without corrected H I)")
                        Console.print_verbose_info(f"O VI (uncorrected: raw): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (uncorrected: raw):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif no_h1_corrections and not no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_o6_tau))
                        labels[datafile_index].append("O VI (corrected, without corrected H I)")
                        Console.print_verbose_info(f"O VI (corrected: H I uncorrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (corrected: H I uncorrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    elif not no_h1_corrections and no_metal_corrections:
                        flux_data[datafile_index].append(np.exp(-10**uncorrected_hydrogen_corrected_o6_tau))
                        labels[datafile_index].append("O VI (raw, with corrected H I)")
                        Console.print_verbose_info(f"O VI (uncorrected: raw, H I corrected): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (uncorrected: raw, H I corrected):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
                    else:
                        flux_data[datafile_index].append(np.exp(-10**corrected_o6_tau))
                        labels[datafile_index].append("O VI (corrected, with corrected H I)")
                        Console.print_verbose_info(f"O VI (corrected: default): got {len(x_data[datafile_index][-1])} pixels.")
                        if Settings.debug: Console.print_debug("O VI (corrected: default):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))

        else:
            if h1:
                Console.print_verbose_info("Loading H I strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data["Wavelength_Ang"][:])
                flux_data[datafile_index].append(np.exp(-selected_spectrum_data["h1/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
                labels[datafile_index].append("H I")
                colours[datafile_index].append("black")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"H I (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("H I (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
            if c4:
                Console.print_verbose_info("Loading C IV strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data["Wavelength_Ang"][:])
                flux_data[datafile_index].append(np.exp(-selected_spectrum_data["c4/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
                labels[datafile_index].append("C IV")
                colours[datafile_index].append("blue")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"C IV (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("C IV  (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
            if o6:
                Console.print_verbose_info("Loading O VI strongest transition opticaldepths...", end = "")
                x_data[datafile_index].append(data["Wavelength_Ang"][:])
                flux_data[datafile_index].append(np.exp(-selected_spectrum_data["o6/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
                labels[datafile_index].append("O VI")
                colours[datafile_index].append("red")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
                print("done")
                Console.print_verbose_info(f"O VI (RSODOST): got {len(x_data[datafile_index][-1])} pixels.")
                Console.print_debug("O VI  (RSODOST):\n" + ArrayVisuliser([x_data[datafile_index][-1], flux_data[datafile_index][-1]], ["Wavelengths", "Fluxes"]).render(show_plus_minus_1 = True))
        '''
                
    # Plot results

    if len(datafiles) == 1:
        figure, axis = plotting.spectrum(
            pixel_x_values = x_data[0],
            pixel_fluxes = flux_data[0],
            spectrum_type = plotting.PlottedSpectrumType.wavelength if wavelength else plotting.PlottedSpectrumType.velocity if velocity else plotting.PlottedSpectrumType.redshift,
            dataset_labels = labels[0],
            dataset_colours = colours[0],
            dataset_linestyles = linestyles[0],
            dataset_alphas = alphas[0],
            x_min = x_min,
            x_max = x_max,
            y_min = y_min,
            y_max = y_max,
            title = title,
            show = output_file is None,
            figure_creation_kwargs = { "layout": "constrained" }
        )

    else:
        avalible_line_styles = ("-", "--", "-.", ":")
        figure, axis = plotting.spectrum(
            pixel_x_values = [x_data[datafile_index][0] for datafile_index in range(len(datafiles))],
            pixel_fluxes = [flux_data[datafile_index][0] for datafile_index in range(len(datafiles))],
            spectrum_type = plotting.PlottedSpectrumType.wavelength if wavelength else plotting.PlottedSpectrumType.velocity if velocity else plotting.PlottedSpectrumType.redshift,
            dataset_labels = datafile_names,
            dataset_colours = [colours[datafile_index][0] for datafile_index in range(len(datafiles))],
            dataset_linestyles = [avalible_line_styles[datafile_index % len(avalible_line_styles)] for datafile_index in range(len(datafiles))],
            dataset_alphas = [alphas[datafile_index][0] for datafile_index in range(len(datafiles))],
            x_min = x_min,
            x_max = x_max,
            y_min = y_min,
            y_max = y_max,
            title = title,
            show = output_file is None,
            figure_creation_kwargs = { "layout": "constrained" }
        )

    # Decide what to do with the plot

    if output_file is not None:
        figure.savefig(output_file)
