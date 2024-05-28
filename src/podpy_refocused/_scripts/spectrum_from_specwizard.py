# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
import numpy as np
from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper
from typing import Union, List
from matplotlib import pyplot as plt
import h5py as h5
import datetime

from .._Spectrum import from_SpecWizard, fit_continuum, recover_c4, recover_o6
from .._TauBinned import bin_combined_pixels_from_SpecWizard, BinnedOpticalDepthResults
from .. import plotting
from .._universe import c as speed_of_light_kms

def __main(
           datafiles: List[str],
           datafile_names: Union[List[str], None],
           raw: bool,
           h1: bool,
           c4: bool,
           o6: bool,
           target_spectra: List[int],
           wavelength: bool,
           velocity: bool,
           redshift: bool,
           x_min: Union[float, None],
           x_max: Union[float, None],
           no_h1_corrections: bool,
           no_metal_corrections: bool,
           compare_corrections: bool,
           use_RSODOST_field: bool,
           title: Union[str, None],
           output_file: Union[str, None]
        ) -> None:
    
    if not (raw or h1 or c4 or o6):
        Console.print_error("Target ion(s) must be specified!")
        Console.print_info("Terminating...")
        return

    if not (wavelength or velocity or redshift):
        Console.print_error("Spectrum X-axis type must be specified! (options are '--wavelength', '--velocity' or '--redshift')")
        Console.print_info("Terminating...")
        return
    
    if len(datafiles) > 1 and sum([raw, h1, c4, o6]) > 1:
        Console.print_error("Only one ion can be specified when comparing different files!")
        Console.print_info("Terminating...")
    
    if len(datafiles) > 1 and len(target_spectra) != 1 and len(target_spectra) != len(datafiles):
        Console.print_error("Invalid number of spectrum indexes.\nEither just a single index should be specified, or a number of indexes matching the number of data files.")
        Console.print_info("Terminating...")

    if len(datafiles) > 1 and datafile_names is not None and len(datafile_names) != len(datafiles):
        Console.print_error("Invalid number of data file names.\nIf specified, must be a number of elements matching the number of data files.")
        Console.print_info("Terminating...")
    
    if datafile_names is None:
        datafile_names = [f"File #{n}" for n in range(1, 1 + len(datafiles))]
    
    # Load and process data

    x_data = []
    flux_data = []
    labels = []
    colours = []
    linestyles = []
    alphas = []

    for datafile_index in range(len(datafiles)):
        datafile = datafiles[datafile_index]
        target_spectrum = target_spectra[0] if len(target_spectra) == 1 else target_spectra[datafile_index]

        x_data.append([])
        flux_data.append([])
        labels.append([])
        colours.append([])
        linestyles.append([])
        alphas.append([])

        data = h5.File(datafile)
        first_spec_num = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["first_specnum"])
        n_spectra = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["NumberOfSpectra"])
        if (target_spectrum >= 0 and target_spectrum >= n_spectra) or (target_spectrum < 0 and -target_spectrum > n_spectra):
            raise IndexError(f"Unable to select sightline index {target_spectrum} as there are only {n_spectra} spectra avalible.")
        selected_spectrum_data = data[f"Spectrum{first_spec_num + target_spectrum}"]

        if raw:
            x_data[datafile_index].append(data["Wavelength_Ang"][:])
            flux_data[datafile_index].append(selected_spectrum_data["Flux"][:])
            labels[datafile_index].append("Raw Flux")
            colours[datafile_index].append("grey")
            linestyles[datafile_index].append("-")
            alphas[datafile_index].append(1.0)

        if not use_RSODOST_field:

            uncorrected_h1_wavelength = None
            uncorrected_c4_wavelength = None
            uncorrected_o6_wavelength = None
            uncorrected_h1_redshift = None
            uncorrected_c4_redshift = None
            uncorrected_o6_redshift = None

            uncorrected_h1_tau = None
            uncorrected_c4_tau = None
            uncorrected_o6_tau = None

            corrected_h1_wavelength = None
            corrected_c4_wavelength = None
            corrected_o6_wavelength = None
            corrected_h1_redshift = None
            corrected_c4_redshift = None
            corrected_o6_redshift = None

            corrected_h1_tau = None
            corrected_c4_tau = None
            corrected_o6_tau = None

            if h1 or c4:
                dla_masked_spectrum_object = from_SpecWizard(
                                                                filepath = datafile,
                                                                sightline_filter = [target_spectrum],
                                                                identify_h1_contamination = compare_corrections or not no_h1_corrections,
                                                                correct_h1_contamination = compare_corrections or not no_h1_corrections,
                                                                mask_dla = True
                                                            )[0]
                if c4:
                    if compare_corrections or not no_metal_corrections:
                        recover_c4(dla_masked_spectrum_object, apply_recomended_corrections = True)
                        corrected_c4_wavelength = dla_masked_spectrum_object.c4.lambdaa
                        corrected_c4_redshift = dla_masked_spectrum_object.c4.z
                        corrected_c4_tau = dla_masked_spectrum_object.c4.tau_rec
                    if compare_corrections or no_metal_corrections:
                        recover_c4(dla_masked_spectrum_object, apply_recomended_corrections = False)
                        uncorrected_c4_wavelength = dla_masked_spectrum_object.c4.lambdaa
                        uncorrected_c4_redshift = dla_masked_spectrum_object.c4.z
                        uncorrected_c4_tau = dla_masked_spectrum_object.c4.tau_rec
                
                if h1:
                    if compare_corrections or not no_h1_corrections:
                        corrected_h1_wavelength = dla_masked_spectrum_object.h1.lambdaa
                        corrected_h1_redshift = dla_masked_spectrum_object.h1.z
                        corrected_h1_tau = dla_masked_spectrum_object.h1.tau_rec
                        if compare_corrections:
                            # Compute the uncorrected H I optical depths
                            uncorrected_h1_spectrum_object = from_SpecWizard(
                                                                            filepath = datafile,
                                                                            sightline_filter = [target_spectrum],
                                                                            identify_h1_contamination = False,
                                                                            correct_h1_contamination = False,
                                                                            mask_dla = True
                                                                        )[0]
                            uncorrected_h1_wavelength = uncorrected_h1_spectrum_object.h1.lambdaa
                            uncorrected_h1_redshift = uncorrected_h1_spectrum_object.h1.z
                            uncorrected_h1_tau = uncorrected_h1_spectrum_object.h1.tau_rec
                    else:
                        uncorrected_h1_wavelength = dla_masked_spectrum_object.h1.lambdaa
                        uncorrected_h1_redshift = dla_masked_spectrum_object.h1.z
                        uncorrected_h1_tau = dla_masked_spectrum_object.h1.tau_rec

            if o6:
                non_dla_masked_spectrum_object = from_SpecWizard(
                                    filepath = datafile,
                                    sightline_filter = [target_spectrum],
                                    identify_h1_contamination = not no_h1_corrections,
                                    correct_h1_contamination = not no_h1_corrections,
                                    mask_dla = False
                                )[0]
                if compare_corrections or not no_metal_corrections:
                    recover_o6(non_dla_masked_spectrum_object, apply_recomended_corrections = True)
                    corrected_o6_wavelength = non_dla_masked_spectrum_object.o6.lambdaa
                    corrected_o6_redshift = non_dla_masked_spectrum_object.o6.z
                    corrected_o6_tau = non_dla_masked_spectrum_object.o6.tau_rec
                if compare_corrections or no_metal_corrections:
                    recover_o6(non_dla_masked_spectrum_object, apply_recomended_corrections = False)
                    uncorrected_o6_wavelength = non_dla_masked_spectrum_object.o6.lambdaa
                    uncorrected_o6_redshift = non_dla_masked_spectrum_object.o6.z
                    uncorrected_o6_tau = non_dla_masked_spectrum_object.o6.tau_rec

            if corrected_h1_tau is not None:
                x_data[datafile_index].append(corrected_h1_wavelength if wavelength else corrected_h1_redshift if redshift else corrected_h1_redshift * speed_of_light_kms)
                flux_data[datafile_index].append(np.exp(-10**corrected_h1_tau))
                labels[datafile_index].append("H I" if not compare_corrections else "H I (corrected)")
                colours[datafile_index].append("black")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
            if uncorrected_h1_tau is not None:
                x_data[datafile_index].append(uncorrected_h1_wavelength if wavelength else uncorrected_h1_redshift if redshift else uncorrected_h1_redshift * speed_of_light_kms)
                flux_data[datafile_index].append(np.exp(-10**uncorrected_h1_tau))
                labels[datafile_index].append("Ly $\\alpha$" if not compare_corrections else "H I (Ly $\\alpha$)")
                colours[datafile_index].append("black")
                linestyles[datafile_index].append("--" if compare_corrections else "-")
                alphas[datafile_index].append(1.0)
            if corrected_c4_tau is not None:
                x_data[datafile_index].append(corrected_c4_wavelength if wavelength else corrected_c4_redshift if redshift else corrected_c4_redshift * speed_of_light_kms)
                flux_data[datafile_index].append(np.exp(-10**corrected_c4_tau))
                labels[datafile_index].append("C IV" if not compare_corrections else "C IV (corrected)")
                colours[datafile_index].append("blue")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
            if uncorrected_c4_tau is not None:
                x_data[datafile_index].append(uncorrected_c4_wavelength if wavelength else uncorrected_c4_redshift if redshift else uncorrected_c4_redshift * speed_of_light_kms)
                flux_data[datafile_index].append(np.exp(-10**uncorrected_c4_tau))
                labels[datafile_index].append("C IV" if not compare_corrections else "C IV (raw)")
                colours[datafile_index].append("blue")
                linestyles[datafile_index].append("--" if compare_corrections else "-")
                alphas[datafile_index].append(1.0)
            if corrected_o6_tau is not None:
                x_data[datafile_index].append(corrected_o6_wavelength if wavelength else corrected_o6_redshift if redshift else corrected_o6_redshift * speed_of_light_kms)
                flux_data[datafile_index].append(np.exp(-10**corrected_o6_tau))
                labels[datafile_index].append("O VI" if not compare_corrections else "O VI (corrected)")
                colours[datafile_index].append("red")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
            if uncorrected_o6_tau is not None:
                x_data[datafile_index].append(uncorrected_o6_wavelength if wavelength else uncorrected_o6_redshift if redshift else uncorrected_o6_redshift * speed_of_light_kms)
                flux_data[datafile_index].append(np.exp(-10**uncorrected_o6_tau))
                labels[datafile_index].append("O VI" if not compare_corrections else "O VI (raw)")
                colours[datafile_index].append("red")
                linestyles[datafile_index].append("--" if compare_corrections else "-")
                alphas[datafile_index].append(1.0)

        else:
            if h1:
                x_data[datafile_index].append(data["Wavelength_Ang"][:])
                flux_data[datafile_index].append(np.exp(-selected_spectrum_data["h1/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
                labels[datafile_index].append("H I")
                colours[datafile_index].append("black")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
            if c4:
                x_data[datafile_index].append(data["Wavelength_Ang"][:])
                flux_data[datafile_index].append(np.exp(-selected_spectrum_data["c4/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
                labels[datafile_index].append("C IV")
                colours[datafile_index].append("blue")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)
            if o6:
                x_data[datafile_index].append(data["Wavelength_Ang"][:])
                flux_data[datafile_index].append(np.exp(-selected_spectrum_data["o6/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
                labels[datafile_index].append("O VI")
                colours[datafile_index].append("red")
                linestyles[datafile_index].append("-")
                alphas[datafile_index].append(1.0)

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
            title = title,
            show = output_file is None
        )

    else:
        avalible_line_styles = ("-", "--", "-.", ":")
        figure, axis = None, None
        for datafile_index in range(len(datafiles)):
            figure, axis = plotting.spectrum(
                pixel_x_values = x_data[datafile_index],
                pixel_fluxes = flux_data[datafile_index],
                spectrum_type = plotting.PlottedSpectrumType.wavelength if wavelength else plotting.PlottedSpectrumType.velocity if velocity else plotting.PlottedSpectrumType.redshift,
                dataset_labels = [datafile_names[datafile_index]],
                dataset_colours = colours[datafile_index],
                dataset_linestyles = [avalible_line_styles[datafile_index % len(avalible_line_styles)]],
                dataset_alphas = alphas[datafile_index],
                x_min = x_min,
                x_max = x_max,
                title = title,
                show = (output_file is None) and datafile_index + 1 == len(datafiles),
                figure = figure,
                axis = axis
            )

    # Decide what to do with the plot

    if output_file is not None:
        figure.savefig(output_file)

def main():
    script_wrapper = ScriptWrapper(
        command = "pod-plot-specwizard-spectrum",
        authors = [ScriptWrapper.AuthorInfomation(given_name = "Christopher", family_name = "Rowe", email = "contact@cjrrowe.com", website_url = "cjrrowe.com")],
        version = "1.0.0",
        edit_date = datetime.date(2024, 5, 26),
        description = "Plots a spectrum from data in SpecWizard output files.",
        dependancies = ["podpy-refocused"],
        usage_paramiter_examples = None,
        parameters = [
            ScriptWrapper.OptionalParam[str](
                "datafiles", "i",
                description = "SpecWizard output file(s).\nDefaults to a file in the current working directory named \"LongSpectrum.hdf5\".\nSpecify as a semicolon seperated list to compare data between files.",
                default_value = ["./LongSpectrum.hdf5"],
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.OptionalParam[str](
                "datafile-names",
                description = "Display names for each input file.\Redundant if only one data file is specified.\nSpecify a number of names as a semicolon seperated list.",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                requirements = ["datafiles"]
            ),
            ScriptWrapper.Flag(
                "raw",
                description = "Plot the raw flux.\nOnly supported for --wavelength",
                conflicts = ["velocity", "redshift"],
                requirements = ["wavelength"]
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
                "o6",
                description = "Plot fluxes from O VI."
            ),
            ScriptWrapper.OptionalParam[int](
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
                description = "Use velocity pixels.",
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
                conflicts = ["no-h1-corrections", "no-metal-corrections", "compare-corrections"]
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
    )
#    script_wrapper = ScriptWrapper("pod-plot-specwizard-spectrum",
#                                   "Christopher Rowe",
#                                   "1.0.0",
#                                   "26/05/2024",
#                                   "Plots a spectrum from data in SpecWizard output files.",
#                                   ["podpy-refocused"],
#                                   [],
#                                   [],
#                                   [["datafile",                "i",    "SpecWizard output file. Defaults to a file in the current working directory named \"LongSpectrum.hdf5\".", False, False, None, "./LongSpectrum.hdf5"],
#                                    ["raw",                     None,   "Plot the raw flux.\nOnly supported for --wavelength", False, True, None, None, ["velocity", "redshift"]],
#                                    ["h1",                      None,   "Plot fluxes from H I.", False, True, None, None],
#                                    ["c4",                      None,   "Plot fluxes from C IV.", False, True, None, None],
#                                    ["o6",                      None,   "Plot fluxes from O VI.", False, True, None, None],
#                                    ["target-spectrum",         "s",    "Index of the spectrum to plot.\nDefault is 0.", False, False, float, 0],
#                                    ["wavelength",              "w",    "Use wavelength pixels.", False, True, None, None, ["velocity", "redshift"]],
#                                    ["velocity",                None,    "Use velocity pixels.", False, True, None, None, ["raw", "wavelength", "redshift"]],
#                                    ["redshift",                "r",    "Use redshift pixels.", False, True, None, None, ["raw", "wavelength", "velocity"]],
#                                    ["x-min",                   None,   "Minimum log10-space value of tau_HI to consider.\nDefault is -1.0", False, False, float, None],
#                                    ["x-max",                   None,   "Maximum log10-space value of tau_HI to consider.\nDefault is 2.5", False, False, float,  None],
#                                    ["no-h1-corrections",       None,   "Make no alterations to saturated H I pixels.", False, True, None, None, ["compare-corrections", "use-RSODOST-field"]],
#                                    ["no-metal-corrections",    None,   "Make no alterations to saturated metal pixels.", False, True, None, None, ["compare-corrections", "use-RSODOST-field"]],
#                                    ["compare-corrections",     None,   "Plot comparison lines for all selected ions to show their un-corrected values.", False, True, None, None, ["no-h1-corrections", "no-metal-corrections", "use-RSODOST-field"]],
#                                    ["use-RSODOST-field",       None,   "Get ion flux from the 'RedshiftSpaceOpticalDepthOfStrongestTransition' field.", False, True, None, None, ["no-h1-corrections", "no-metal-corrections", "compare-corrections"]],
#                                    ["title",                   None,   "Optional plot title.", False, False, None, None],
#                                    ["output-file",             "o",    "Name of the file to save plot as. Leave unset to show the plot in a matplotlib interactive window.", False, False, None, None]
#                                   ])
    
    script_wrapper.run(__main)
