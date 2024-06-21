# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from .._Ions import Ion
from .._SpecWizard import SpecWizard_Data
from .._Spectrum import from_SpecWizard, fit_continuum, recover_c4, recover_si4, recover_o6
from .._TauBinned import bin_combined_pixels_from_SpecWizard, BinnedOpticalDepthResults
from .. import plotting

from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper
from typing import Union, List
from matplotlib import pyplot as plt
import datetime
import os
import h5py as h5
import numpy as np

DEFAULT_SPECWIZARD_FILENAME = "LongSpectrum.hdf5"



def __main(
           datafiles: List[str],
           datafile_names: Union[List[str], None],
           h1: bool,
           c4: bool,
           si4: bool,
           o6: bool,
           bin_width: bool,
           input_los_mask: Union[List[List[bool]], None],
           x_min: Union[float],
           x_max: Union[float],
           y_min: Union[float, None],
           y_max: Union[float, None],
           no_h1_corrections: bool,
           no_metal_corrections: bool,
           compare_corrections: bool,
           use_RSODOST_field: bool,
           title: Union[str, None],
           output_file: Union[str, None]
        ) -> None:

    if not (h1 or c4 or si4 or o6):
        Console.print_error("Target ion(s) must be specified!")
        Console.print_info("Terminating...")
        return

    if len(datafiles) > 1 and sum([h1, c4, si4, o6]) > 1:
        Console.print_error("Only one ion can be specified when comparing different files!")
        Console.print_info("Terminating...")
        return

    if len(datafiles) > 1 and datafile_names is not None and len(datafile_names) != len(datafiles):
        Console.print_error("Invalid number of data file names.\nIf specified, must be a number of elements matching the number of data files.")
        Console.print_info("Terminating...")
        return

    if Settings.verbose:
        Console.print_verbose_info("Reading data from the following files/directories:\n    {}".format("    \n".join(datafiles)))
        if h1: Console.print_verbose_info("Displaying H I data.")
        if c4: Console.print_verbose_info("Displaying C IV data.")
        if si4: Console.print_verbose_info("Displaying Si IV data.")
        if o6: Console.print_verbose_info("Displaying O VI data.")
        if not compare_corrections:
            Console.print_verbose_info("H I corrections: {}".format("no" if no_h1_corrections else "yes"))
            Console.print_verbose_info("Metal corrections: {}".format("no" if no_metal_corrections else "yes"))
        else:
            Console.print_verbose_info("Compairing corrections.")
        if use_RSODOST_field: Console.print_verbose_info("Reading optical depths from \"RedshiftSpaceOpticalDepthOfStrongestTransition\" field.")
        Console.print_verbose_info("Output: {}".format("interactive plot" if output_file is None else output_file))

    if datafile_names is None:
        datafile_names = [f"File #{n}" for n in range(1, 1 + len(datafiles))]

    opticaldepth_datasets = []
    overdensity_datasets = []
    labels = []
    colours = []
    linestyles = []
    alphas = []

    for i, filepath in enumerate(datafiles):
        if not os.path.isfile(filepath):
            if os.path.isdir(filepath):
                test_filepath = os.path.join(filepath, DEFAULT_SPECWIZARD_FILENAME)
                if os.path.isfile(test_filepath):
                    filepath = test_filepath
                else:
                    raise FileNotFoundError(f"Directory was provided, but contained no file named \"{DEFAULT_SPECWIZARD_FILENAME}\".\nDirectory was: {filepath}")
            else:
                raise FileNotFoundError(f"No file exists at: {filepath}")

        opticaldepth_datasets.append([])
        overdensity_datasets.append([])
        labels.append([])
        colours.append([])
        linestyles.append([])
        alphas.append([])

        sw_data = SpecWizard_Data(filepath)

        if h1 or c4 or si4:

            #uncorrected_h1_dla_masked_spectrum_objects = from_SpecWizard(
            #    filepath = filepath,
            #    sightline_filter = None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i],
            #    identify_h1_contamination = False,
            #    correct_h1_contamination = False,
            #    mask_dla = True
            #)
            corrected_h1_dla_masked_spectrum_objects = from_SpecWizard(
                filepath = filepath,
                sightline_filter = None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i],
                identify_h1_contamination = True,
                correct_h1_contamination = True,
                mask_dla = True
            )
            corrected_h1_dla_masked_h1_log10_tau = np.concatenate([spec.h1_data.tau_rec for spec in corrected_h1_dla_masked_spectrum_objects])

            if h1:
                h1_px_overdensity_arrays = sw_data.get_opticaldepth_weighted_overdensity(Ion.H_I,  None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i])
                h1_px_overdensity = np.concatenate([corrected_h1_dla_masked_spectrum_objects[i].h1_data.filter_pixel_data(h1_px_overdensity_arrays[i]) for i in range(len(corrected_h1_dla_masked_spectrum_objects))])

                #TODO: consider other flags for other recovery

                if Settings.debug: assert len(corrected_h1_dla_masked_h1_log10_tau) == len(h1_px_overdensity)

                #opticaldepth_datasets[-1].append(h1_log10_tau)
                opticaldepth_datasets[-1].append(corrected_h1_dla_masked_h1_log10_tau)
                overdensity_datasets[-1].append(np.log10(h1_px_overdensity))
                labels[-1].append("H I")
                colours[-1].append("black")
                linestyles[-1].append("-")
                alphas[-1].append(1)

            if c4:
                recover_c4(*corrected_h1_dla_masked_spectrum_objects)
                #c4_log10_tau = np.concatenate([spec.c4_data.tau_rec for spec in corrected_h1_dla_masked_spectrum_objects])
                c4_px_overdensity_arrays = sw_data.get_opticaldepth_weighted_overdensity(Ion.C_IV, None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i])
                #c4_px_overdensity = np.concatenate([corrected_h1_dla_masked_spectrum_objects[i].c4_data.filter_pixel_data(c4_px_overdensity_arrays[i]) for i in range(len(corrected_h1_dla_masked_spectrum_objects))])
                c4_px_overdensity = np.concatenate([corrected_h1_dla_masked_spectrum_objects[i].interp_f_lambda(corrected_h1_dla_masked_spectrum_objects[i].lambdaa, c4_px_overdensity_arrays[i])(corrected_h1_dla_masked_spectrum_objects[i].c4_data.lambdaa) for i in range(len(corrected_h1_dla_masked_spectrum_objects))])

                #TODO: consider other flags for other recovery

                if Settings.debug: assert len(corrected_h1_dla_masked_h1_log10_tau) == len(c4_px_overdensity)

                #opticaldepth_datasets[-1].append(c4_log10_tau)
                opticaldepth_datasets[-1].append(corrected_h1_dla_masked_h1_log10_tau)
                overdensity_datasets[-1].append(np.log10(c4_px_overdensity))
                labels[-1].append("C IV")
                colours[-1].append("blue")
                linestyles[-1].append("--")
                alphas[-1].append(1)

            if si4:
                # Get H I pixel opticaldepths
                corrected_h1_dla_masked_h1_log10_tau__by_spectrum = [spec.h1_data.tau_rec for spec in corrected_h1_dla_masked_spectrum_objects]

                # Get O VI pixel overdensity
                recover_si4(*corrected_h1_dla_masked_spectrum_objects)
                si4_px_overdensity_arrays = sw_data.get_opticaldepth_weighted_overdensity(Ion.Si_IV, None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i])

                overlapping_redshifts = [(
                    max(spec.h1_data.z[ 0], spec.si4_data.z[ 0]),
                    min(spec.h1_data.z[-1], spec.si4_data.z[-1])
                ) for spec in corrected_h1_dla_masked_spectrum_objects]

                si4_alligned__corrected_h1_non_dla_masked_h1_log10_tau = np.concatenate([corrected_h1_dla_masked_h1_log10_tau__by_spectrum[i][(spec.h1_data.z >= overlapping_redshifts[i][0]) & (spec.h1_data.z <= overlapping_redshifts[i][1])] for i, spec in enumerate(corrected_h1_dla_masked_spectrum_objects)])
                si4_px_overdensity = np.concatenate([spec.interp_f_lambda(spec.lambdaa, si4_px_overdensity_arrays[i])(spec.si4_data.lambdaa[(spec.si4_data.z >= overlapping_redshifts[i][0]) & (spec.si4_data.z <= overlapping_redshifts[i][1])]) for i, spec in enumerate(corrected_h1_dla_masked_spectrum_objects)])

                #TODO: consider other flags for other recovery

                if Settings.debug: assert len(si4_alligned__corrected_h1_non_dla_masked_h1_log10_tau) == len(si4_px_overdensity)

                opticaldepth_datasets[-1].append(si4_alligned__corrected_h1_non_dla_masked_h1_log10_tau)
                overdensity_datasets[-1].append(np.log10(si4_px_overdensity))
                labels[-1].append("Si IV")
                colours[-1].append("yellow")
                linestyles[-1].append("-.")
                alphas[-1].append(1)

        if o6:

            #uncorrected_h1_non_dla_masked_spectrum_objects = from_SpecWizard(
            #    filepath = filepath,
            #    sightline_filter = None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i],
            #    identify_h1_contamination = False,
            #    correct_h1_contamination = False,
            #    mask_dla = False
            #)
            corrected_h1_non_dla_masked_spectrum_objects = from_SpecWizard(
                filepath = filepath,
                sightline_filter = None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i],
                identify_h1_contamination = True,
                correct_h1_contamination = True,
                mask_dla = False
            )

            # Get H I pixel opticaldepths
            corrected_h1_non_dla_masked_h1_log10_tau__by_spectrum = [spec.h1_data.tau_rec for spec in corrected_h1_non_dla_masked_spectrum_objects]

            # Get O VI pixel overdensity
            recover_o6(*corrected_h1_non_dla_masked_spectrum_objects)
            o6_px_overdensity_arrays = sw_data.get_opticaldepth_weighted_overdensity(Ion.O_VI, None if input_los_mask is None else input_los_mask[0] if len(input_los_mask) == 1 else input_los_mask[i])

            overlapping_redshifts = [(
                max(spec.h1_data.z[ 0], spec.o6_data.z[ 0]),
                min(spec.h1_data.z[-1], spec.o6_data.z[-1])
            ) for spec in corrected_h1_non_dla_masked_spectrum_objects]

            o6_alligned__corrected_h1_non_dla_masked_h1_log10_tau = np.concatenate([corrected_h1_non_dla_masked_h1_log10_tau__by_spectrum[i][(spec.h1_data.z >= overlapping_redshifts[i][0]) & (spec.h1_data.z <= overlapping_redshifts[i][1])] for i, spec in enumerate(corrected_h1_non_dla_masked_spectrum_objects)])
            o6_px_overdensity = np.concatenate([spec.interp_f_lambda(spec.lambdaa, o6_px_overdensity_arrays[i])(spec.o6_data.lambdaa[(spec.o6_data.z >= overlapping_redshifts[i][0]) & (spec.o6_data.z <= overlapping_redshifts[i][1])]) for i, spec in enumerate(corrected_h1_non_dla_masked_spectrum_objects)])

            #TODO: consider other flags for other recovery

            if Settings.debug: assert len(o6_alligned__corrected_h1_non_dla_masked_h1_log10_tau) == len(o6_px_overdensity)

            #opticaldepth_datasets[-1].append(o6_log10_tau)
            opticaldepth_datasets[-1].append(o6_alligned__corrected_h1_non_dla_masked_h1_log10_tau)
            overdensity_datasets[-1].append(np.log10(o6_px_overdensity))
            labels[-1].append("O VI")
            colours[-1].append("red")
            linestyles[-1].append(":")
            alphas[-1].append(1)

    # Plot results

    if len(datafiles) == 1:
        figure, axis = plotting.pod_overdensity_relation(
            x_min = x_min,
            x_max = x_max,
            bin_width = bin_width,
            pixel_log10_h1_optical_depths = opticaldepth_datasets[0],
            pixel_log10_overdensities = overdensity_datasets[0],
            dataset_labels = labels[0],
            dataset_colours = colours[0],
            dataset_linestyles = linestyles[0],
            dataset_alphas = alphas[0],
            use_bin_median_as_centre = False,
            y_min = y_min,
            y_max = y_max,
            title = title,
            plot_1_to_1 = True,
            show = output_file is None
        )

    else:
        avalible_line_styles = ("-", "--", "-.", ":")
        figure, axis = plotting.pod_overdensity_relation(
            x_min = x_min,
            x_max = x_max,
            bin_width = bin_width,
            pixel_log10_h1_optical_depths = [d[0] for d in opticaldepth_datasets],
            pixel_log10_overdensities = [d[0] for d in overdensity_datasets],
            dataset_labels = datafile_names,
            dataset_colours = [colours[datafile_index][0] for datafile_index in range(len(datafiles))],
            dataset_linestyles = [avalible_line_styles[datafile_index % len(avalible_line_styles)] for datafile_index in range(len(datafiles))],
            dataset_alphas = [alphas[datafile_index][0] for datafile_index in range(len(datafiles))],
            use_bin_median_as_centre = False,
            y_min = y_min,
            y_max = y_max,
            title = title,
            plot_1_to_1 = True,
            show = output_file is None
        )

    # Decide what to do with the plot

    if output_file is not None:
        figure.savefig(output_file)



def main():
    ScriptWrapper(
        command = "pod-plot-specwizard-overdensity",
        authors = [ScriptWrapper.AuthorInfomation(given_name = "Christopher", family_name = "Rowe", email = "contact@cjrrowe.com", website_url = "cjrrowe.com")],
        version = "1.0.0",
        edit_date = datetime.date(2024, 6, 3),
        description = "Plots the relationship between optical depth and overdensity for different ions from data in SpecWizard output files.",
        dependancies = ["podpy-refocused"],
        usage_paramiter_examples = [f"-i ./{DEFAULT_SPECWIZARD_FILENAME} --h1 --c4 --o6 --y-min -0.6 --y-max 2.2 --title \"S/N = 100\"'"],
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
            ScriptWrapper.OptionalParam[float](
                "bin-width",
                description = "Width of the optical depth bins (in log10 sopace). Default is 0.2",
                conversion_function = float,
                default_value = 0.2
            ),
            ScriptWrapper.OptionalParam[List[List[bool]] | None](
                "input-los-mask",
                description = "Mask for line-of-sight selection from input file(s).\nUse a string of 'T' and 'F' seperated by a semicolon for each input file if more than one is specified.\nFor example: \"TFTTF\" for a file with 5 lines of sight.\nSpecifying only one input mask and many files will result in the mask being appled to each file.\nSpecifying fewer mask elements than lines of sight avalible will result in just the first N lines of sight\nbeing considered for a mask of length N.",
                conversion_function = ScriptWrapper.make_list_converter(";", lambda v: [e.lower() == "t" for e in v])
            ),
            ScriptWrapper.OptionalParam[float](
                "x-min",
                description = "Minimum X value to display. Default is -1.0",
                conversion_function = float,
                default_value = -1.0
            ),
            ScriptWrapper.OptionalParam[float](
                "x-max",
                description = "Maximum X value to display. Default is 2.5",
                conversion_function = float,
                default_value = 2.5
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
    ).run(__main)
