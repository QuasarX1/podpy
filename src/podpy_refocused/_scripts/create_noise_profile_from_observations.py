# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from .._SpecWizard import SpecWizard_NoiseProfile
from .. import plotting

import datetime
import os
from typing import Union, List, Callable

from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper, Struct, TypedAutoProperty, TypeCastAutoProperty, TypeShield, NestedTypeShield, cast_unyt_quantity, cast_unyt_array
from matplotlib import pyplot as plt
import numpy as np
from unyt import unyt_quantity, unyt_array, angstrom
from astropy.io import fits
import h5py as h5



def rms(values: np.ndarray):
    if len(values) == 0:
        return 0.0
    return np.sqrt((values**2).mean())



class ObservedData(Struct):
    """
    wavelengths

    normalised_fluxes

    number_of_pixels

    pixel_width

    read_fits(filepath) (static)

    copy()

    filter(wavelength_min, wavelength_max)
    """

    wavelengths = TypeCastAutoProperty[unyt_array](
        cast_unyt_array(angstrom)
    )

    normalised_fluxes = TypedAutoProperty[np.ndarray](
        NestedTypeShield(np.ndarray, float)
    )

    number_of_pixels = TypedAutoProperty[int](
        TypeShield(int)
    )

    pixel_width = TypeCastAutoProperty[int](
        cast_unyt_quantity(angstrom)
    )

    def __len__(self) -> int:
        return len(self.wavelengths)

    @property
    def min_wavelength(self) -> unyt_quantity:
        return self.wavelengths[0]
    @property
    def max_wavelength(self) -> unyt_quantity:
        return self.wavelengths[-1]

    @staticmethod
    @TypeShield(str)
    def read_fits(filepath: str) -> "ObservedData":
        """
        Read a processed FITS data file.

        Format:
            Chunk 0:
                Dataset 0: 1D normalised flux
            Header:
                UP_WLSRT: Wavelength of first pixel in Angstroms
                UP_WLEND: Wavelength of last pixel in Angstroms
        """

        with fits.open(filepath) as file:
            pixel_wavelengths = np.linspace(file[0].header["UP_WLSRT"], file[0].header["UP_WLEND"], len(file[0].data[0]))
            pixel_fluxes = file[0].data[0]

            n_px = len(pixel_wavelengths)
            pixel_width = (file[0].header["UP_WLEND"] - file[0].header["UP_WLSRT"]) / n_px

            return ObservedData(
                wavelengths = pixel_wavelengths,
                normalised_fluxes = pixel_fluxes,
                number_of_pixels = n_px,
                pixel_width = pixel_width
            )
        
    def get_min_flux(self) -> float:
        return float(self.normalised_fluxes.min())
    def get_max_flux(self) -> float:
        return float(self.normalised_fluxes.max())

    def copy(self):
        return ObservedData(
            wavelengths = self.wavelengths.copy(),
            normalised_fluxes = self.normalised_fluxes.copy(),
            number_of_pixels = self.number_of_pixels,
            pixel_width = self.pixel_width
        )

    def filter(self, wavelength_min: Union[float, None] = None, wavelength_max: Union[float, None] = None) -> "ObservedData":
        if wavelength_min is None and wavelength_max is None:
            Console.print_verbose_warning("ObservedData filter() called with no limits.\nIf this is intentional, use the copy() method instead.")
            return self.copy()
        pixel_filter = np.full_like(self.wavelengths, True)
        if wavelength_min is not None:
            pixel_filter[self.wavelengths < wavelength_min] = False
        if wavelength_max is not None:
            pixel_filter[self.wavelengths >= wavelength_max] = False
        return ObservedData(
                wavelengths = self.wavelengths.copy(),
                normalised_fluxes = self.normalised_fluxes.copy(),
                number_of_pixels = self.number_of_pixels,
                pixel_width = self.pixel_width
        )
    


def main():
    ScriptWrapper(
        command = "pod-create-specwizard-noise-file",
        authors = [ScriptWrapper.AuthorInfomation(given_name = "Christopher", family_name = "Rowe", email = "contact@cjrrowe.com", website_url = "cjrrowe.com")],
        version = "1.0.0",
        edit_date = datetime.date(2024, 6, 24),
        description = "Plots the relation between two ion's tau values.",
        dependancies = ["podpy-refocused"],
        usage_paramiter_examples = None,
        parameters = [
            ScriptWrapper.OptionalParam[Union[str, None]](
                name = "input-filepath",
                short_name = "i",
                description = "Observation data file to load flux data from."
            ),
            ScriptWrapper.OptionalParam[Union[str, None]](
                name = "output-filepath",
                short_name = "o",
                description = "File to write noise data to. HDF5 format."
            ),
            ScriptWrapper.Flag(
                name = "overwrite-noise-file",
                sets_param = "allow_overwrite_output_file",
                description = "Allow overwriting of an existing noise file with the same name."
            ),
            ScriptWrapper.Flag(
                name = "standard-deviation",
                sets_param = "use_standard_deviation",
                description = "Noise sigma is the standard deviation of the normalised flux.\nIncompatible with --rms and --set-uniform-value.\nOnly valid when using input observational data.",
                requirements = ["input-filepath"],
                conflicts = ["rms", "set-uniform-value"]
            ),
            ScriptWrapper.Flag(
                name = "rms",
                sets_param = "use_rms",
                description = "Noise sigma is the RMS of the normalised flux.\nIncompatible with --standard-deviation and --set-uniform-value.\nOnly valid when using input observational data.",
                requirements = ["input-filepath"],
                conflicts = ["standard-deviation", "set-uniform-value"]
            ),
            ScriptWrapper.OptionalParam[Union[float, None]](
                name = "set-uniform-value",
                description = "Set the noise sigma in all pixels to a fixed value.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[Union[unyt_quantity, None]](
                name = "default-pixel-width",
                description = "Pixel width (in Angstroms) to use when no observational data is provided.",
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom),
                conflicts = ["input-filepath"]
            ),
            ScriptWrapper.OptionalParam[unyt_quantity](
                "wavelength-window-width",
                description = "Width of the window in Angstroms.",
                default_value = unyt_quantity(150.0, units = angstrom),
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom)
            ),
            ScriptWrapper.OptionalParam[float](
                "flux-chunk-size",
                description = "Size of the notmalised flux bins.",
                default_value = 0.2,
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[unyt_quantity | None](
                "min-wavelength",
                description = "Minimum wavelength to consider.\nMay be onitted if an observational data file is provided, in which case\nthe minimum wavelength will be set by the avalible data.\nWhen set by input data, this will be resolved before the min and max flux.",
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom)
            ),
            ScriptWrapper.OptionalParam[unyt_quantity | None](
                "max-wavelength",
                description = "Maximum wavelength to consider.\nMay be onitted if an observational data file is provided, in which case\nthe maximum wavelength will be set by the avalible data.\nWhen set by input data, this will be resolved before the min and max flux.",
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom)
            ),
            ScriptWrapper.OptionalParam[float | None](
                "min-flux",
                description = "Minimum normalised flux to consider.\nMay be onitted if an observational data file is provided, in which case\nthe minimum flux will be set by the avalible data.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "max-flux",
                description = "Maximum normalised flux to consider.\nMay be onitted if an observational data file is provided, in which case\nthe maximum flux will be set by the avalible data.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "min-sigma",
                description = "Minimum sigma value to consider.\nSmaller values will be set to this value (including zero values).\nIncompattible with --set-uniform-value.",
                conversion_function = float,
                conflicts = ["set-uniform-value"]
            ),
            ScriptWrapper.OptionalParam[float | None](
                "max-sigma",
                description = "Maximum sigma value to consider.\nLarger values will be restricted to this value.\nIncompattible with --set-uniform-value.",
                conversion_function = float,
                conflicts = ["set-uniform-value"]
            ),
            ScriptWrapper.OptionalParam[float | None](
                "default-sigma",
                description = "Sigma value for pixels outside of the wavelength range of input observational data.\nIncompattible with --set-uniform-value.",
                conversion_function = float,
                requirements = ["input-filepath"],
                conflicts = ["set-uniform-value", "default-sigma-is-min", "default-sigma-is-max"]
            ),
            ScriptWrapper.Flag(
                name = "default-sigma-is-min",
                description = "Automatically set the value of --default-sigma to the minumum calculated value, or the specified minimum if set.\nIncompattible with --set-uniform-value, --default-sigma and --default-sigma-is-max.",
                conflicts = ["set-uniform-value", "default-sigma", "default-sigma-is-max"]
            ),
            ScriptWrapper.Flag(
                name = "default-sigma-is-max",
                description = "Automatically set the value of --default-sigma to the minumum calculated value, or the specified maximum if set.\nIncompattible with --set-uniform-value, --default-sigma and --default-sigma-is-min.",
                conflicts = ["set-uniform-value", "default-sigma", "default-sigma-is-min"]
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "plot-filename",
                short_name = "p",
                description = "Plot the results and save the image to this file."
            ),
            ScriptWrapper.Flag(
                name = "show",
                short_name = "s",
                sets_param = "show_plot",
                description = "Plot the results and show using the matplotlib interactive window."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "colourmap-name",
                short_name = "c",
                description = "Set the colourmap used by matplotlib using a built-in colourmap name.\nDefaults to \"viridis\".",
                default_value = "viridis"
            ),
            ScriptWrapper.OptionalParam[float](
                name = "figure-width",
                description = "Set the width of the plot in inches.\nDefaults to 12.",
                default_value = 12.0
            ),
            ScriptWrapper.OptionalParam[float](
                name = "figure-height",
                description = "Set the height of the plot in inches.\nDefaults to 6.",
                default_value = 6.0
            )
        ]
    ).run(__main)




def __main(
            input_filepath: Union[str, None],
            output_filepath: Union[str, None],
            allow_overwrite_output_file: bool,
            use_standard_deviation: bool,
            use_rms: bool,
            set_uniform_value: float,
            default_pixel_width: Union[unyt_quantity, None], # Angstroms
            wavelength_window_width: unyt_quantity, # Angstroms
            flux_chunk_size: float, # Normalised Flux
            min_wavelength: Union[unyt_quantity, None], # Angstroms
            max_wavelength: Union[unyt_quantity, None], # Angstroms
            min_flux: Union[float, None], # Normalised Flux
            max_flux: Union[float, None], # Normalised Flux
            min_sigma: Union[float, None],
            max_sigma: Union[float, None],
            default_sigma: Union[float, None],
            default_sigma_is_min: bool,
            default_sigma_is_max: bool,
            plot_filename: str,
            show_plot: bool,
            colourmap_name: str,
            figure_width: float,
            figure_height: float
        ) -> None:

    # Is a plot requested

    create_plot: bool = plot_filename is not None or show_plot

    # Ensure a form of output is specified

    if output_filepath is None and not create_plot:
        Console.print_error("Plotting not enabled and no output file specified!")
        Console.print_info("Terminating...")
        return

    if not use_standard_deviation and not use_rms and set_uniform_value is None:
        Console.print_error("No error scheme specified.")
        Console.print_info("Terminating...")
        return

    # Ensure a form of input is specified

    if input_filepath is None and set_uniform_value is None:
        Console.print_error("No input format specified. Either provide observational data or specify a uniform value.")
        Console.print_info("Terminating...")
        return

    # Ensure the space is well defined
    
    if input_filepath is None:
        if default_pixel_width is None:
            Console.print_error("No input data specified and no value for --default-pixel-width.")
            Console.print_info("Terminating...")
            return
        elif min_wavelength is None:
            Console.print_error("No input data specified and not all limits are set. No value for --min-wavelength.")
            Console.print_info("Terminating...")
            return
        elif max_wavelength is None:
            Console.print_error("No input data specified and not all limits are set. No value for --max-wavelength.")
            Console.print_info("Terminating...")
            return
        elif min_flux is None:
            Console.print_error("No input data specified and not all limits are set. No value for --min-flux.")
            Console.print_info("Terminating...")
            return
        elif max_flux is None:
            Console.print_error("No input data specified and not all limits are set. No value for --max-flux.")
            Console.print_info("Terminating...")
            return

    # Check existing file can be overwritten
    
    if output_filepath is not None and not allow_overwrite_output_file and os.path.exists(output_filepath):
        Console.print_error(f"Output noise file already exists at \"{output_filepath}\".\nPlease remove the file or specify --overwrite-noise-file")
        Console.print_info("Terminating...")
        return

    # Set default sigma if not data dependant

    if min_sigma is not None and default_sigma_is_min:
        default_sigma = min_sigma
    if max_sigma is not None and default_sigma_is_max:
        default_sigma = max_sigma

    # Is observational data avalible

    if input_filepath is not None:

        # Read data

        data: ObservedData = ObservedData.read_fits(input_filepath)

        # Restrict pixels if wavelength limits are narrower

        restrict_min = min_wavelength > data.min_wavelength if min_wavelength is not None else False
        restrict_max = max_wavelength < data.max_wavelength if max_wavelength is not None else False
        if restrict_min or restrict_max:
            data = data.filter(
                wavelength_min = min_wavelength if restrict_min else None,
                wavelength_max = max_wavelength if restrict_max else None
            )
        
        # Grab missing space definitions

        if min_wavelength is None:
            min_wavelength = data.min_wavelength
        if max_wavelength is None:
            max_wavelength = data.max_wavelength
        if min_flux is None:
            min_flux = data.get_min_flux()
        if max_flux is None:
            max_flux = data.get_max_flux()

        # Get the best matching wavelength window width in pixels

        wavelength_window_pixel_width: int = int(np.ceil(wavelength_window_width / data.pixel_width).value)
        if wavelength_window_pixel_width % 2 == 0:
            # If the number of wavelength pixels in the window is even, make it odd so that it becomes symetrical
            wavelength_window_pixel_width += 1
        wavelength_window_wing_length = int((wavelength_window_pixel_width - 1) / 2) # This will always be exact due to above even correction

        # Ensure the wavelength limimits are exact pixel values

        limits_outside_data = False
        if min_wavelength != data.min_wavelength:
            # The min wavelength is outside the limits of the observational data
            # Need to make sure it falls on an exact pixel!
            min_wavelength = data.min_wavelength - (data.pixel_width * np.ceil((data.min_wavelength - min_wavelength) / data.pixel_width))
            limits_outside_data = True
        if max_wavelength != data.max_wavelength:
            # The max wavelength is outside the limits of the observational data
            # Need to make sure it falls on an exact pixel!
            max_wavelength = data.max_wavelength + (data.pixel_width * np.ceil((max_wavelength - data.max_wavelength) / data.pixel_width))
            limits_outside_data = True

        # Recover flux limits if unspecified

        if min_flux is None:
            min_flux = data.get_min_flux()
        if max_flux is None:
            max_flux = data.get_max_flux()

        if default_sigma is None and limits_outside_data and set_uniform_value is None and not default_sigma_is_min and not default_sigma_is_max:
            Console.print_error(f"Wavelength limits ({min_wavelength} -> {max_wavelength}) lie outside the bounds of the input data ({data.min_wavelength} -> {data.max_wavelength}) but no default value for sigma is specified.")
            Console.print_info("Terminating...")
            return

        pixel_width: unyt_quantity = data.pixel_width

    else:
        pixel_width: unyt_quantity = default_pixel_width

    # Initialise wavelength array

    if Settings.debug:
        assert (((max_wavelength - min_wavelength) / pixel_width) + 1) % 1 == 0, "Calculation for number of selected pixels didn't yeild an exact integer!"

    selected_wavelengths: unyt_array = unyt_array(np.linspace(min_wavelength, max_wavelength, int(((max_wavelength - min_wavelength) / pixel_width) + 1)), dtype = float, units = angstrom)

    number_of_pixels = selected_wavelengths.shape[0]

    # Identify the flux windows

    # Ensurethat there is a flux window centred on 0.0
    half_flux_chunk_size = flux_chunk_size / 2
    n_flux_windows_above_zero = int(np.ceil((max_flux - half_flux_chunk_size) / flux_chunk_size))
    n_flux_windows_below_zero = ( int(np.ceil((-min_flux - half_flux_chunk_size) / flux_chunk_size)) ) if min_flux < 0 else 0
    n_flux_windows = n_flux_windows_above_zero + 1 + n_flux_windows_below_zero
    max_window_flux = half_flux_chunk_size + (n_flux_windows_above_zero * flux_chunk_size)
    min_window_flux = -half_flux_chunk_size - (( n_flux_windows_below_zero * flux_chunk_size ) if n_flux_windows_below_zero > 0 else 0)
    #TODO: this isn't a great way of handling non-negitive windows!
    flux_centres = np.linspace(min_window_flux + half_flux_chunk_size, max_window_flux - half_flux_chunk_size, n_flux_windows)#TODO: how many STEPS?

    if input_filepath is not None:

        # Compute masks for each flux window

        flux_filters = [(data.normalised_fluxes >= (central_flux - half_flux_chunk_size)) & (data.normalised_fluxes < (central_flux + half_flux_chunk_size)) for central_flux in flux_centres]

        # Set sigma calculation function

        compute_me: Union[Callable[[np.ndarray], float], None] = (lambda d: float(np.std(d)) if len(d) > 0 else 0.0) if use_standard_deviation else rms if use_rms else None

    # Calculate errors

    sigma_of_flux = np.empty(shape = (number_of_pixels, n_flux_windows), dtype = float)

    if set_uniform_value is None:

        Console.print_info(f"Computing sigma for {len(selected_wavelengths)} pixels ({selected_wavelengths[0]} -> {selected_wavelengths[-1]}, sliding window width {(wavelength_window_pixel_width * data.pixel_width)})\nand {n_flux_windows} normalised flux bins (centres from {flux_centres[0]} to {flux_centres[-1]}, bin width {flux_chunk_size})")

        obs_data_pixels = (data.min_wavelength <= selected_wavelengths) & (selected_wavelengths <= data.max_wavelength)
        obs_data_pixel_indexes = np.argwhere(obs_data_pixels).flatten()

        for px_i in range(len(data)):
            wavelength_window_start = px_i - wavelength_window_wing_length
            wavelength_window_end = px_i + wavelength_window_wing_length + 1 # +1 needed as the slice needs to be the next index
            if wavelength_window_start < 0:
                wavelength_window_start = 0
            if wavelength_window_end > len(data):
                wavelength_window_end = len(data)
            print(f"\rLambda px_i {px_i}/{len(data) - 1} [{wavelength_window_start} : {wavelength_window_end}]", end = "            ")
            for i in range(n_flux_windows):
                sigma_of_flux[obs_data_pixel_indexes[px_i], i] = compute_me(data.normalised_fluxes[wavelength_window_start : wavelength_window_end][flux_filters[i][wavelength_window_start : wavelength_window_end]])
        print()

        # Apply sigma limits

        if min_sigma is not None:
            sigma_of_flux[(sigma_of_flux < min_sigma) & obs_data_pixels] = min_sigma
        if max_sigma is not None:
            sigma_of_flux[(sigma_of_flux > max_sigma) & obs_data_pixels] = max_sigma

        # Ensure the default sigma value is set

        if default_sigma is None:
            if default_sigma_is_min:
                default_sigma = sigma_of_flux[obs_data_pixels, :].min()
            elif default_sigma_is_max:
                default_sigma = sigma_of_flux[obs_data_pixels, :].max()
            else:
                if (~obs_data_pixels).sum() > 0:
                    # Not possible unless something broke!
                    raise RuntimeError("This shouldn't happen! Please report!")

        # Apply the default value to pixels outside the range

        sigma_of_flux[~obs_data_pixels, :] = default_sigma

        Console.print_info("Done calculating 2D sigma interpolation table.")

    else:
        sigma_of_flux.fill(set_uniform_value)

        Console.print_info(f"Created 2D sigma interpolation table with all sigma values set to {set_uniform_value}.")

    # Create object

    noise_profile = SpecWizard_NoiseProfile(
        wavelengths = selected_wavelengths,
        normalised_fluxes = flux_centres,
        sigma_table = sigma_of_flux
    )

    # Save to file

#    if output_filepath is not None:
#        with h5.File(output_filepath, "w") as f:
#            f.create_dataset(name = "NormalizedFlux",      data = np.array(flux_centres, dtype = np.float32))
#            f.create_dataset(name = "Wavelength_Angstrom", data = np.array(selected_wavelengths, dtype = np.float32))
#            f.create_dataset(name = "NormalizedNoise",     data = np.array(sigma_of_flux.T, dtype = np.float32))
    noise_profile.write(output_filepath, allow_overwrite = allow_overwrite_output_file)

    # Plot

    if create_plot:

#        plt.imshow(sigma_of_flux.T, aspect = "auto", origin = "lower", extent = (selected_wavelengths[0], selected_wavelengths[-1], flux_centres[0], flux_centres[-1]), interpolation = "none", cmap = colourmap_name)
#        plt.colorbar(label = "RMS Error")
#        #plt.plot(selected_wavelengths, pixel_fluxes, color = "orange", label = "Normalised Flux")
#        plt.xlim((selected_wavelengths[0], selected_wavelengths[-1]))
#        plt.ylim((flux_centres[0], flux_centres[-1]))
#        plt.xlabel(f"Wavelength ({angstrom.units})")
#        plt.ylabel("Normalized Flux")
#        #plt.legend()
        fig, axis = plotting.specwizard_noise_table(
#            wavelengths = selected_wavelengths,
#            flux_bin_centres = flux_centres,
#            sigma_table = sigma_of_flux,
            noise_profile = noise_profile,
            sigma_label = "RMS Error" if use_rms else "Standard Deviation" if use_standard_deviation else None,
            colourmap = colourmap_name,
            axis_label_fontsize = "large",
            figure_creation_kwargs = {
                "figsize": (figure_width, figure_height),
                "layout": "constrained"
#                "layout": "tight"
            }
        )

        # Save plot
    
        if plot_filename is not None:
            fig.savefig(plot_filename)

        # Show plot
    
        if show_plot:
            plt.show()














"""
SHOW_PLOT = False

X_MIN = None
X_MAX = None
WAVELENGTH_CHUNK_WIDTH = 120.0 # A
FLUX_CHUNK_WIDTH = 0.2 # Normalised flux

import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import h5py as h5
from QuasarCode import Console
from QuasarCode.IO.Configurations import PropertiesConfig

def rms(values: np.ndarray):
    if len(values) == 0:
        return 0.0
#    return np.sqrt((np.abs(values - values.mean())**2).mean())
    return np.sqrt((values**2).mean())

object_name = sys.argv[1] if len(sys.argv) > 1 else "Q1317-0507"

mapping = PropertiesConfig(False, "T16_mapping.properties")

filename = mapping[object_name]

f = fits.open(filename)



pixel_wavelengths = np.linspace(f[0].header["UP_WLSRT"], f[0].header["UP_WLEND"], len(f[0].data[3]))
#pixel_filter = (pixel_wavelengths >= X_MIN) & (pixel_wavelengths <= X_MAX)
pixel_fluxes = f[0].data[0]

n_px = len(pixel_wavelengths)
pixel_width = (f[0].header["UP_WLEND"] - f[0].header["UP_WLSRT"]) / n_px



f.close()



wavelength_window_pixel_width = int(np.ceil(WAVELENGTH_CHUNK_WIDTH / pixel_width))
if wavelength_window_pixel_width % 2 == 0:
    # If the number of wavelength pixels in the window is even, make it odd so that it becomes symetrical
    wavelength_window_pixel_width += 1
wavelength_window_wing_length = int((wavelength_window_pixel_width - 1) / 2) # This will always be exact due to above even correction

min_flux = -2.0#pixel_fluxes.min()
max_flux = 2.0#pixel_fluxes.max()
# Ensurethat there is a flux window centred on 0.0
n_flux_windows_above_zero = int(np.ceil((max_flux - (FLUX_CHUNK_WIDTH / 2)) / FLUX_CHUNK_WIDTH))
n_flux_windows_below_zero = ( int(np.ceil((-min_flux - (FLUX_CHUNK_WIDTH / 2)) / FLUX_CHUNK_WIDTH)) ) if min_flux < 0 else 0
n_flux_windows = n_flux_windows_above_zero + 1 + n_flux_windows_below_zero
max_window_flux = (FLUX_CHUNK_WIDTH / 2) + (n_flux_windows_above_zero * FLUX_CHUNK_WIDTH)
min_window_flux = -(FLUX_CHUNK_WIDTH / 2) - (( n_flux_windows_below_zero * FLUX_CHUNK_WIDTH ) if n_flux_windows_below_zero > 0 else 0)
#TODO: this isn't a great way of handling non-negitive windows!
flux_centres = np.linspace(min_window_flux + (FLUX_CHUNK_WIDTH / 2), max_window_flux - (FLUX_CHUNK_WIDTH / 2), n_flux_windows)#TODO: how many STEPS?
flux_filters = [(pixel_fluxes >= (central_flux - (FLUX_CHUNK_WIDTH / 2))) & (pixel_fluxes < (central_flux + (FLUX_CHUNK_WIDTH / 2))) for central_flux in flux_centres]



print(pixel_wavelengths.shape[0], n_flux_windows)
n_selected_px = np.zeros_like(pixel_wavelengths, dtype = int)
sigma_of_flux = np.empty(shape = (pixel_wavelengths.shape[0], n_flux_windows), dtype = float)
for px_i in range(n_px):
#    Console.print_info(f"Lambda px_i {px_i}/{n_px}")
#    print(f"\rLambda px_i {px_i}/{n_px}", end = "            ")
    wavelength_window_start = px_i - wavelength_window_wing_length
    wavelength_window_end = px_i + wavelength_window_wing_length + 1 # +1 needed as the slice needs to be the next index
    if wavelength_window_start < 0:
        wavelength_window_start = 0
    if wavelength_window_end > n_px:
        wavelength_window_end = n_px
    print(f"\rLambda px_i {px_i}/{n_px} [{wavelength_window_start} : {wavelength_window_end}]", end = "            ")
#    input()
    n_selected_px[px_i] = wavelength_window_end - wavelength_window_start
    for i in range(n_flux_windows):
        sigma_of_flux[px_i, i] = rms(pixel_fluxes[wavelength_window_start : wavelength_window_end][flux_filters[i][wavelength_window_start : wavelength_window_end]])
print()

#plt.plot(pixel_wavelengths, n_selected_px)
#plt.show()
#exit()



with h5.File(f"SpecWizard-noise-files/noise_{object_name}.hdf5", "w") as f:
    f.create_dataset(name = "NormalizedFlux",      data = np.array(flux_centres, dtype = np.float32))
    f.create_dataset(name = "Wavelength_Angstrom", data = np.array(pixel_wavelengths, dtype = np.float32))
    f.create_dataset(name = "NormalizedNoise",     data = np.array(sigma_of_flux.T, dtype = np.float32))



if SHOW_PLOT:
    plt.imshow(sigma_of_flux.T, aspect = "auto", origin = "lower", extent = (pixel_wavelengths[0], pixel_wavelengths[-1], flux_centres[0], flux_centres[-1]), interpolation = "none")
    plt.colorbar(label = "RMS Error")
    #plt.plot(pixel_wavelengths, pixel_fluxes, color = "orange", label = "Normalised Flux")
    plt.xlim((pixel_wavelengths[0], pixel_wavelengths[-1]))
    plt.ylim((flux_centres[0], flux_centres[-1]))
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Normalized Flux")
    #plt.legend()
    plt.show()
"""
