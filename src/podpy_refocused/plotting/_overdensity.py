# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from QuasarCode import Console, Settings
from QuasarCode.Tools import ArrayVisuliser
from typing import Union, List, Tuple, Dict, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._tools import PlotObjects, require_plottable, showable
from .._TauBinned import BinnedOpticalDepthResults

class PlottedSpectrumType(Enum):
    wavelength = 0
    velocity = 1
    redshift = 2

@showable(kwarg_name = "show")
@require_plottable(combined_kwarg_name = "plot_objects", figure_creation_settings_kwarg_name = "figure_creation_kwargs", figure_kwarg_name = "figure", axis_kwarg_name = "axis")
def plot_overdensity_relation(
                    x_min: float,
                    x_max: float,
                    bin_width: float,
                    pixel_log10_h1_optical_depths: Union[List[np.ndarray], np.ndarray],
                    pixel_log10_overdensities: Union[List[np.ndarray], np.ndarray],
                    dataset_labels: List[str],
                    dataset_colours: Union[List[str], None],
                    dataset_linestyles: Union[List[str], None],
                    dataset_alphas: Union[List[float], None],
                    use_bin_median_as_centre: bool = False,
                    y_min: Union[float, None] = None,
                    y_max: Union[float, None] = None,
                    title: Union[str, None] = None,
                    plot_1_to_1: bool = True,
                    #figure: Union[Figure, None] = None,
                    #axis: Union[Axes, None] = None,
                    #figure_creation_kwargs: Union[Dict[str, Any], None] = None,
                    plot_objects: Union[PlotObjects, None] = None,
                    #show: bool = False,
                ) -> PlotObjects:
    """
    Plot the relation between optical depth and overdensity.
    """

    if not isinstance(pixel_log10_h1_optical_depths, list):
        pixel_log10_h1_optical_depths = [pixel_log10_h1_optical_depths]
    if not isinstance(pixel_log10_overdensities, list):
        pixel_log10_overdensities = [pixel_log10_overdensities]

    n_datasets = len(pixel_log10_h1_optical_depths)

    if plot_objects is None or plot_objects.axis is None:
        raise RuntimeError("Internal error! Required plot axis object not generated.")
    
    # Create a filter to identify valid pixels
    valid_pixels = [~np.isnan(pixel_log10_overdensities[dataset_index]) for dataset_index in range(n_datasets)]
    
    # Calculate median overdensity

    bin_edges = np.arange(x_min - bin_width / 2.0, x_max + bin_width / 2.0, bin_width)
    n_bins = len(bin_edges) - 1
    bin_numbers = [np.digitize(pixel_log10_h1_optical_depths[dataset_index], bins = bin_edges) for dataset_index in range(n_datasets)]
    bin_locations = [np.empty(shape = n_bins, dtype = float) for dataset_index in range(n_datasets)] if use_bin_median_as_centre else ((bin_edges[1:] + bin_edges[:-1]) / 2)
    median_overdensities = [np.empty(shape = n_bins, dtype = float) for dataset_index in range(n_datasets)]
    valid_bins = [np.full(shape = n_bins, fill_value = True, dtype = bool) for dataset_index in range(n_datasets)]
    for bin_index in range(n_bins):
        for dataset_index in range(n_datasets):
            pixel_filter = (bin_numbers[dataset_index] == bin_index + 1) & valid_pixels[dataset_index]
            if pixel_filter.sum() > 0:
                if use_bin_median_as_centre:
                    bin_locations[dataset_index][bin_index] = np.median(pixel_log10_h1_optical_depths[dataset_index][pixel_filter])
                median_overdensities[dataset_index][bin_index] = np.median(pixel_log10_overdensities[dataset_index][pixel_filter])
                if Settings.debug:
                    if median_overdensities[dataset_index][bin_index] == -np.inf:
                        Console.print_warning("Median in bin is zero overdensity (aka log10(0) = -inf)")
                    else:
                        Console.print_debug(median_overdensities[dataset_index][bin_index], len(pixel_log10_overdensities[dataset_index][pixel_filter]))
                        #Console.print_debug(ArrayVisuliser([pixel_log10_overdensities[dataset_index][pixel_filter]], [f"{bin_index}"]).render())
            else:
                if use_bin_median_as_centre:
                    bin_locations[dataset_index][bin_index] = np.nan
                median_overdensities[dataset_index][bin_index] = np.nan
                valid_bins[dataset_index][bin_index] = False

    # Plot results

    for i in range(n_datasets):
        plot_objects.axis.plot(
            bin_locations[i][valid_bins[i]] if use_bin_median_as_centre else bin_locations[valid_bins[i]],
            median_overdensities[i][valid_bins[i]],
            color = dataset_colours[i] if dataset_colours is not None else None,
            linestyle = dataset_linestyles[i] if dataset_linestyles is not None else None,
            alpha = dataset_alphas[i] if dataset_alphas is not None else None,
            label = dataset_labels[i],
            zorder = n_datasets - i
        )

    # Set the axis limits

    plot_objects.axis.set_xlim((x_min, x_max))
    default_y_lims = plot_objects.axis.get_ylim()
    plot_objects.axis.set_ylim((
        default_y_lims[0] if y_min is None else y_min,
        default_y_lims[1] if y_max is None else y_max
    ))

    # Plot optional 1:1 relation

    if plot_1_to_1:
        x_lims = plot_objects.axis.get_xlim()
        y_lims = plot_objects.axis.get_ylim()
        line_points = (
            min(plot_objects.axis.get_xlim()[0], plot_objects.axis.get_ylim()[0]),
            max(plot_objects.axis.get_xlim()[1], plot_objects.axis.get_ylim()[1]),
        )
        plot_objects.axis.plot(
            line_points,
            line_points,
            color = "grey",
            linestyle = ":",
            alpha = 0.5,
            label = "1:1",
            zorder = -100
        )
        plot_objects.axis.set_xlim(x_lims)
        plot_objects.axis.set_ylim(y_lims)

    if title is not None:
        plot_objects.axis.set_title(title)
    plot_objects.axis.set_xlabel(("median " if use_bin_median_as_centre else "") + "$\\rm log_{\\rm 10}$ $\\tau_{\\rm H I}$")
    plot_objects.axis.set_ylabel("median $\\rm log_{\\rm 10}$ $\\delta$")
    plot_objects.axis.legend()

    return plot_objects
