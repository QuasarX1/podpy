# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from matplotlib import pyplot as plt
from QuasarCode import Console
from typing import Union, Tuple, Dict, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._tools import PlotObjects, require_plottable, showable
from .._TauBinned import BinnedOpticalDepthResults

@showable(kwarg_name = "show")
@require_plottable(combined_kwarg_name = "plot_objects", figure_creation_settings_kwarg_name = "figure_creation_kwargs", figure_kwarg_name = "figure", axis_kwarg_name = "axis")
def plot_pod_statistics(results: BinnedOpticalDepthResults,
                        label: str,
                        colour: Any = "blue",
                        linestyle: Any = "-",
                        hide_errors: bool = False,
                        hide_tau_min_label: bool = False,
                        x_min: Union[float, None] = None,
                        x_max: Union[float, None] = None,
                        y_min: Union[float, None] = None,
                        y_max: Union[float, None] = None,
                        x_label: Union[str, None] = None,
                        y_label: Union[str, None] = None,
                        title: Union[str, None] = None,
                        #figure: Union[Figure, None] = None,
                        #axis: Union[Axes, None] = None,
                        #figure_creation_kwargs: Union[Dict[str, Any], None] = None,
                        plot_objects: Union[PlotObjects, None] = None,
                        allow_auto_set_labels: bool = True,
                        allow_auto_set_limits: bool = True,
                        #show: bool = False,
                        density = False,
                        density_colourmap = "viridis",
                        density_uses_all_pixels = False
                    ) -> PlotObjects:
    """
    Plot the percentile statistics from a set of binned pixel optical depths.
    """

#    if figure is None:
#        if axis is None:
#            figure = plt.figure(**(figure_creation_kwargs if figure_creation_kwargs is not None else {}))
#            axis = figure.gca()
#    elif axis is None:
#        axis = figure.gca()
#    figure, axis = ensure_plottable(figure, axis, figure_creation_kwargs)
    if plot_objects is None or plot_objects.axis is None:
        raise RuntimeError("Internal error! Required plot axis object not generated.")

    if x_min is None:
        x_min = results.tau_binned_x[0]
    if x_max is None:
        x_max = results.tau_binned_x[-1]
    if x_label is None:
        x_label = f"$\\rm log_{{10}}$ $\\tau_{{\\rm {results.ion_x}}}$"
    if y_label is None:
        percentile_last_digit = int(results.percentile / 10) - (10 * int(results.percentile / 10))
        y_label = ("Median" if results.is_median else f"{results.percentile}{'st' if (results.percentile % 1.0 == 0 and percentile_last_digit == 1) else 'nd' if (results.percentile % 1.0 == 0 and percentile_last_digit == 2) else 'rd' if (results.percentile % 1.0 == 0 and percentile_last_digit == 3) else 'th' if (results.percentile % 1.0 == 0) else ''} percentile") + f" $\\rm log_{{10}}$ $\\tau_{{\\rm {results.ion_x}}}$"

    if density:
        plot_objects = plot_pod_pair_density(results, plot_objects = plot_objects, colourmap = density_colourmap, use_all_pixels = density_uses_all_pixels)

    line_object = plot_objects.axis.plot((x_min, x_max), (results.tau_min, results.tau_min), color = colour, linestyle = ":", label = "$\\tau_{\\rm min}$" if not hide_tau_min_label else None, linewidth = 2)
    if results.has_errors and not hide_errors:
        pass#TODO: add errors!
    plot_objects.axis.plot(results.tau_binned_x, results.tau_binned_y, color = line_object[0].get_color(), linestyle = linestyle, label = label, linewidth = 2)
    #axis.title("T+16 Fig. 2 comparison -- Q1317-0507, z=3.7")
    if title is not None:
        plot_objects.axis.set_title(title)
    if allow_auto_set_labels:
        plot_objects.axis.set_xlabel(x_label)
        plot_objects.axis.set_ylabel(y_label)
    plot_objects.axis.legend()
    if allow_auto_set_limits:
        plot_objects.axis.set_xlim((x_min, x_max))
        plot_objects.axis.set_ylim((y_min, y_max))

#    if show:
#        if figure is None:
#            Console.print_warning("Unable to show figure as axis object reference provided without figure object reference.")
#        else:
#            figure.show()
    return plot_objects

@showable(kwarg_name = "show")
@require_plottable(combined_kwarg_name = "plot_objects", figure_creation_settings_kwarg_name = "figure_creation_kwargs", figure_kwarg_name = "figure", axis_kwarg_name = "axis")
def plot_pod_pair_density(
        results: BinnedOpticalDepthResults,
        #figure: Union[Figure, None] = None,
        #axis: Union[Axes, None] = None,
        #figure_creation_kwargs: Union[Dict[str, Any], None] = None,
        plot_objects: Union[PlotObjects, None] = None,
        colourmap: str = "viridis",
        use_all_pixels: bool = False
    ) -> PlotObjects:
    """
    Create a hexbin plot to show the density of optical depth pixel pairs.
    """

    if not use_all_pixels:
        raise NotImplementedError("No current way to retrive exact contributing pixels. All pixels must be used at present!")

    if plot_objects is None or plot_objects.axis is None:
        raise RuntimeError("Internal error! Required plot axis object not generated.")
    
    hexbin_object = plot_objects.axis.hexbin(results.pixel_x, results.pixel_y, bins = "log", gridsize = 500, cmap = colourmap, zorder = -int(2**64))
    plot_objects.figure.colorbar(hexbin_object, label = "Number of Pixels")#TODO: integrate with API
    
    return plot_objects
