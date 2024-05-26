# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from QuasarCode import Console
from matplotlib import pyplot as plt
from functools import wraps
#from collections import namedtuple
from typing import NamedTuple, Union, Tuple, Dict, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes

#PlotObjects = namedtuple(typename = "PlotObjects",
#                         field_names = ["figure", "axis"],
#                         defaults = [None, None])
class PlotObjects(NamedTuple):
    figure: Union[Figure, None] = None
    axis: Union[Axes, None] = None

def ensure_plottable(plot_objects: PlotObjects = None,
                     figure_creation_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Union[Figure, None], Axes]:
#def ensure_plottable(figure: Union[Figure, None] = None,
#                     axis: Union[Axes, None] = None,
#                     figure_creation_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Union[Figure, None], Axes]:
    figure = plot_objects.figure
    axis = plot_objects.axis
    if figure is None:
        if axis is None:
            figure = plt.figure(**(figure_creation_kwargs if figure_creation_kwargs is not None else {}))
            axis = figure.gca()
    elif axis is None:
        axis = figure.gca()
    return PlotObjects(figure = figure, axis = axis)

def require_plottable(combined_kwarg_name = "plot_objects", figure_creation_settings_kwarg_name = "figure_creation_kwargs", figure_kwarg_name = "figure", axis_kwarg_name = "axis", argument_passthrough = False):
    """
    Autogenerates a missing axis object from an exising figure, or a new figure & axis pair.
    Provides both objects as a PlotObjects named tuple (names are 'figure' and 'axis').
    """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_grabbing_func = kwargs.get if argument_passthrough else kwargs.pop
            input_objects = arg_grabbing_func(combined_kwarg_name, None)
            if input_objects is None:
                input_objects = PlotObjects(figure = arg_grabbing_func(figure_kwarg_name, None), axis = arg_grabbing_func(axis_kwarg_name, None))
            plot_objects = ensure_plottable(
                plot_objects = input_objects,
                figure_creation_kwargs = arg_grabbing_func(figure_creation_settings_kwarg_name, None)
            )
            return func(*args, **kwargs, plot_objects = plot_objects)
        return wrapper
    if not isinstance(combined_kwarg_name, str):
        # You proberbly forgot the blank function brackets after the decorator!
        raise RuntimeError("Internal error. Forgot to call outer decorator method!")
    return inner

def showable(kwarg_name: str = "show", argument_passthrough = False):
    """
    Allows a plot to be shown in an interactive window.
    """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_grabbing_func = kwargs.get if argument_passthrough else kwargs.pop
            show = arg_grabbing_func(kwarg_name, False)
            result: PlotObjects = func(*args, **kwargs)
            if show:
                if isinstance(result, PlotObjects):
                    if result.figure is None:
                        Console.print_warning("Unable to show figure as axis object reference provided without figure object reference.")
                    else:
                        result.figure.show()#TODO: why dosen't this do anything?!?!
                        plt.show()
            return result
        return wrapper
    if not isinstance(kwarg_name, str):
        # You proberbly forgot the blank function brackets after the decorator!
        raise RuntimeError("Internal error. Forgot to call outer decorator method!")
    return inner