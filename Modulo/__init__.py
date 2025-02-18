from .Fitting.Regresion import (FitLineal, GFit, generatedata, gdata, PostDistrib, plot_results, plot_data_and_fit, model, model2, GaussNewton, GaussNeewton2, fbayes)
from .Uncertainty.Propagation import (error_sum, error_product, error_power, error_log, error_exponential, error_mean)

__all__ = ["FitLineal", "GFit", "generatedata", "gdata", "PostDistrib", "plot_results", "plot_data_and_fit", "model", "model2", "GaussNewton", "GaussNeewton2", "fbayes", "error_sum", "error_product", "error_power", "error_log", "error_exponential", "error_mean"]