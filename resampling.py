import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def resampling(ys, ys_err, num_resamples, xs, xs_fit, model_func, seed, plot=True):
    # Perform resampling
    # Container for params values
    params_values = []

    np.random.seed(seed)

    for i in range(num_resamples):
        perturbed_ys = ys + np.random.normal(0, ys_err)

        # Fit the model to the perturbed data
        popt, _ = opt.curve_fit(model_func, xs, perturbed_ys)

        # Take the parameters
        params_values.append(popt)

        # Plot a few fitting curves for visualization
        if i < 50 and plot:  # Plot first 20 perturbation fits for clarity
            ys_fit = model_func(xs_fit, *popt)
            plt.plot(xs_fit, ys_fit, color="gray", alpha=0.3)

    # Calculate mean and standard deviation of params values
    params_values = np.array(params_values)
    params_mean = np.mean(params_values, axis=0)
    params_std = np.std(params_values, axis=0)
    return params_mean, params_std
