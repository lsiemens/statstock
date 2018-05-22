""" 
stat_model.py: is a module for preforming some statistical analasys of
historial stock data.
"""

import numpy
from scipy.stats import chi2
import scipy.stats

class StatEstimate:
    """ 
    Compute mean and standard variation of stock data, with confidence intervals.
    """

    def __init__(self, n_points, alpha):
        self.n_points = n_points
        self.alpha = alpha

    def computeStats(self, time, data, old=True):
        if len(data) != len(time):
            raise ValueError("time and data have mismatched shape.")

        if len(data) < self.n_points*2:
            raise ValueError("data must have " + str(self.n_points*2) + " points or more.")

        raw_time, raw_data = time[len(time) % self.n_points:], data[len(data) % self.n_points:]

        raw_data = raw_data.reshape((len(raw_data)//self.n_points, self.n_points)).swapaxes(0, 1)
        time = raw_time[self.n_points - 1::self.n_points]
        mean = numpy.mean(raw_data, axis=0)
        std = numpy.std(raw_data, axis=0, ddof=1)

        mean_err = std/numpy.sqrt(self.n_points)
        std_low = numpy.sqrt((self.n_points - 1)*std**2/chi2.isf(self.alpha/2.0, self.n_points - 1))
        std_high = numpy.sqrt((self.n_points - 1)*std**2/chi2.isf(1 - self.alpha/2.0, self.n_points - 1))

        return (time, mean, std, (mean_err, std_low, std_high), raw_data)

class StatModel:
    """ 
    Model for estimating future statistics.
    """

    def __init__(self, verbose=False, plot=None):
        self._pyplot = None
        self._verbose = verbose
        self._plot = plot
        if plot is not None:
            from matplotlib import pyplot
            self._pyplot = pyplot

    def predictStats(self, time, mean, std, error):
        time, mean, std, error = (None, None, None, (None, None, None))
        return (time, mean, std, error)

    def evaluateModel(self, time, mean, std, error):
        dof = len(mean) - 1
        mean_err, std_low, std_high = error
        ptime, pmean, pstd, perror = self.predictStats(time, mean, std, error)
        pmean_err, pstd_low, pstd_high = perror

        std_low = std - std_low
        std_high = std_high - std
        pstd_low = pstd - pstd_low
        pstd_high = pstd_high - pstd

        #----- I dont know that to use in the chi2 statistic -----#
        best_err = numpy.sqrt(mean_err[1:]**2 + perror[0][:-1]**2)
        best_err = (mean_err[1:] + perror[0][:-1])/2.0
        best_err = mean_err[1:]
        #----- end of options ------------------------------------#

        chi2_statistic_mean = numpy.sum(((pmean[:-1] - mean[1:])/best_err)**2)
        p_value_mean = 1 - chi2.cdf(chi2_statistic_mean, dof)

        #----- I dont know that to use in the chi2 statistic -----#
        best_high = numpy.sqrt(std_high[1:]**2 + pstd_low[:-1]**2)
        best_low = numpy.sqrt(std_low[1:]**2 + pstd_high[:-1]**2)

        best_high = (std_high[1:] + pstd_low[:-1])/2.0
        best_low = (std_low[1:] + pstd_high[:-1])/2.0

        best_high = std_high[1:]
        best_low = std_low[1:]
        #----- end of options ------------------------------------#

        temp_stat_std = (pstd[:-1] - std[1:])
        temp_stat_std[temp_stat_std <= 0] = temp_stat_std[temp_stat_std <= 0]/best_high[temp_stat_std <= 0]
        temp_stat_std[temp_stat_std > 0] = temp_stat_std[temp_stat_std > 0]/best_low[temp_stat_std > 0]
        chi2_statistic_std = numpy.sum(temp_stat_std**2)
        p_value_std = 1 - chi2.cdf(chi2_statistic_std, dof)

        if self._verbose:
            print("chi squared statistic (mean): " + str(chi2_statistic_mean))
            print("chi squared statistic (std): " + str(chi2_statistic_std))
            print("degrees of freedom: " + str(dof))

        if self._plot & 1 == 1:
            self._pyplot.plot(mean[1:], "r-")
            self._pyplot.plot(mean[1:] + best_err, "k-.")
            self._pyplot.plot(mean[1:] - best_err, "k-.")
            self._pyplot.plot(pmean[:-1], "g-")
            self._pyplot.plot(mean[1:]*0.0 + 1.0, "k-")
            self._pyplot.title("Mean with confidence interval")
            self._pyplot.show()

        if self._plot & 2 == 2:
            self._pyplot.plot(std[1:], "r-")
            self._pyplot.plot(std[1:] + best_high, "k-.")
            self._pyplot.plot(std[1:] - best_low, "k-.")
            self._pyplot.plot(pstd[:-1], "g-")
            self._pyplot.title("Standard deviation with confidence interval")
            self._pyplot.show()

        if self._plot & 4 == 4:
            self._pyplot.plot(mean[1:], "r-")
            self._pyplot.plot(mean[1:] + std[1:], "k-.")
            self._pyplot.plot(mean[1:] - std[1:], "k-.")
            self._pyplot.plot(pmean[:-1], "g-")
            self._pyplot.plot(mean[1:]*0.0 + 1.0, "k-")
            self._pyplot.title("Mean with standard deviation")
            self._pyplot.show()

        return (p_value_mean, p_value_std, (ptime[-1], pmean[-1], pstd[-1], (perror[0][-1], perror[1][-1], perror[2][-1])))

class trivial(StatModel):
    def predictStats(self, time, mean, std, error):
        mean_err, std_low, std_high = error
        dt = numpy.mean(time[:-1] - time[1:])
        time[:-1] = time[1:]
        time[-1] = time[-2] + dt
        return (time, mean, std, (mean_err, std_low, std_high))

class random(StatModel):
    def __init__(self, points, mean=1.000523, std=0.036, verbose=False, plot=None):
        super().__init__(verbose=verbose, plot=plot)
        self.n_points = points
        self.normal = scipy.stats.norm(loc=mean, scale=std)

    def predictStats(self, time, mean, std, error):
        mean_err, std_low, std_high = error
        dt = numpy.mean(time[:-1] - time[1:])
        time[:-1] = time[1:]
        time[-1] = time[-2] + dt
        simulated_data = self.normal.rvs((self.n_points, len(mean)))
        mean = numpy.mean(simulated_data, axis=0)
        std = numpy.std(simulated_data, axis=0, ddof=1)
        return (time, mean, std, (mean_err, std_low, std_high))
