""" 
Give the expected return from a product distribution model using
conditional expectation values cutting out some outliers
"""

import scipy.stats
import numpy
from matplotlib import pyplot

#statistics for daily relative change
mean_rel = 1.000353
std_rel = 0.012
#number of days to compute over
n = 20*12

#prepaire aproximate product distribution (lognorm distribution)
ln_mean = numpy.log(mean_rel**(2*n)/numpy.sqrt((mean_rel**2 + std_rel**2)**n))
ln_std = numpy.sqrt(numpy.log((mean_rel**2 + std_rel**2)**n/mean_rel**(2*n)))
lognorm = scipy.stats.lognorm(s=ln_std, loc=0, scale=numpy.exp(ln_mean))


min, max, dx = 0.0, lognorm.ppf(0.95), 0.001
#mid = lognorm.ppf(0.5)
mid = 1.0

total = max*10
print("min", round(min, 5), "mid", round(mid, 5), "max", round(max, 5), "dx", round(dx, 5))

def integrate(f, min, max, dx=0.1):
    x = numpy.linspace(min, max, (max - min)/dx)
    dx = numpy.mean(x[1:] - x[:-1])
    return numpy.trapz(f(x), x, dx)

# normalization constants
n1, n2, n3 = integrate(lambda x: lognorm.pdf(x), min, mid, dx), integrate(lambda x: lognorm.pdf(x), mid, max, dx), integrate(lambda x: lognorm.pdf(x), max, total, dx)
# expected values of truncated distributions
e1, e2, e3 = integrate(lambda x: x*lognorm.pdf(x)/n1, min, mid, dx), integrate(lambda x: x*lognorm.pdf(x)/n2, mid, max, dx), integrate(lambda x: x*lognorm.pdf(x)/n3, max, total, dx)

print("E[x|" + str(round(lognorm.cdf(min), 3)) + "<p<" + str(round(lognorm.cdf(mid), 3)) + "]", round(e1, 5), "E[x|" + str(round(lognorm.cdf(mid), 3)) + "<p<" + str(round(lognorm.cdf(max), 3)) + "]", round(e2, 5), "E[p>" + str(round(lognorm.cdf(max), 3)) + "]", round(e3, 5))
print("Median:", round(lognorm.ppf(0.5), 5))
print("~E[x|p<" + str(round(lognorm.cdf(max), 3)) + "]", round((n1*e1 + n2*e2)/(n1 + n2), 5))
print("~E[x]", round(n1*e1 + n2*e2 + n3*e3, 5))
print("E[x]", round(lognorm.mean(), 5))
