from matplotlib import pyplot
import statstock

import numpy

#symbols = numpy.array([0.95, 0.97, 0.98, 0.99, 0.995, 0.9975, 1.0, 1.0025, 1.005, 1.01, 1.02, 1.03, 1.05])
symbols = numpy.linspace(0.975, 1.025, 21)
#symbols = numpy.linspace(0.99, 1.01, 11)
#symbols = numpy.linspace(0.99, 1.01, 3)
data = statstock.data.Yahoo("./data/SPY.csv", "spy")
unbinned = data.data["time"], data.price_to_relative(data.data["open"])
binner = statstock.data.Binning(symbols, n=1)

binned = binner.bin_data(unbinned[0], unbinned[1], product=True)

#pyplot.scatter(unbinned[0], unbinned[1])
#pyplot.plot(unbinned[0], unbinned[1])
#pyplot.plot(binned[0], symbols[binned[1]])
#pyplot.show()

mk = statstock.markov.Markov(binned[1], symbols, order=2)
A = numpy.zeros(shape=(len(symbols), len(symbols)))
for i in range(len(symbols)):
    for j in range(len(symbols)):
        if mk._key([i, j]) in mk.data:
            A[i, j] = mk.data[mk._key([i, j])]

A = A/A.max()
A = numpy.swapaxes(A, 0, 1)
pyplot.imshow(A, origin="lower")
pyplot.xlabel("inital")
pyplot.ylabel("final")
pyplot.show()
