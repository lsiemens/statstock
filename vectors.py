import numpy as np

import dataQuest


def makeEVector(candles, mode="parkinson"):
    """Make EVector from questrade client call

    Create EVector from the standard return information of the get_candles and
    get_n_candles methods of QuestradeClient.
    """

    data, interval, symbol = candles
    symbolID, ticker, currency = symbol
    return EVector(data, interval, currency, mode)


class EVector:
    """A vector class for error analysis

    Contains a vector of log mean values and standard deviations. In addition
    there is some basic book keeping information such as the interval.

    Note all prices are converted to CAD.
    """

    USDtoCAD = None

    def __init__(self, data, interval, currency, mode="parkinson"):
        self.interval = interval

        self.vector = np.empty(shape=(len(data), 2))
        self.vector[:, 0] = np.log(data[:, 0])
        self.vector[:, 1] = self._find_errors(data, mode)

        match currency:
            case "CAD":
                pass
            case "USD":
                if self.USDtoCAD is None:
                    EVector.USDtoCAD = float(input("Exchange rate USD to CAD:"))
                self.vector[:, 0] += np.log(self.USDtoCAD)
            case _:
                raise NotImplementedError(f"The mode {mode} has not been implemented.")

    def _find_errors(self, candles, mode):
        """Estimate errors from OHLC

        Assume the vector is of the shape (n, 6) with elements
        [VWAP, open, high, low, close, volume].

        The available modes are:
        - parkinson:
        - garman-klass:
        - rogers-satchell:
        """

        O, H, L, C = candles[:, 1], candles[:, 2], candles[:, 3], candles[:, 4]
        errors = None
        match mode:
            case "parkinson":
                errors = np.log(H/L)/(2*np.sqrt(np.log(2)))
            case "garman-klass":
                errors = np.sqrt((np.log(H/L))**2/2
                                 - (2*np.log(2) - 1)*(np.log(C/O))**2)
            case "rogers-satchell":
                errors = np.sqrt(np.log(H/O)*np.log(H/C)
                                 + np.log(L/O)*np.log(L/C))
            case _:
                raise NotImplementedError(f"The mode {mode} has not been implemented.")

        # correction from sample variance to VWAP err
        errors = errors/np.sqrt(3)
        return errors


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n = 52*5
    ticker = "TSLA"
    interval = "OneWeek"

    clinet = dataQuest.QuestradeClient()
    data = clinet.get_n_candles(ticker, n, interval)
    evector = makeEVector(data, "rogers-satchell")

    logprice, logerr = evector.vector[:, 0], evector.vector[:, 1]

    plt.plot(np.exp(logprice), "k-", label=f"{ticker}, " + "$\\bar{p}$")
    plt.fill_between(np.arange(len(logprice)),
                     np.exp(logprice - logerr),
                     np.exp(logprice + logerr),
                     color="k", alpha=0.5,
                     label=f"{ticker}, " + "$\\bar{p} \\pm \\sigma$")
    plt.title(ticker)
    plt.xlabel(interval[3:])
    plt.ylabel("Price in (CAD)")
    plt.legend()
    plt.show()
