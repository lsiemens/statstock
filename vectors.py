import numpy as np

import dataQuest

# Vector (np.ndarray(shape=(n, 2)), interval in ["OneDay", "OneWeek"])

def makeEVector(candles, mode="parkinson"):
    data, interval, symbol = candles
    symbolID, ticker, currency = symbol
    return EVector(data, interval, currency, mode)

class EVector:
    """A vector class for error analysis

    Contains a vector of log mean values and standard deviations. In addition there is
    some basic book keeping information such as the interval.
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

        Assume the vector is of the shape (n, 6) with elements [VWAP, open, high,
        low, close, volume].

        The avaliable modes are:
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
                errors = np.sqrt(  (np.log(H/L))**2/2
                                 - (2*np.log(2) - 1)*(np.log(C/O))**2)
            case "rogers-satchell":
                errors = np.sqrt(  np.log(H/O)*np.log(H/C)
                                 + np.log(L/O)*np.log(L/C))
            case _:
                raise NotImplementedError(f"The mode {mode} has not been implemented.")
        errors = errors/np.sqrt(3) # correction from sample variance to VWAP err
        return errors

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    clinet = dataQuest.QuestradeClient()
    ticker = "TSLA"
    interval = "OneWeek"
    candles = clinet.get_n_candles(ticker, 2*52, interval)
    data = makeEVector(candles, "rogers-satchell")
  
    logprice, logerr = data.vector[:, 0], data.vector[:, 1]
    plt.plot(np.exp(logprice), "k-", label = f"{ticker}, " + "$\\bar{p}$")
    plt.fill_between(np.arange(len(data.vector)),
                     np.exp(logprice - logerr),
                     np.exp(logprice + logerr),
                     color="k", alpha=0.5,
                     label = f"{ticker}, " + "$\\bar{p} \\pm \\sigma$")

    plt.legend()
    plt.show()

