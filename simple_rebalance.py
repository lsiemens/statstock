import numpy as np

import rebalance
import portfolio


class SimpleRebalance(rebalance.Rebalance):
    """Use simple means to estimate returns
    """

    def market_statistics(self):
        match self.interval:
            case "OneDay":
                periods_year = 52*5
            case "OneWeek":
                periods_year = 52
            case _:
                raise NotImplimentedError(f"The interval {self.interval} is not implimented")

        mask = np.isfinite(self.logprice).all(axis=0)
        lnRet = np.diff(self.logprice[:, mask], axis=1)

        ElnRet = np.nanmean(lnRet, axis=1)
        CovlnRet = np.cov(lnRet)
        VarlnRet = np.diag(CovlnRet)

        ElnRet_err = np.sqrt(VarlnRet/len(lnRet))
        Var_CovlnRet = (VarlnRet**2 + np.outer(VarlnRet, VarlnRet))/len(ElnRet)
        CovlnRet_err = np.sqrt(Var_CovlnRet)

        ElnRet = ElnRet*periods_year
        ElnRet_err = ElnRet_err*np.sqrt(periods_year)

        CovlnRet = CovlnRet*periods_year
        CovlnRet_err = CovlnRet_err*np.sqrt(periods_year)
        return (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err), lnRet.shape[1]


if __name__ == "__main__":
    #p_0 = portfolio.Portfolio("./all_holdings.csv", 365*2, "OneDay")
    p_0 = portfolio.Portfolio("./all_holdings.csv", 52*15, "OneWeek")
    MPT = SimpleRebalance(p_0, ["APPL", "ARM", "IBIT", "U", "DJT", "DGRC.TO", "LHX"])
    MPT.data_info()
    MPT.show_market_statistics()

    def g(w):
        return 1 - np.sum(w)

    gamma = MPT.get_gamma()

    def U(weight):
        (mu, _), (Sigma, _), _ = MPT.market_statistics()
        return -(MPT.utility(mu, Sigma, weight, gamma) + (0.1/MPT.width)/np.sum(weight**2))

    bounds = [(0.0, 1.0) for i in range(MPT.width)]
    weights, result = MPT.solver(U, eq_consts=[g], bounds=bounds)

    print(f"{result.message} after {result.nit} iterations with utility u(weight) = {-100*result.fun:.1f}%")

    MPT.show_portfolio_statistics(weights)
