import numpy as np

import rebalance
import portfolio
import forcast


class SimpleRebalance(rebalance.Rebalance):
    """Simple MPT rebalancing

    Calculate expected future log returns as the simple mean of past log
    returns. Likewise calculate the covariance matrix as the covariance of the
    past log returns.
    """

    def sample_lnRet(self, n_intervals):
        (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err) = self.market_statistics(annualized=False)

        sample_mu = self.rng.normal(ElnRet, ElnRet_err)
        sample_Sigma = self.rng.normal(CovlnRet, CovlnRet_err)

        if not self.check_SPSD(sample_Sigma):
          sample_Sigma = self.fix_SPSD(sample_Sigma)

        sample_lnRet = self.rng.multivariate_normal(sample_mu, sample_Sigma, n_intervals)
        sample_lnRet = sample_lnRet.T
        return sample_lnRet

    def market_statistics(self, annualized=True):
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

        if annualized:
            ElnRet = ElnRet*periods_year
            ElnRet_err = ElnRet_err*np.sqrt(periods_year)

            CovlnRet = CovlnRet*periods_year
            CovlnRet_err = CovlnRet_err*np.sqrt(periods_year)

        return (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err)


if __name__ == "__main__":
    #p_0 = portfolio.Portfolio("./all_holdings.csv", 365*2, "OneDay")
    p_0 = portfolio.Portfolio("./all_holdings.csv", 52*15, "OneWeek")
    MPT = SimpleRebalance(p_0, ["APPL", "ARM", "IBIT", "U", "DJT", "DGRC.TO", "LHX", "GME"])
    MPT.data_info()
    #MPT.show_market_statistics()

    def g(w):
        return 1 - np.sum(w)

    gamma = MPT.get_gamma()

    def U(weight):
        (mu, _), (Sigma, _) = MPT.market_statistics()
        return -(MPT.utility(mu, Sigma, weight, gamma) + (0.2/MPT.width)/np.sum(weight**2))

    bounds = [(0.0, 1.0) for i in range(MPT.width)]
    weights = MPT.solver(U, eq_consts=[g], bounds=bounds)

    MPT.show_portfolio_statistics(weights)

    def cash_flow(year):
        if year > 65:
            return -25000
        else:
            return 0.1*np.maximum(40000*(year - 10)/55, 0)

    prediction = forcast.Forcast(MPT, weights, V_0=1000, cash_flow=cash_flow)
    #data = prediction.single_forcast(52*8 + 26)
    #prediction.show_single_forcast(*data)
    data = prediction.n_forcasts(52*90, 1000)
    prediction.show_n_forcasts(*data)
