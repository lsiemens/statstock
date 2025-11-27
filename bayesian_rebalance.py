import numpy as np

import rebalance
import portfolio
import forcast


class BayesianRebalance(rebalance.Rebalance):
    """Bayesian MPT rebalancing

    Calculate expected future log returns using bayesian statistics. Prices are
    assumed to be log normal dependent random variables. The log returns for n
    tickers is assumed to be distributed as an n-dimensional multivariate normal
    distribution with mean mu and covariance Sigma. The prior distribution of
    (mu, Sigma) is assumed to be a Normal-Inverse Wishart distribution for
    simplicity (it is the conjugate prior of the multivariate normal
    distribution).

    x | mu, Sigma ~ N(mu, Sigma)
    (mu, Sigma) ~ NIW(mu_0, lambda, psi, nu) = N(mu | mu_0, Sigma/lambda) IW(Sigma | psi, nu)

    dim(mu_0) = n
    lambda > 0, nu > n - 1

    mean(psi) = psi/(nu - n - 1)

    poasterior (mu, Sigma | Vec(x)) ~ NIW(mu_k, lambda_k, psi_k, nu_k)

    lambda_k = lambda_0 + k
    nu_k = lambda_0 + k
    mu_k = (lambda_0 mu_0 + k Mean(x))/lambda_k
    psi_k = psi_0 + k*Cov(x, x) + (lambda_0*k/lambda_k)*(bar(x) - mu_0)*(bar(x) - mu_0)

    Mean(mu | x) = mu_k
    Mean(Sigma | x) = psi_k / (nu_k - n - 1)


    In the limit of a large number of samples the mean and standard error of mu
    and Sigma will tend to the same values as in the SimpleRebalance class.

    """

    def initialize(self, nu_0=None, lambda_0=None, mu_0=None, Psi_0=None):
        """Initialize prior

        Set Normal Inverse-Wishart prior. If a parameter is left as None then
        that parameter is replaced with one corresponding to a relatively
        uninformative prior.
        """
        epsilon = 1e-3

        if nu_0 is None:
            self.nu_0 = self.width + 2
        else:
            if not (nu_0 > self.width - 1):
                raise ValueError(f"For convergent results nu_0 > n - 1, but nu_0 = {nu_0:.1f} <= {self.width - 1} was given.")
            self.nu_0 = nu_0

        if lambda_0 is None:
            self.lambda_0 = epsilon
        else:
            if not (lambda_0 > 0):
                raise ValueError(f"For convergent results lambda_0 > 0, but lambda_0 = {lambda_0:.1f} was given.")
            self.lambda_0 = lambda_0

        if mu_0 is None:
            self.mu_0 = np.zeros(self.width)
        else:
            self.mu_0 = mu_0

        if Psi_0 is None:
            self.Psi_0 = epsilon*np.identity(self.width)
        else:
            self.Psi_0 = Psi_0

    # TODO sample from NIW distribution instead of the market_statistics
    def sample_lnRet(self, n_intervals):
         (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err), _ = self.market_statistics(annualized=False)

         sample_mu = self.rng.normal(ElnRet, ElnRet_err)
         sample_Sigma = self.rng.normal(CovlnRet, CovlnRet_err)

         if not self.check_SPSD(sample_Sigma):
           sample_Sigma = self.fix_SPSD(sample_Sigma)

         sample_lnRet = self.rng.multivariate_normal(sample_mu, sample_Sigma, n_intervals)
         sample_lnRet = sample_lnRet.T
         return sample_lnRet

    # TODO validate equation for the posterior, expected mu and Sigma and the
    # equations for the standard error
    def market_statistics(self, annualized=True):
        match self.interval:
            case "OneDay":
                periods_year = 52*5
            case "OneWeek":
                periods_year = 52
            case _:
                raise NotImplimentedError(f"The interval {self.interval} is not implimented")

        self.logprice.astype(np.float128)
        mask = np.isfinite(self.logprice).all(axis=0)
        lnRet = np.diff(self.logprice[:, mask], axis=1)

        # Find mean and scatter
        x_bar = np.mean(lnRet, axis=1)
        S = (lnRet - x_bar[:, None]) @ (lnRet - x_bar[:, None]).T
        n = lnRet.shape[1]
        d = self.width

        # Bayesian update: find parameters of posterior NIW distribution
        lambda_k = self.lambda_0 + n
        nu_k = self.nu_0 + n
        mu_k = (self.lambda_0*self.mu_0 + n*x_bar)/lambda_k
        Psi_k = self.Psi_0 + S + (self.lambda_0*n/lambda_k)*((x_bar - self.mu_0) @ (x_bar - self.mu_0).T)

        # Calculated expected mu, Sigma and their errors
        ElnRet = mu_k
        CovlnRet = Psi_k/(nu_k - d - 1)

        ElnRet_err = np.sqrt(np.diag(Psi_k)/(lambda_k*(nu_k - d + 1)))
        _off_axis = (nu_k - d + 1)*Psi_k**2
        _diagonal = (nu_k - d - 1)*np.outer(np.diag(Psi_k), np.diag(Psi_k))
        CovlnRet_err = np.sqrt((_off_axis + _diagonal)/((nu_k -d)*(nu_k - d - 1)**2*(nu_k - d - 3)))

        # annualize the mean and covarience
        if annualized:
            ElnRet = ElnRet*periods_year
            ElnRet_err = ElnRet_err*np.sqrt(periods_year)

            CovlnRet = CovlnRet*periods_year
            CovlnRet_err = CovlnRet_err*np.sqrt(periods_year)

        return (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err), lnRet.shape[1]

if __name__ == "__main__":
    #p_0 = portfolio.Portfolio("./all_holdings.csv", 365*2, "OneDay")
    p_0 = portfolio.Portfolio("./all_holdings.csv", 52*15, "OneWeek")
    MPT = BayesianRebalance(p_0, ["APPL", "ARM", "IBIT", "U", "DJT", "DGRC.TO", "LHX"])
    MPT.data_info()
    #MPT.show_market_statistics()

    def g(w):
        return 1 - np.sum(w)

    gamma = MPT.get_gamma()

    def U(weight):
        (mu, _), (Sigma, _), _ = MPT.market_statistics()
        return -(MPT.utility(mu, Sigma, weight, gamma) + (0.1/MPT.width)/np.sum(weight**2))

    bounds = [(0.0, 1.0) for i in range(MPT.width)]
    weights = MPT.solver(U, eq_consts=[g], bounds=bounds)

    MPT.show_portfolio_statistics(weights)

    prediction = forcast.Forcast(MPT, weights)

    values = prediction.n_forcasts(52*15, 1000)
    prediction.show_n_forcasts(values)
