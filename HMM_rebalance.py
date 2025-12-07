import numpy as np
import scipy.stats

import rebalance
import portfolio
import forcast


class HMMRebalance(rebalance.Rebalance):
    """HMM MPT rebalancing

    Use a hidden markov model to model general market trends. Calculate expected
    future log returns using bayesian statistics for both lognormal distribution
    and the markov model.

    This system can be factored into the hidden markov model and the state
    dependent sampling model. The markov model is defined by a left transition
    matrix A (left or right are equal upto transpostion of the equations), the
    components A_i are in the range (0, 1) and each column sums to one. A
    probablistic state v is defined as a column vector with element in (0, 1) that
    sums to one. Then given state vector after i - 1 transitions the ith state
    vector is,

    v_i = A v_i

    and given a row vector vI with each element equal to one, the normalization
    constraint on A can be expressed as,

    vI A = vI

    The eigenvectors of A describe stable components of probability
    distributions (they are not nessisarily valid probability distributions
    themselves) and the eigenvalues of A gaurenteed to land in the unit disk
    with atleast one eigenvalue of one. If there are multiple eigenvalues that
    are roots of unity then there will be multiple stable distributions (ex from
    disconnected graphs, or rotating through the states). If there is a unique
    eigenvector with eigenvalue of one, then that vector is the stable
    probability distibution from any initial vector in the limit as the number
    of stransitions approches infinity.

    Prices are assumed to be log normal dependent random variables. The log returns for n
    tickers is assumed to be distributed as an n-dimensional multivariate normal
    distribution with mean mu and covariance Sigma. The prior distribution of
    (mu, Sigma) is assumed to be a Normal-Inverse Wishart distribution for
    simplicity.

    x | mu, Sigma ~ N(mu, Sigma)
    (mu, Sigma) ~ NIW(mu_0, lambda_0, Psi_0, nu_0) = N(mu | mu_0, Sigma/lambda_0) IW(Sigma | Psi_0, nu_0)

    There are restrictions on the value of nu_0 based on the dimension of the
    problem with n = dim(mu_0) then nu_0 > n - 1 also note lambda_0 > 0

    The mean mu and sigma sampled from the NIW prior is given by

    mean(mu) = mu_0
    mean(psi) = Psi_0/(nu_0 - n - 1)

    The NIW distribution is the conjugate prior of the multivariate normal
    distribution so the posterior is also distributed as a NIW.

    (mu, Sigma | Vec(x)) ~ NIW(mu_k, lambda_k, psi_k, nu_k)

    Where the posterior parameters are updated from the prior according to the
    following rules where x is a vector of k samples.

    lambda_k = lambda_0 + k
    nu_k = lambda_0 + k
    mu_k = (lambda_0*mu_0 + k*Mean(x))/lambda_k
    psi_k = psi_0 + k*Cov(x, x) + (lambda_0*k/lambda_k)*(bar(x) - mu_0)*(bar(x) - mu_0)

    Like with the prior the mean from the posterior distribution is given by,

    Mean(mu | x) = mu_k
    Mean(Sigma | x) = psi_k / (nu_k - n - 1)

    In the limit of a large number of samples the mean and standard error of mu
    and Sigma will tend to the same values as in the SimpleRebalance class.

    https://isdsa.org/jbds/fulltext/v1n2/p2/
    """

    def initialize(self, states=3):
        """Initialize prior

        Set Normal Inverse-Wishart prior. If a parameter is left as None then
        that parameter is replaced with one corresponding to a relatively
        uninformative prior.
        """
        match self.interval:
            case "OneDay":
                periods_year = 52*5
            case "OneWeek":
                periods_year = 52
            case _:
                raise NotImplimentedError(f"The interval {self.interval} is not implimented")

        self.states = states
        self.alpha = np.random.default_rng().uniform(0, 10, (self.states, self.states))
        self.alpha_0 = np.sum(self.alpha, axis=0)

        # find mean transition matrix
        A = self.alpha/np.sum(self.alpha, axis=0)[None, :]
        self.get_steady_state(A)

        self.i_nu_0 = [periods_year + self.width - 1]*self.states

        self.i_lambda_0 = [periods_year]*self.states

        # corresponds to a 6.7% return
        self.i_mu_0 = [(0.065/periods_year)*np.ones(self.width)]*self.states # scale to the interval
        self.i_mu_0[0] = -1*self.i_mu_0[0] # TODO remove

        # corresponds to a 15.4% standard deviation
        varience = 0.02/periods_year # scale to the interval
        correlation = 0.2
        #CovlnRet_0 = Psi_0/(nu_0 - d - 1)
        Cov_0 = varience*(correlation*np.ones((self.width, self.width)) + (1 - correlation)*np.identity(self.width))
        self.i_Psi_0 = [(nu_0 - self.width - 1)*Cov_0 for nu_0 in self.i_nu_0]

    def get_steady_state(self, A):
        """
        Raise ValueError if the transition matrix is invalid or if it is not
        unique
        returns the unique steady state distibution 
        """
        # check if A is a valid transition matrix
        if np.any(A < 0) or np.any(A > 1):
            raise ValueError("All values in the transition matrix must be in (0, 1)")
        if not np.all(np.isclose(np.sum(A, axis=0), 1)):
            raise ValueError("All columns in the transition matrix must sum to one")

        # check if A has a unique steady state distribution
        eigen_values, eigen_vectors = np.linalg.eig(A)
        if np.sum(np.isclose(np.abs(eigen_values), 1)) != 1:
            raise ValueError("The transition matrix does not have a unique steady state")

        # find the steady state distribution
        index = np.argmax(np.abs(eigen_values))
        steady_state = eigen_vectors[:, index]
        steady_state = steady_state/np.sum(steady_state)
        if not np.all(np.isclose(np.imag(steady_state), 0)):
            raise ValueError("The steady state solution has a nonzero imaginary component")
        else:
            steady_state = np.real(steady_state)

        return steady_state

    def find_posterior_parameters(self):
        """
        mask = np.isfinite(self.logprice).all(axis=0)
        lnRet = np.diff(self.logprice[:, mask], axis=1)

        # Find mean and scatter
        x_bar = np.mean(lnRet, axis=1)
        S = (lnRet - x_bar[:, None]) @ (lnRet - x_bar[:, None]).T
        n = lnRet.shape[1]

        # Bayesian update: find parameters of posterior NIW distribution
        lambda_k = self.lambda_0 + n
        nu_k = self.nu_0 + n
        mu_k = (self.lambda_0*self.mu_0 + n*x_bar)/lambda_k
        Psi_k = self.Psi_0 + S + (self.lambda_0*n/lambda_k)*((x_bar - self.mu_0) @ (x_bar - self.mu_0).T)
        return lambda_k, nu_k, mu_k, Psi_k
        """
        return self.alpha, self.i_lambda_0, self.i_nu_0, self.i_mu_0, self.i_Psi_0

    def sample_lnRet(self, n_intervals):
        alpha, i_lambda, i_nu, i_mu, i_Psi = self.find_posterior_parameters()

        # sample the Gaussian model parameters per state
        i_sample_Sigma = np.empty((self.states, self.width, self.width))
        i_sample_mu = np.empty((self.states, self.width))
        for state in range(self.states):
            i_sample_Sigma[state] = scipy.stats.invwishart.rvs(i_nu[state], i_Psi[state], random_state=self.rng)
            i_sample_mu[state] = self.rng.multivariate_normal(i_mu[state], i_sample_Sigma[state]/i_lambda[state])

            if not self.check_SPSD(i_sample_Sigma[state]):
                print("The sampled Sigma matrix required correction")
                i_sample_Sigma[state] = self.fix_SPSD(i_sample_Sigma[state])

        # sample the state Categorical parameters
        A = np.empty((self.states, self.states))
        for state in range(self.states):
            A[:, state] = self.rng.dirichlet(alpha[:, state])

        # choose the starting state
        steady_state = self.get_steady_state(A)
        inital_state = self.rng.choice(self.states, p=steady_state)

        # sample state chain
        state_chain = np.empty(n_intervals, dtype=int)
        state_chain[0] = inital_state
        for i in range(1, n_intervals):
            state = state_chain[i - 1]
            state_chain[i] = self.rng.choice(self.states, p=A[:, state])

        # count samples in each state
        samples_in_state = np.empty(self.states, dtype=int)
        for state in range(self.states):
            samples_in_state[state] = np.sum(state_chain == state)

        # sample each state distribution
        sample_lnRet = np.empty((self.width, n_intervals))
        for state in range(self.states):
            gaussian_samples = self.rng.multivariate_normal(i_sample_mu[state],
                                                            i_sample_Sigma[state],
                                                            samples_in_state[state])
            mask = (state_chain == state)
            sample_lnRet[:, mask] = gaussian_samples.T

        return sample_lnRet

    def market_state_statistics(self, lambda_k, nu_k, mu_k, Psi_k):
        """Per state market statistics
        """
        d = self.width

        # Calculated expected mu, Sigma and their errors
        ElnRet = mu_k
        CovlnRet = Psi_k/(nu_k - d - 1)

        ElnRet_err = np.sqrt(np.diag(Psi_k)/(lambda_k*(nu_k - d + 1)))
        _off_axis = (nu_k - d + 1)*Psi_k**2
        _diagonal = (nu_k - d - 1)*np.outer(np.diag(Psi_k), np.diag(Psi_k))
        CovlnRet_err = np.sqrt((_off_axis + _diagonal)/((nu_k - d)*(nu_k - d - 1)**2*(nu_k - d - 3)))

        return (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err)

    def market_statistics(self, annualized=True):
        match self.interval:
            case "OneDay":
                periods_year = 52*5
            case "OneWeek":
                periods_year = 52
            case _:
                raise NotImplimentedError(f"The interval {self.interval} is not implimented")

        alpha, i_lambda, i_nu, i_mu, i_Psi = self.find_posterior_parameters()

        i_ElnRet = np.empty((self.states, self.width))
        i_ElnRet_err = np.empty((self.states, self.width))
        i_CovlnRet = np.empty((self.states, self.width, self.width))
        i_CovlnRet_err = np.empty((self.states, self.width, self.width))

        for state in range(self.states):
            stats = self.market_state_statistics(i_lambda[state], i_nu[state],
                                                 i_mu[state], i_Psi[state])
            i_ElnRet[state] = stats[0][0]
            i_ElnRet_err[state] = stats[0][1]
            i_CovlnRet[state] = stats[1][0]
            i_CovlnRet_err = stats[1][1]

        A = alpha/np.sum(alpha, axis=0)[None, :]
        # TODO estimate sampled mean steady state instead of steady state of mean sample
        steady_state = self.get_steady_state(A)

        # TODO include variance in the steady state
        ElnRet = steady_state @ i_ElnRet
        ElnRet_err = steady_state @ i_ElnRet_err
        CovlnRet = np.sum(steady_state[:, None, None]*i_CovlnRet, axis=0)
        CovlnRet_err = np.sum(steady_state[:, None, None]*i_CovlnRet_err, axis=0)

        # annualize the mean and covarience
        if annualized:
            ElnRet = ElnRet*periods_year
            ElnRet_err = ElnRet_err*np.sqrt(periods_year)

            CovlnRet = CovlnRet*periods_year
            CovlnRet_err = CovlnRet_err*np.sqrt(periods_year)

        return (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err)

if __name__ == "__main__":
    #p_0 = portfolio.Portfolio("./all_holdings.csv", 365*2, "OneDay")
    p_0 = portfolio.Portfolio("./all_holdings.csv", 52*15, "OneWeek")
    MPT = HMMRebalance(p_0, ["APPL", "ARM", "IBIT", "U", "DJT", "DGRC.TO", "LHX", "GME"])
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
    data = prediction.n_forcasts(52*10, 1000)
    prediction.show_n_forcasts(*data)
