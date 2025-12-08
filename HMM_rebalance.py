"""
System architecture, a hidden markov model using states to switch multivariate
gaussian models of returns.

HMM Transition Matrix
---------------------
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

Bayesian Gaussian, NIW
----------------------
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

Bayesian Category, Dirichlet
----------------------------
Samaples are assumed to be selected from k categories with probability p. The
prior distirbution of p is assumed to be a Dirichlet distribution.

x | p ~ Cat(p)
p ~ Dir(alpha)

where each component of p is in (0, 1) and sum(p) = 1 making it a valid descrete
probability and each component of alpha is positive.

the mean and variance of p sampled from the Dirchlet distirbution is

E[p_i] = alpha_i/sum(alpha)
Var[p_i] = E[p_i](1 - E[p_i])/(sum(alpha) + 1)

As the conjugate prior of the Category distribution the posterior will also be a
Dirichlet distribution.

(p | Vec(x)) ~ Dir(alpha_n)

where the posterior parameters are updated from the prior accoring to the
following rules where x is a vector of n samples, and c dimension k vector of
the occurrences of each category in the data x.

alpha_in = alpha_i + c_i

So the prior parameters alpha act like a vector of prior observed occurrences.

Bayesian Matrix Category, Matrix Dirichlet
------------------------------------------
Taking the previous Category, Dirichlet model an combining k copies gives a
bayesian model for markov chains. A markov chain is defined by the transition
matrix where (for left transition matrices) each column represents the
transitions from the state represented by the column to the states represented
by the rows. So each column of the transition matrix represents a seperate
Category distribution and we will use an acopanying Dirichlet distribution. The
logic for bayesian Category, Dirichlet remains the same but applied column wise
on the model parameter A (the transition matrix) and the matrices alpha/alpha_n
(the prior parameter and posterior parameter)

Bayesian Hidden Markov Model
----------------------------
use modified baum-welch algorithm. Use the prior to estimate the gamma_i values
(probability of the element beloging to a given state), then update each
bayesian model for the a given state with data weighted by said state. then
iterate

"""

import numpy as np
import scipy.stats

import rebalance
import portfolio
import forcast

from matplotlib import pyplot as plt

class HMMRebalance(rebalance.Rebalance):
    """HMM MPT rebalancing

    Use a hidden markov model to model general market trends. Calculate expected
    future log returns using bayesian statistics for both lognormal distribution
    and the markov model.
    """

    def initialize(self, states=2):
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

        # TODO add reasoned prior for alpha
        self.states = states
        self.alpha = 100*np.identity(self.states) + np.random.default_rng().uniform(0, 1, (self.states, self.states))

        self.i_nu_0 = [periods_year/self.states + self.width - 1]*self.states

        self.i_lambda_0 = [periods_year/self.states]*self.states

        # corresponds to a 6.7% return
        self.i_mu_0 = [(0.065/periods_year)*np.ones(self.width)]*self.states # scale to the interval
        self.i_mu_0[0] = -1*self.i_mu_0[0] # TODO remove

        # corresponds to a 15.4% standard deviation
        varience = 0.02/periods_year # scale to the interval
        correlation = 0.2
        #CovlnRet_0 = Psi_0/(nu_0 - d - 1)
        Cov_0 = varience*(correlation*np.ones((self.width, self.width)) + (1 - correlation)*np.identity(self.width))
        self.i_Psi_0 = [(nu_0 - self.width - 1)*Cov_0 for nu_0 in self.i_nu_0]


        self.i_nu_0 = np.array(self.i_nu_0)
        self.i_lambda_0 = np.array(self.i_lambda_0)
        self.i_mu_0 = np.array(self.i_mu_0)
        self.i_Psi_0 = np.array(self.i_Psi_0)

        self._theta = None # posterior parameters

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

    def forward_backward(self, lnRet, theta):
        # if sampling from the posterior then solution tolerance should be set
        # by the expected sample variance
        #sample_A, i_sample_mu, i_sample_Sigma = self.sample_posterior(theta)
        alpha_k, _, i_nu_k, i_mu_k, i_Psi_k = theta

        # Calculated expected A, mu, Sigma
        sample_A = alpha_k/np.sum(alpha_k, axis=0)[None, :]
        i_sample_mu = i_mu_k
        i_sample_Sigma = i_Psi_k/(i_nu_k[:, None, None] - self.width - 1)

        ln_sample_A = np.log(sample_A)
        ln2pi = np.log(2*np.pi)
        i_ln_det_Sigma = np.log(np.linalg.det(i_sample_Sigma))
        i_inv_Sigma = np.linalg.inv(i_sample_Sigma)

        def lnMN(x):
            prefactor = self.width*ln2pi + i_ln_det_Sigma
            inner_product = np.einsum("ij,ijk,ik->i", (x - i_sample_mu), i_inv_Sigma, (x - i_sample_mu))
            return -0.5*(prefactor + inner_product)

        N = lnRet.shape[1]

        steady_state = self.get_steady_state(sample_A)

        i_ln_alpha = np.empty((N, self.states))
        i_ln_beta = np.empty((N, self.states))

        i_ln_alpha[0, :] = np.log(steady_state) + lnMN(lnRet[:, 0])
        for i in range(N - 1):
            exponent = ln_sample_A + i_ln_alpha[i, :][:, None]
            index = np.argmax(exponent.flatten())
            offset = exponent.flatten()[index]
            i_ln_alpha[i + 1, :] = lnMN(lnRet[:, i + 1]) + np.log(np.sum(np.exp(exponent - offset), axis=0)) + offset

        i_ln_beta[-1, :] = 0
        for i in range(N - 1, 0, -1):
            exponent = i_ln_beta[i, :] + ln_sample_A + lnMN(lnRet[:, i])
            index = np.argmax(exponent.flatten())
            offset = exponent.flatten()[index]
            i_ln_beta[i - 1, :] = np.log(np.sum(np.exp(exponent - offset), axis=1)) + offset

        i_ln_gamma = i_ln_alpha + i_ln_beta
        for i in range(N):
            exponent = i_ln_gamma[i, :]
            index = np.argmax(exponent.flatten())
            offset = exponent.flatten()[index]
            i_ln_gamma[i, :] = i_ln_gamma[i, :] - (np.log(np.sum(np.exp(i_ln_gamma[i, :] - offset))) + offset)

        ln_xi = np.empty((N - 1, self.states, self.states))
        for i in range(N - 1):
            ln_xi[i] = i_ln_alpha[i, :][:, None] + ln_sample_A + i_ln_beta[i + 1, :] + lnMN(lnRet[:, i + 1])
            exponent = ln_xi[i]
            index = np.argmax(exponent.flatten())
            offset = exponent.flatten()[index]
            ln_xi[i] = ln_xi[i] - (np.log(np.sum(np.exp(exponent - offset))) + offset)

        return np.exp(i_ln_gamma), np.exp(ln_xi)

    def _isclose(self, theta, theta_0, rtol=1e-5, atol=1e-8):
        n_parameters = np.sum([np.prod(parameter.shape) for parameter in theta])
        p_norm_0 = np.array([np.linalg.norm(parameter) for parameter in theta_0])
        delta_p_norm = np.array([np.linalg.norm(theta[i] - theta_0[i]) for i in range(len(theta))])

        norm_0 = np.sqrt(np.sum(p_norm_0**2))
        delta_norm = np.sqrt(np.sum(delta_p_norm**2))

        print("Norms", delta_norm, delta_norm/norm_0)
        return delta_norm < (atol + rtol*norm_0)*np.sqrt(n_parameters)

    def get_posterior_parameters(self):
        if self._theta is None:
            self._theta = self.find_posterior_parameters()
            return self._theta
        else:
            return self._theta

    def find_posterior_parameters(self):
        mask = np.isfinite(self.logprice).all(axis=0)
        lnRet = np.diff(self.logprice[:, mask], axis=1)

        # current best estimate of posterior parameters
        theta = (self.alpha, self.i_lambda_0, self.i_nu_0, self.i_mu_0, self.i_Psi_0)

        print("Solve for posterior")
        max_depth = 50
        for _ in range(max_depth):
            # calculate weights
            i_gamma, xi = self.forward_backward(lnRet, theta)
            theta_0 = theta

            # calculate weighted statistics
            i_N = np.sum(i_gamma, axis=0)
            Xi = np.sum(xi, axis=0)
            i_x_bar = (lnRet @ i_gamma).T/i_N[:, None]
            i_S = np.empty((self.states, self.width, self.width))
            for state in range(self.states):
                diff = (lnRet - i_x_bar[state][:, None])
                i_S[state] = (diff * i_gamma[:, state]) @ diff.T

            # Bayesian update of the posterior
            alpha_k = self.alpha + Xi
            i_lambda_k = self.i_lambda_0 + i_N
            i_nu_k = self.i_nu_0 + i_N
            i_mu_k = (self.i_lambda_0[:, None]*self.i_mu_0 + i_N[:, None]*i_x_bar)/i_lambda_k[:, None]
            i_Psi_k = np.empty((self.states, self.width, self.width))
            for state in range(self.states):
                prefactor = (self.i_lambda_0[state]*i_N[state]/i_lambda_k[state])
                diff = (i_x_bar[state] - self.i_mu_0[state])
                cross_term = prefactor*(diff @ diff.T)
                i_Psi_k[state] = self.i_Psi_0[state] + i_S[state] + cross_term

            theta = (alpha_k, i_lambda_k, i_nu_k, i_mu_k, i_Psi_k)
            if self._isclose(theta, theta_0):
                return theta

    def sample_posterior(self, theta):
        """Sample model parameters from the provided posterior parameters
        """
        alpha, i_lambda, i_nu, i_mu, i_Psi = theta

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
        sample_A = np.empty((self.states, self.states))
        for state in range(self.states):
            sample_A[:, state] = self.rng.dirichlet(alpha[:, state])

        return sample_A, i_sample_mu, i_sample_Sigma

    def sample_lnRet(self, n_intervals):
        theta = self.get_posterior_parameters()
        sample_A, i_sample_mu, i_sample_Sigma = self.sample_posterior(theta)

        # choose the starting state
        steady_state = self.get_steady_state(sample_A)
        inital_state = self.rng.choice(self.states, p=steady_state)

        # sample state chain
        state_chain = np.empty(n_intervals, dtype=int)
        state_chain[0] = inital_state
        for i in range(1, n_intervals):
            state = state_chain[i - 1]
            state_chain[i] = self.rng.choice(self.states, p=sample_A[:, state])

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

        alpha, i_lambda, i_nu, i_mu, i_Psi = self.get_posterior_parameters()

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
    #p_0 = portfolio.Portfolio("./long_holdings.csv", 52*30, "OneWeek")
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
