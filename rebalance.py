import numpy as np
import scipy
import scipy.optimize
from matplotlib import pyplot as plt


class Rebalance:
    """Parent class for portfolio rebalancing strategies

    Framework for rebalancing schemes based on mean-variance analysis.
    """

    def __init__(self, portfolio, ignore=[]):
        mask = [(ticker not in ignore) for ticker in portfolio.tickers]

        self.width = np.sum(mask)
        self.length = portfolio.length
        self.tickers = [portfolio.tickers[i] for i in range(portfolio.width) if mask[i]]
        self.interval = portfolio.interval
        self.weights = None
        self._Ret_free = None
        self._default_Ret_free = 3.82

        self.logprice = portfolio.logprice[mask, :]
        self.logerror = portfolio.logerror[mask, :]

        n_shares = portfolio.n_shares[mask]

        self.weights = np.empty(self.width)
        for i in range(self.width):
            self.weights[i] = n_shares[i]*np.exp(self.logprice[i, -1])
        self.weights = self.weights/np.mean(self.weights)

        self.rng = np.random.default_rng()

        self.initialize()

    def initialize(self):
        """Custom initialization hook

        Initialize is called at the end of __init__
        """
        pass

    def sample_lnRet(self, n_intervals):
        """Generate samples of lnRet

        The sampled log returns should not be annualized.

        Parameters
        ----------
        n_intervals : int
            Number of samples to return
        
        Returns
        -------
        array(width, n_intervals)
            Random Samples of log returns matching the observed statistics.
        """
        raise NotImplementedError("sample_lnRet must be implemented in a subclass")
  
    def check_SPSD(self, A, rtol=1e-5, atol=1e-8):
        """Check if matrix is symmetric positive semidefinite
        """

        sym_abs_error = np.max(np.abs(A - A.T))
        if not np.isclose(sym_abs_error, 0):
            return False

        A_sym = (A + A.T)/2

        if np.min(np.diag(A_sym)) < 0:
            return False

        diagonal, orthogonal = np.linalg.eigh(A_sym)
        
        if np.min(diagonal) < 0:
            return False

        return True

    def fix_SPSD(self, A):
        sym_A = (A + A.T)/2
        #  diagonal, orthogonal   = ...
        eigenvalues, eigenvectors = np.linalg.eigh(sym_A)
        # note diagonal is a vector not a matrix

        min_eigenvalue = np.min(np.abs(eigenvalues))
        # replace negative eigenvalues with a positive eigenvalue while
        # maintaining the condition number of the matrix
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        # (Orth * Diag) @ Orth.T = Orth @ np.diag(Diag) @ Orth.T
        SPSD_A = (eigenvectors * eigenvalues) @ eigenvectors.T
        return SPSD_A

    def get_Ret_free(self):
        if self._Ret_free is not None:
            return self._Ret_free

        print("\n(FRED 3-month T-bill rate)")

        try:
            self._Ret_free = 0.01*float(input(f"Risk free return rate [{self._default_Ret_free}]: "))
        except ValueError:
            self._Ret_free = 0.01*self._default_Ret_free

        return self._Ret_free

    def market_statistics(self, annualized=True):
        """Generate statistics about the market

        The statistics should include the expected future log return for the
        next interval along with the covariance matrix.

        By default the statistics should be given for annualized returns.

        Parameters
        ----------
        annualized : bool
            Should returned statistics be annualized.

        Returns
        -------
        tuple(Array(width), Array(width))
            ElnRet: Expected future log returns and standard error
        tuple(Array(width, width), Array(width, width))
            CovlnRet: Covariance matrix of the future log returns and standard error
        """
        raise NotImplementedError("`market_statistics` must be implemented in a child class")

    def data_info(self):
        """Show datapoints per ticker
        """
        data_points = np.sum(np.isfinite(self.logprice), axis=1)
        print(f"The time series has {self.logprice.shape[1]} points sampled with a {self.interval} interval")
        print("Trading periods: ")
        data_point_str = ""
        block_size = 6
        for i in range(self.width):
            data_point_str += f"{self.tickers[i]} = {data_points[i]} "
            if i % block_size == (block_size - 1):
                data_point_str += "\n                 "
        print(data_point_str)

    def show_portfolio_statistics(self, weights):
        """Print useful portfolio statistics

        Use a set of weights and the statistics from market_statistics to
        calculate useful metrics for the individual stocks being analyized.
        These include the expected return, Sharpe ratio, Kelly ratio, alpha and
        beta.

        If avaliable VTI or SPY will be used as the market return.
        """

        # Get annuallized returns and covarience matrix
        (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err) = self.market_statistics()

        VarlnRet = np.diag(CovlnRet)
        VarlnRet_err = np.diag(CovlnRet_err)

        market_symbol_id = None
        if "VTI" in self.tickers:
            market_symbol_id = self.tickers.index("VTI")
        if "SPY" in self.tickers:
            market_symbol_id = self.tickers.index("SPY")
        else:
            print("No reference EFT for the whole market")

        # Get the risk free return
        sim_Ret_free = None
        if market_symbol_id is not None:
            sim_Ret_free = self.get_Ret_free()


        print("\nPortfolio statistics:")
        # Get portfolio log returns and simple returns
        P_ElnRet = weights @ ElnRet
        P_ElnRet_err = np.sqrt(np.sum((weights*ElnRet_err)**2))

        P_VarlnRet = weights @ CovlnRet @ weights
        P_VarlnRet_err = np.sqrt(np.sum((np.outer(weights, weights)*CovlnRet_err)**2))

        print(f"Expected mean log return: {P_ElnRet:.3f} ± {P_ElnRet_err:.3f}")
        print(f"Expected log return range: ({(P_ElnRet - np.sqrt(P_VarlnRet)):.3f}, {(P_ElnRet + np.sqrt(P_VarlnRet)):.3f}), given sample standard deviation: {np.sqrt(P_VarlnRet):.3f} ± {(P_VarlnRet_err/(2*np.sqrt(P_VarlnRet))):.3f}")

        P_sim_Ret = np.exp(P_ElnRet + 0.5*P_VarlnRet) - 1
        P_sim_Ret_err = (P_sim_Ret + 1)*np.sqrt(P_ElnRet_err**2 + 0.25*P_VarlnRet_err**2)

        P_Var_sim_Ret = (P_sim_Ret + 1)**2*(np.exp(P_VarlnRet) - 1)
        _err_mean = 2*(P_sim_Ret + 1)*P_sim_Ret_err*(np.exp(P_VarlnRet) - 1)
        _err_var = (P_sim_Ret + 1)**2*np.exp(P_VarlnRet)*P_VarlnRet_err
        P_Var_sim_Ret_err = np.sqrt(_err_mean**2 + _err_var**2)

        P_sim_Ret_low = np.exp(P_ElnRet - np.sqrt(P_VarlnRet)) - 1
        P_sim_Ret_high = np.exp(P_ElnRet + np.sqrt(P_VarlnRet)) - 1

        print(f"Expected mean return: {100*P_sim_Ret:.1f}% ± {100*P_sim_Ret_err:.1f}%")
        print(f"Expected return range: ({100*P_sim_Ret_low:.1f}%, {100*P_sim_Ret_high:.1f}%), given sample standard deviation: {100*np.sqrt(P_Var_sim_Ret):.1f}% ± {100*(P_Var_sim_Ret_err/(2*np.sqrt(P_Var_sim_Ret))):.1f}%")

        # Calculate Sharpe ratio and Kelly ratio
        sharpe_ratio = (P_sim_Ret - sim_Ret_free)/np.sqrt(P_Var_sim_Ret)
        sharpe_ratio_err = np.sqrt(P_sim_Ret_err**2/P_Var_sim_Ret + (sharpe_ratio*P_Var_sim_Ret_err/(2*P_Var_sim_Ret))**2)
        print(f"Sharpe ratio: {sharpe_ratio:.2f} ± {sharpe_ratio_err:.2f}")

        kelly_ratio = (P_sim_Ret - sim_Ret_free)/P_Var_sim_Ret
        kelly_ratio_err = np.sqrt(P_sim_Ret_err**2 + (kelly_ratio*P_Var_sim_Ret_err)**2)/P_Var_sim_Ret
        print(f"Kelly ratio: {kelly_ratio:.2f} ± {kelly_ratio_err:.2f}")

        # Find market correlation and beta
        beta = None
        if market_symbol_id is not None:
            M_lnRet = ElnRet[market_symbol_id]
            M_lnRet_err = ElnRet_err[market_symbol_id]

            M_VarlnRet = CovlnRet[market_symbol_id][market_symbol_id]
            M_VarlnRet_err = CovlnRet_err[market_symbol_id][market_symbol_id]

            PM_CovlnRet = (weights @ CovlnRet[market_symbol_id])
            PM_CovlnRet_err = np.sqrt(np.sum((weights*CovlnRet_err[market_symbol_id])**2))

            M_sim_Ret = np.exp(M_lnRet + 0.5*M_VarlnRet) - 1
            M_sim_Ret_err = (M_sim_Ret + 1)*np.sqrt(M_lnRet_err**2 + 0.25*M_VarlnRet_err**2)

            PM_Cov_sim_Ret = (M_sim_Ret + 1)*(P_sim_Ret + 1)*(np.exp(PM_CovlnRet) - 1)
            _P_err_mean = (M_sim_Ret + 1)*P_sim_Ret_err*(np.exp(PM_CovlnRet) - 1)
            _M_err_mean = M_sim_Ret_err*(P_sim_Ret + 1)*(np.exp(PM_CovlnRet) - 1)
            _err_cov = (M_sim_Ret + 1)*(P_sim_Ret + 1)*np.exp(PM_CovlnRet)*PM_CovlnRet_err
            PM_Cov_sim_Ret_err = np.sqrt(_P_err_mean**2 + _M_err_mean**2 + _err_cov**2)

            M_Var_sim_Ret = (M_sim_Ret + 1)**2*(np.exp(M_VarlnRet) - 1)
            _M_err_mean = 2*(M_sim_Ret + 1)*M_sim_Ret_err*(np.exp(M_VarlnRet) - 1)
            _M_err_var = (M_sim_Ret + 1)**2*np.exp(M_VarlnRet)*M_VarlnRet_err
            M_Var_sim_Ret_err = np.sqrt(_M_err_mean**2 + _M_err_var**2)

            pearson = PM_Cov_sim_Ret/np.sqrt(M_Var_sim_Ret*P_Var_sim_Ret)
            _cov_err = PM_Cov_sim_Ret_err/PM_Cov_sim_Ret
            _P_err = P_Var_sim_Ret_err/(2*P_Var_sim_Ret)
            _M_err = M_Var_sim_Ret_err/(2*M_Var_sim_Ret)
            pearson_err = pearson*np.sqrt(_cov_err**2 + _P_err**2 + _M_err**2)
            print(f"pearson correlation: {pearson:.3f} ± {pearson_err:.3f}")

            beta = PM_Cov_sim_Ret/M_Var_sim_Ret
            beta_err = np.sqrt((PM_Cov_sim_Ret_err/M_Var_sim_Ret)**2 + (beta*M_Var_sim_Ret_err/M_Var_sim_Ret)**2)

            print(f"beta: {beta:.3f} ± {beta_err:.3f}")

        # Find alpha
        if (sim_Ret_free is not None) and (beta is not None):
            alpha = (P_sim_Ret - sim_Ret_free) - beta*(M_sim_Ret - sim_Ret_free)

            _beta_err = beta_err*(M_sim_Ret - sim_Ret_free)
            _market_err = beta*M_sim_Ret_err
            alpha_err = np.sqrt(P_sim_Ret_err**2 + _beta_err**2 + _market_err**2)

            print(f"alpha: {100*alpha:.1f}% ± {100*alpha_err:.1f}")

        # Calculate the effective number of bets
        n_eff = 1/np.sum(weights**2)
        print(f"Effective number of positions: {n_eff:.1f} = 1/sum(weights^2)")

        # Show the weights
        indices = sorted(np.arange(self.width), key=lambda i: weights[i])[::-1]
        block_width = 7
        weight_str = "weights: "
        for i in range(self.width):
            n = indices[i]
            weight_str += f"{self.tickers[n]} = {100*weights[n]:.1f}% "
            if i % block_width == (block_width - 1):
                weight_str += "\n         "
        print(weight_str)

    def show_market_statistics(self):
        """Print useful market statistics

        Use the statistics from market_statistics to calculate useful metrics
        for the individual stocks being analyized. These include the expected
        return, Sharpe ratio, Kelly ratio, alpha and beta.

        If avaliable VTI or SPY will be used as the market return.
        """

        # Get annuallized returns and covarience matrix
        (ElnRet, ElnRet_err), (CovlnRet, CovlnRet_err) = self.market_statistics()

        VarlnRet = np.diag(CovlnRet)
        VarlnRet_err = np.diag(CovlnRet_err)

        market_symbol_id = None
        if "VTI" in self.tickers:
            market_symbol_id = self.tickers.index("VTI")
        if "SPY" in self.tickers:
            market_symbol_id = self.tickers.index("SPY")
        else:
            print("No reference EFT for the whole market")

        # Get the risk free return
        sim_Ret_free = None
        if market_symbol_id is not None:
            sim_Ret_free = self.get_Ret_free()

        # Compute simple returns and covariance matrix
        sim_Ret = np.exp(ElnRet + 0.5*VarlnRet) - 1
        sim_Ret_err = (sim_Ret + 1)*np.sqrt(ElnRet_err**2 + 0.25*VarlnRet_err**2)

        Cov_sim_Ret = np.outer(sim_Ret + 1, sim_Ret + 1)*(np.exp(CovlnRet) - 1)
        _dsim_ret_sim_ret = (np.outer(sim_Ret + 1, sim_Ret_err) + np.outer(sim_Ret_err, sim_Ret + 1))
        _err_mean = _dsim_ret_sim_ret*(np.exp(CovlnRet) - 1)
        _err_cov = np.outer(sim_Ret + 1, sim_Ret + 1)*np.exp(CovlnRet)*CovlnRet_err
        Cov_sim_Ret_err = np.sqrt(_err_mean**2 + _err_cov**2)

        Var_sim_Ret = np.diag(Cov_sim_Ret)
        Var_sim_Ret_err = np.diag(Cov_sim_Ret_err)

        sim_Ret_low = np.exp(ElnRet - np.sqrt(VarlnRet)) - 1
        sim_Ret_high = np.exp(ElnRet + np.sqrt(VarlnRet)) - 1

        # Get market returns if SPY or VTI is in the data set
        sim_Ret_market = None
        sim_Ret_market_err = None
        if market_symbol_id is not None:
            sim_Ret_market = sim_Ret[market_symbol_id]
            sim_Ret_market_err = sim_Ret_err[market_symbol_id]
            print(f"Using {self.tickers[market_symbol_id]} to track the market")
            print(f"the return is {100*sim_Ret_market:.1f}% ± {100*sim_Ret_market_err:.1f} and the risk free return is {100*sim_Ret_free:.1f}%")

        # Show statistics for individual stocks
        print(f"Market statistics, {n} samples at {self.interval} intervals")
        for symbolID in range(self.width):
            _sim_Ret = sim_Ret[symbolID]
            _sim_Ret_err = sim_Ret_err[symbolID]

            _Var_sim_Ret = Var_sim_Ret[symbolID]
            _Var_sim_Ret_err = Var_sim_Ret_err[symbolID]

            _sim_Ret_low = sim_Ret_low[symbolID]
            _sim_Ret_high = sim_Ret_high[symbolID]

            print(f"\nTicker {self.tickers[symbolID]}:")
            print(f"Expected mean return: {100*_sim_Ret:.1f}% ± {100*_sim_Ret_err:.1f}%")
            print(f"Expected return range: ({100*_sim_Ret_low:.1f}%, {100*_sim_Ret_high:.1f}%), given sample standard deviation: {100*np.sqrt(_Var_sim_Ret):.1f}% ± {100*(_Var_sim_Ret_err/(2*np.sqrt(_Var_sim_Ret))):.1f}%")

            # find sharpe and kelly ratios
            if sim_Ret_free is not None:
                sharpe_ratio = (_sim_Ret - sim_Ret_free)/np.sqrt(_Var_sim_Ret)
                sharpe_ratio_err = np.sqrt(_sim_Ret_err**2/_Var_sim_Ret + (sharpe_ratio*_Var_sim_Ret_err/(2*_Var_sim_Ret))**2)
                print(f"Sharpe ratio: {sharpe_ratio:.2f} ± {sharpe_ratio_err:.2f}")

                kelly_ratio = (_sim_Ret - sim_Ret_free)/_Var_sim_Ret
                kelly_ratio_err = np.sqrt(_sim_Ret_err**2 + (kelly_ratio*_Var_sim_Ret_err)**2)/_Var_sim_Ret
                print(f"Kelly ratio: {kelly_ratio:.2f} ± {kelly_ratio_err:.2f}")

            # find correlation and beta
            beta = None
            if market_symbol_id is not None:
                _Cov_i_market = Cov_sim_Ret[symbolID, market_symbol_id]
                _Cov_i_market_err = Cov_sim_Ret_err[symbolID, market_symbol_id]
                _Var_market = Var_sim_Ret[market_symbol_id]
                _Var_market_err = Var_sim_Ret_err[market_symbol_id]

                pearson = _Cov_i_market/np.sqrt(_Var_market*_Var_sim_Ret)
                _cov_err = _Cov_i_market_err/_Cov_i_market
                _i_err = _Var_sim_Ret_err/(2*_Var_sim_Ret)
                _market_err = _Var_market_err/(2*_Var_market)
                pearson_err = pearson*np.sqrt(_cov_err**2 + _i_err**2 + _market_err**2)
                print(f"pearson correlation: {pearson:.3f} ± {pearson_err:.3f}")

                beta = _Cov_i_market/_Var_market
                beta_err = np.sqrt((_Cov_i_market_err/_Var_market)**2 + (beta*_Var_market_err/_Var_market)**2)

                print(f"beta: {beta:.3f} ± {beta_err:.3f}")

            # find alpha
            if (sim_Ret_free is not None) and (beta is not None):
                alpha = (_sim_Ret - sim_Ret_free) - beta*(sim_Ret_market - sim_Ret_free)

                _beta_err = beta_err*(sim_Ret_market - sim_Ret_free)
                _market_err = beta*sim_Ret_market_err
                alpha_err = np.sqrt(_sim_Ret_err**2 + _beta_err**2 + _market_err**2)

                print(f"alpha: {100*alpha:.1f}% ± {100*alpha_err:.1f}")
            print()

        # show Covariance condition number and plot the correlation matrix.
        print(f"Covariance matrix condition number: {np.linalg.cond(CovlnRet)}\n")
        pearson = Cov_sim_Ret/np.sqrt(np.outer(Var_sim_Ret, Var_sim_Ret))
        plt.imshow(pearson)
        plt.colorbar()
        plt.title("Pearson correlation")
        ax = plt.gca()
        ax.set_xticks(np.arange(len(self.tickers)))
        ax.set_xticklabels(self.tickers, rotation=90)
        ax.set_yticks(np.arange(len(self.tickers)))
        ax.set_yticklabels(self.tickers)
        plt.show()

    def gamma_huristic(self):
        """Ask the user about acceptible risk return and give a guide for gamma
        values"""

        sim_Ret = 0.1
        try:
            sigma_sim_Ret = 0.01*float(input(f"Given a annual return of {100*sim_Ret:.1f}% what standard deviation would you accept: "))
        except ValueError:
            self.gamma_huristic()
            return

        gamma = sim_Ret/(sigma_sim_Ret)**2
        print(f"This suggests a gamma near {gamma:.2f}")

    def get_gamma(self):
        print("Leave this field blank for a guide.")
        try:
            gamma = float(input("Set gamma: "))
        except ValueError:
            self.gamma_huristic()
            gamma = self.get_gamma()
        return gamma

    def utility(self, mu, Sigma, weight, gamma):
        """Optimization utility function

        Mean-variance utility function, for optimizing portfolios while
        balancing risks and returns.

        utility(w) = mu @ weight - (gamma / 2) weight @ Sigma @ weight

        Parameters
        ----------
        mu : array(width)
            Expected future log returns
        Sigma : array(width, width)
            Covariance matrix of expected future log returns
        weight : array(width)
            A vector of the portfolio weights
        gamma : float
            The risk avoidance parameter

        Returns
        -------
        float
            Utility value given the weights and parameters
        """
        value = weight @ mu - (gamma/2)*(weight @ Sigma @ weight)
        return value

    def solver(self, U, eq_consts=[], ineq_consts=[],
               bounds=None, w_0=None):
        """
        Parameters
        ----------
        U : func(array(width)) -> Float
            A function that takes an array of weights and is minimized to find
            the optimum portfolio weights. The function returns the utility value
        """

        if w_0 is None:
            w_0 = self.weights

        constraints = []
        for eq_const in eq_consts:
            constraints.append({"type": "eq", "fun": eq_const})

        for ineq_const in ineq_consts:
            constraints.append({"type": "ineq", "fun": ineq_const})

        result = scipy.optimize.minimize(U, w_0, method="SLSQP",
                                         bounds=bounds,
                                         constraints=constraints)

        print(f"{result.message} after {result.nit} iterations with utility u(weight) = {-100*result.fun:.1f}%")
        return result.x
