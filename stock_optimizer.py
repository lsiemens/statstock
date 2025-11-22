from matplotlib import pyplot as plt
import numpy as np
import scipy

import portfolio

class Optimizer:
    def __init__(self, portfolio):
        self.width = portfolio.width
        self.length = portfolio.length
        self.interval = portfolio.interval
        self.tickers = portfolio.tickers

        self.logprice = np.array(portfolio.logprice)
        self.logerror = np.array(portfolio.logerror)
        self.weights = np.array(portfolio.weights)

        self.Elnret = None
        self.Elncov = None

    def setup_simple_mean(self):
        match self.interval:
            case "OneDay":
                periods_year = 52*5
            case "OneWeek":
                periods_year = 52
            case _:
                raise NotImplementedError(f"The interval {self.interval} is not implemented")

        mask = np.isfinite(self.logprice).all(axis=0)
        lnret = np.diff(self.logprice[:, mask], axis=1)

        self.Elnret = np.nanmean(lnret, axis=1)*periods_year
        self.Elncov = np.cov(lnret)*periods_year

    def utility(self, weight, gamma, extra_term=None):
        weight = np.asarray(weight)
        value = weight @ self.Elnret - (weight @ self.Elncov @ weight)*gamma/2
        if extra_term is not None:
            value += extra_term(weight)
        return value

    def solver(self, gamma,
               extra_term=None, eq_consts=[], ineq_consts=[],
               bounds=None, w_0=None):
        if w_0 is None:
            w_0 = np.ones(self.width)/self.width

        def U(weight):
            return -1*self.utility(weight, gamma, extra_term)

        constraints = []
        for eq_const in eq_consts:
            constraints.append({"type": "eq", "fun": eq_const})

        for ineq_const in ineq_consts:
            constraints.append({"type": "ineq", "fun": ineq_const})

        results = scipy.optimize.minimize(U, w_0, method="SLSQP", bounds=bounds,
                                          constraints=constraints)

        return results.x, results
        
if __name__ == "__main__":
    p_0 = portfolio.Portfolio("./all_holdings.csv", 500, "OneDay")
    MPT = Optimizer(p_0)
    MPT.setup_simple_mean()

    def g(w):
        return 1 - np.sum(w)

    bounds = [(0, 1.0) for i in range(MPT.width)]
    result = MPT.solver(4, eq_consts = [g], bounds=bounds)

    if (not result[1].success):
        print("Failed to converge")

    weight = result[0]
    Elnret_p = weight @ MPT.Elnret
    Elnvar_p = weight @ MPT.Elncov @ weight

    for i in range(MPT.width):
        print(f"  {MPT.tickers[i]}: {weight[i]:.2f} * {MPT.Elnret[i]:.2f}")

    print(Elnret_p, Elnvar_p)

    Eret = np.exp(Elnret_p + Elnvar_p/2)
    Eret_bounds = (np.exp(Elnret_p - Elnvar_p), np.exp(Elnret_p + Elnvar_p))
    
    print(Eret, Eret_bounds)
    


