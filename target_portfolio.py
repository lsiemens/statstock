import numpy as np
from matplotlib import pyplot as plt

import light_portfolio

class Group:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

        self.tickers = []
        self.quantities = {}
        self.tags = {}

    def add_ticker(self, ticker, quantity, tags=[]):
        self.tickers.append(ticker)
        self.quantities[ticker] = quantity
        self.tags[ticker] = tags

    def finalize(self):
        if not np.isnan(self.weight):
            norm = np.sum(list(self.quantities.values()))
            for key in self.quantities:
                self.quantities[key] = self.quantities[key]/norm
            if not np.isclose(norm, 1):
                print(f"Warring the group {self.name} is not normalized")
                print(f"The missing factor in the weights is {1 - norm:.2e}")
                print("The weights have been normalized to one.\n")

    def __str__(self):
        string = ""
        if np.isnan(self.weight):
            string = f"{self.name}:\n"
            for ticker in self.tickers:
                string += f"    {ticker}: {int(self.quantities[ticker]):d} shares\n"
        else:
            string = f"{self.name}: group weight {100*self.weight:.1f}%\n"
            for ticker in self.tickers:
                string += f"    {ticker}: {100*self.quantities[ticker]:.1f}%\n"
        return string


class TargetPortfolio:
    def __init__(self, fname, lightportfolio):
        self.fname = fname
        self.width = lightportfolio.width
        self.tickers = lightportfolio.tickers
        self.data = None

        self.price = {}
        self.n_shares = {}
        for i, key in enumerate(self.tickers):
            self.price[key] = lightportfolio.price[i]
            self.n_shares[key] = lightportfolio.n_shares[i]

        self.load_target_portfolio()

    def load_target_portfolio(self):
        lines = None
        with open(self.fname, "r") as fin:
            lines = fin.read().split("\n")
            for i in range(len(lines)):
                lines[i] = lines[i].strip()
                if len(lines[i]) >= 1:
                    if lines[i][0] == "#":
                        lines[i] = ""

        self.data = {}
        group = None
        reading_groups = False
        reading_header = False
        for line in lines:
            if not reading_groups:
                if reading_header:
                    if len(line) > 0:
                        if "Group:" in line:
                            reading_header = False
                            reading_groups = True
                        else:
                            line = line.split(', ')
                            group = line[0].strip()
                            weight = float(line[1])
                            self.data[group] = (Group(group, weight))
                else:
                    if "Group, Weight" in line:
                        reading_header = True

            if reading_groups:
                if "Group:" in line:
                    group = line.split()[1].strip()
                elif "Ticker," in line:
                    continue
                elif len(line) > 0:
                    line = line.split(", ")
                    ticker = line[0]
                    value = float(line[1])
                    flags = line[2:]
                    self.data[group].add_ticker(ticker, value, flags)
        norm = 0
        for key in self.data.keys():
            self.data[key].finalize()
            if not np.isnan(self.data[key].weight):
                norm += self.data[key].weight

        if not np.isclose(norm, 1):
            for key in self.data.keys():
                self.data[key].weight = self.data[key].weight/norm
            print(f"Warring the groups are not normalized")
            print(f"The missing factor in the weights is {1 - norm:.2e}")
            print("The group weights have been normalized to one.\n")

    def flatten(self):
        weighted = {}
        individual = {}
        for key in self.data.keys():
            if np.isnan(self.data[key].weight):
                for ticker in self.data[key].tickers:
                    if ticker in individual:
                        raise ValueError("Ticker weights are not unique, there are repeated tickers")
                    individual[ticker] = self.data[key].quantities[ticker]
            else:
                for ticker in self.data[key].tickers:
                    if ticker in weighted:
                        raise ValueError("Ticker weights are not unique, there are repeated tickers")
                    weighted[ticker] = self.data[key].weight*self.data[key].quantities[ticker]
        for ticker in weighted:
            if ticker in individual:
                raise ValueError("Ticker weights are not unique, there are repeated tickers")
        if not np.isclose(np.sum(list(weighted.values())), 1.0):
            raise ValueError("Ticker weights are not normalized!")
        return weighted, individual

    def check_portfolio(self, cash=0, include_individual=False):
        if not include_individual:
            print("Note percentages are calculated while excluding the unweighted stocks.")

        weighted, individual = self.flatten()
        # Validate target weights
        if not np.isclose(np.sum(list(weighted.values())), 1.0):
            raise ValueError("Target portfolio weights are not normalized")

        tickers = [ticker for ticker in self.tickers if ticker not in individual]
        tickers_individual = [ticker for ticker in self.tickers if ticker in individual]

        all_tickers = [ticker for ticker in set(self.tickers).union(weighted.keys()) if ticker not in individual]
        all_tickers_individual = [ticker for ticker in set(self.tickers).union(individual.keys()) if ticker in individual]

        holding_values = {}
        for ticker in tickers:
            holding_values[ticker] = self.n_shares[ticker]*self.price[ticker]
        total_value = np.sum(list(holding_values.values())) + cash

        individual_values = {}
        for ticker in tickers_individual:
            individual_values[ticker] = self.n_shares[ticker]*self.price[ticker]
        total_value_individual = np.sum(list(individual_values.values()))
        if include_individual:
            total_value += total_value_individual
            for ticker in weighted.keys():
                weighted[ticker] *= (total_value - total_value_individual)/total_value

        weights = {}
        for ticker in tickers:
            weights[ticker] = holding_values[ticker]/total_value
        weights["cash"] = cash/total_value

        # validate portfolio weights
        total_weight = np.sum(list(weights.values()))
        if include_individual:
            total_weight += np.sum([value/total_value for value in individual_values.values()])
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Portfolio weights are not normalized")

        print("Current Weighted portfolio holdings:")
        for ticker in tickers:
            print(f"    {ticker}: {100*weights[ticker]:4.2f}%")
            print(f"        price ${self.price[ticker]:.2f} CAD")
            print(f"        number of shares {self.n_shares[ticker]}")
            print(f"        market value ${holding_values[ticker]:.2f} CAD")
        print(f"    cash: {100*weights['cash']:4.2f}%")
        print(f"        value ${cash:.2f} CAD")

        if len(list(tickers_individual)) > 0:
            print("Current unweighted portfolio holdings:")
        for ticker in tickers_individual:
            print(f"    {ticker}:")
            print(f"        price ${self.price[ticker]:.2f} CAD")
            print(f"        number of shares {self.n_shares[ticker]}")
            print(f"        market value ${self.n_shares[ticker]*self.price[ticker]:.2f} CAD")

        total_delta_value = 0
        print(f"\nPortfolio modifications: starting cash ${cash:.2f} CAD")
        if include_individual:
            print(f"\tTotal value (weighted) + (individual) + (cash) in CAD: ${total_value - total_value_individual - cash:.2f} + ${total_value_individual:.2f} + ${cash:.2f} = ${total_value:.2f}")
        else:
            print(f"\tTotal value (weighted) + (individual) + (cash) in CAD: ${total_value - cash:.2f} + ${total_value_individual:.2f} + ${cash:.2f} = ${total_value + total_value_individual:.2f}")

        for ticker in all_tickers:
            print(f"{ticker}:")
            if ticker in self.tickers:
                if ticker in weighted:
                    print(f"    target weight = {100*weighted[ticker]:4.2f}%")
                    delta_value = weighted[ticker]*total_value - holding_values[ticker]
                else:
                    print("    target weight = 0.0%")
                    delta_value = -holding_values[ticker]
                print(f"    current weight = {100*weights[ticker]:4.2f}%")
                print(f"    Delta value = ${delta_value:.2f} CAD")
                print(f"    Delta shares = {delta_value/self.price[ticker]:.1f}")
            else:
                print(f"    current weight = {0:4.2f}%")
                delta_value = weighted[ticker]*total_value
                print(f"    Delta value = ${delta_value:.2f} CAD")
                print("    Delta shares = (no stock price data)")
            total_delta_value += delta_value

        if not np.isclose(total_delta_value, cash):
            raise ValueError(f"The total change in portfolio value should be equal to the available cash, instead it was ${total_delta_value:.2f} CAD")

        print()
        for ticker in all_tickers_individual:
            print(f"{ticker}:")
            print(f"    target number of shares = {int(individual[ticker]):d}")
            if ticker in self.tickers:
                print(f"    current number of shares = {int(self.n_shares[ticker]):d}")
                print(f"    Delta shares = {int(individual[ticker]) - int(self.n_shares[ticker]):d}")
            else:
                print("    current number of shares = 0")
                print(f"    Delta shares = {int(individual[ticker]):d}")

    def __str__(self):
        string = ""
        for key in self.data.keys():
            string += str(self.data[key]) + "\n"
        return string

if __name__ == "__main__":
    p_0 = light_portfolio.LightPortfolio("./all_holdings.csv")
    target_portfolio = TargetPortfolio("../Notes/target_portfolio.csv", p_0)
    print("Target portfolio")
    print(target_portfolio)

    cash = float(input("Available cash [CAD]:"))
    target_portfolio.check_portfolio(cash, include_individual=True)

