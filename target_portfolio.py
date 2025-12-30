import numpy as np
from matplotlib import pyplot as plt

import portfolio

class Group:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

        self.tickers = []
        self.quantities = []
        self.tags = []

    def add_ticker(self, ticker, quantity, tags=[]):
        self.tickers.append(ticker)
        self.quantities.append(quantity)
        self.tags.append(tags)

    def finalize(self):
        if not np.isnan(self.weight):
            norm = np.sum(self.quantities)
            self.quantities = [value/norm for value in self.quantities]
            if not np.isclose(norm, 1):
                print(f"Warring the group {self.name} is not normalized")
                print(f"The missing factor in the weights is {1 - norm:.2e}")
                print("The weights have been normalized to one.\n")

    def __str__(self):
        string = ""
        if np.isnan(self.weight):
            string = f"{self.name}:\n"
            for i in range(len(self.tickers)):
                string += f"    {self.tickers[i]}: {int(self.quantities[i]):d} shares\n"
        else:
            string = f"{self.name}: group weight {100*self.weight:.1f} %\n"
            for i in range(len(self.tickers)):
                string += f"    {self.tickers[i]}: {100*self.quantities[i]:.1f} %\n"
        return string


class TargetPortfolio:
    def __init__(self, fname, portfolio):
        self.fname = fname
        self.width = portfolio.width
        self.tickers = portfolio.tickers
        self.data = None

        self.logprice = portfolio.logprice
        self.logerror = portfolio.logerror

        self.n_shares = portfolio.n_shares

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
        weighted = ([], [])
        individual = ([], [])
        for key in self.data.keys():
            if np.isnan(self.data[key].weight):
                inames, iweights = individual
                inames += self.data[key].tickers
                iweights += self.data[key].quantities
                individual = (inames, iweights)
            else:
                wnames, wweights = weighted
                wnames += self.data[key].tickers
                wweights += [self.data[key].weight*value for value in self.data[key].quantities]
                weighted = (wnames, wweights)
        all_tickers = weighted[0] + individual[0]
        if len(all_tickers) != len(set(all_tickers)):
            raise ValueError("Ticker weights are not unique, there are repeated tickers")
        return weighted, individual

    def check_portfolio(self, cash=0):
        weighted, individual = self.flatten()
        
        windices = np.array([index for index, ticker in enumerate(self.tickers) if ticker not in individual[0]], dtype=int)
        
        tickers = np.array(self.tickers)[windices]
        holding_values = self.n_shares[windices]*np.exp(self.logprice[windices, -1])
        total_value = np.sum(holding_values) + cash
        weights = holding_values/total_value

        print(f"Portfolio modifications: starting cash ${cash:.2f} CAD")
        for index, ticker in enumerate(weighted[0]):
            print(f"{ticker}:")
            print(f"    target weight = {100*weighted[1][index]:4.1f} %")
            if ticker in self.tickers:
                windex = np.argmax(ticker == tickers)
                print(f"    current weight = {100*weights[windex]:4.1f} %")
                delta_value = weighted[1][index]*total_value - holding_values[windex]
                full_index = np.argmax(ticker == np.array(self.tickers))
                print(f"    Delta value = ${delta_value:.2f} CAD")
                print(f"    Delta shares = {delta_value/np.exp(self.logprice[full_index, -1]):.1f}")
            else:
                print(f"    current weight = {0:4.1f} %")
                delta_value = weighted[1][index]*total_value
                print(f"    Delta value = ${delta_value:.2f} CAD")
                print(f"    Delta shares = (no stock price data)")

        print()
        for index, ticker in enumerate(individual[0]):
            print(f"{ticker}:")
            print(f"    target # shares = {int(individual[1][index]):d}")
            if ticker in self.tickers:
                iindex = np.argmax(ticker == np.array(self.tickers))
                print(f"    current # shares = {int(self.n_shares[iindex]):d}")
                print(f"    Delta # shares = {int(individual[1][index]) - int(self.n_shares[iindex]):d}")
            else:
                print(f"    current # shares = 0")
                print(f"    Delta # shares = {int(individual[1][index]):d}")

    def __str__(self):
        string = ""
        for key in self.data.keys():
            string += str(self.data[key]) + "\n"
        return string

if __name__ == "__main__":
    p_0 = portfolio.Portfolio("./all_holdings.csv", 365, "OneDay")
    target_portfolio = TargetPortfolio("../Notes/target_portfolio.csv", p_0)
    print("Target portfolio")
    print(target_portfolio)

    target_portfolio.check_portfolio(25000)
    #A, B = target_portfolio.flatten()
    #_, weights = A
    #print(weights, np.sum(weights))

