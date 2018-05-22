""" 
Interesting result but the model is wrong
"""

from scipy.stats import norm

a, b = -1.0, 1.0 # a standard devaition is 1.0
n = 30

distribution = norm(loc=0.0, scale=1.0) # zero offset with std of 1.0

alpha, beta = distribution.cdf(a), 1 - distribution.cdf(b)

class tree:
    def __init__(self, n, p=1.0, state="-a"):
        self.n = n
        self.p = p
        self.state = state
        self.children = None

        self.generate_tree()

    def generate_tree(self):
        if (self.n < 1) or (self.state == "b"):
            return None

        if self.state == "-a":
            self.children = [tree(self.n - 1, self.p*alpha, state="a"), tree(self.n - 1, self.p*(1.0 - alpha), state="-a")]
        elif (self.state == "a") or (self.state == "-b"):
            self.children = [tree(self.n - 1, self.p*beta, state="b"), tree(self.n - 1, self.p*(1.0 - beta), state="-b")]

    def calculate(self):
        non, bought, sold = 0.0, 0.0, 0.0
        if self.children is not None:
            for child in self.children:
                cnon, cbought, csold = child.calculate()
                non, bought, sold = non + cnon, bought + cbought, sold + csold
        else:
            if self.state == "b":
                sold = self.p
            elif self.state == "-a":
                non = self.p
            elif (self.state == "a") or (self.state == "-b"):
                bought = self.p

        return non, bought, sold

A = tree(n)
non, bought, sold = A.calculate()
print("no action: " + str(non))
print("purchase: " + str(bought))
print("sale: " + str(sold))
print("sanity check: " + str(non + bought + sold) + " should equal 1.0")
