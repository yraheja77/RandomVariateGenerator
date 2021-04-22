# Make me a nice library of random variate generation routines.
# You can use your favorite high-level language like C++, Java, Python,
# Matlab, or even Excel. Include in your library routines for generating
# random variates from all of the usual discrete and continuous distributions,
# e.g., Bern(p), Geom(p), Exp(λ), Normal(μ,σ2), Gamma(α,β), Weibull(α,β), etc., etc.
# (Just one routine per distribution is fine.)
# Include in your write-up an easy user’s guide, complete source code,
# and some appropriate examples.

import numpy as np
import math
import operator
import random
from numpy.random import rand
from scipy.stats import norm
from decimal import Decimal
# np.random.seed(270270)


# Initialising and Defining the class rvg
class RandVarGen:
    def __init__(self, random_no=0):
        self.seed = random_no

# Validating the random number and raising an exception if error
    seed = property(operator.attrgetter('_seed'))

    @seed.setter
    def seed(self, seed):
        # if not ((seed >= 0) and (seed <= 1)):
        if isinstance(seed, (str, list, dict, tuple)):
            raise Exception("Please enter a valid integer as seed")
        if isinstance(seed, float):
            seed = int(seed)
            print("Converted seed to integer")
        if seed == 0:
            print("Considering default seed of 0")
        self._seed = int(seed)

# To generate a single X of a Bernoulli(p) distribution
    def bern(self, p, u):
        if u <= p:
            return 1
        else:
            return 0

# To generate a np.array of X of a Bernoulli(p) distribution
    def bernoulli(self, p, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.bern(p, unif_no))
        else:
            unif_no = random.random()
            return self.bern(p, unif_no)
        return np.array(out_array)

# To generate X of a Geometric(p) distribution
    def geom(self, p, unif_no):
        return math.ceil(np.log(1-unif_no) / np.log(1-p))

# To generate a np.array of X of a Bernoulli(p) distribution
    def geometric(self, p, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.geom(p, unif_no))
        else:
            unif_no = random.random()
            return self.geom(p, unif_no)
        return np.array(out_array)

# To generate X of a Poisson(lambda) distribution
    def pois(self, a_lambda, unif_no):
        lower_bound = 0
        x = 0
        while x > -1:
            fx = Decimal((Decimal(math.e ** -a_lambda) * (2 ** x)) / Decimal(math.factorial(x)))
            Fx = lower_bound + fx
            if lower_bound <= unif_no <= Fx:
                return x
            else:
                x += 1
                lower_bound = Fx

    # To generate a np.array of X of a Bernoulli(p) distribution
    def poisson(self, a_lambda, size=None):
        random.seed(self.seed)
        if a_lambda > 2:
            return ("Can't process lamda values greater than 2 right now. Please wait for the next version of the library")
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.pois(a_lambda, unif_no))
        else:
            unif_no = random.random()
            return self.pois(a_lambda, unif_no)
        return np.array(out_array)

# To generate X of a Triangular(0,1,2) distribution
    def tria(self, unif_no):
        if unif_no < 0.5:
            return (2 * unif_no) ** 0.5
        else:
            return 2 - (2 * (1-unif_no)) ** 0.5

    # To generate a np.array of X of a Bernoulli(p) distribution
    def triangular(self, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.tria(unif_no))
        else:
            unif_no = random.random()
            return self.tria(unif_no)
        return np.array(out_array)

# To generate X of a Normal(mu, var) distribution
    def norm(self, mu, var, unif_no):
        return norm.ppf(unif_no, loc=mu, scale=var**0.5)

    # To generate a np.array of X of a Bernoulli(p) distribution
    def normal(self, mu=0, var=1, size=None):
        random.seed(self.seed)
        # if a_lambda > 2:
        #     return ("Can't process lamda values greater than 2 right now. Please wait for the next version of the library")
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.norm(mu, var, unif_no))
        else:
            unif_no = random.random()
            return self.norm(mu, var, unif_no)
        return np.array(out_array)


# To generate X of a Weibull(lambda, beta) distribution
    def weib(self, a_lambda, beta):
        return ((1/a_lambda) * (-np.log(1-self.u))**(1/beta))

# To generate X of a Exp(lambda) distribution
    def exp(self, a_lambda):
        return ((-1/a_lambda) * np.log(1-self.u))

# To generate X of a Discrete pmf given any x and f(x) values distribution
    def discrete(self, x, fx):

        # To check if the input list lengths match
        if len(x) != len(fx):
            return("Input Error: list of x and f(x) have different lenghts")

        # To check if the pmf sums to 1
        if (1-sum(fx)) > 0.000000001:
            return "Input Error: pmf does not add to 1"

        lower_bound = 0
        for i in range(len(x)):
            upper_bound = fx[i] + lower_bound
            if lower_bound <= self.u <= upper_bound:
                return x[i]
            else:
                lower_bound = upper_bound

# To generate X of a Uniform(a,b) distribution
    def unif(self, a, b):
        return a + (b-a) * self.u

# To generate X of a Uniform(a,b) distribution
    def disc_unif(self, n):
        return math.ceil(self.u * n)

    def bin(self, n, p):
        Y = 0
        random.seed(int(self.u * 10**14))
        random

        for i in range(n):
            Y += RandVarGen()

# print(RandVarGen(0.51).bern(0.5))
# print(RandVarGen(0.72).geom(0.3))
# print(RandVarGen(0.313).pois(2))
# print(RandVarGen(0.4).tria())
# print(RandVarGen(0.59).norm(3, 16))
# print(RandVarGen(0.59).norm())
# print(RandVarGen(0.59).exp(2))
# print(RandVarGen(0.59).weib(2, 5))
# print(RandVarGen(0.63).discrete([-1, 2.5, 4], [0.6, 0.3, 0.1]))
# print(RandVarGen(0.63).unif(1, 10))
# print(RandVarGen(0.376).disc_unif(10))

# a=0.376
# print(math.ceil(a))

# print(RandVarGen().bernoulli(0.5))
# print(RandVarGen(10000).geometric(0.5, 10))
# print(RandVarGen(12345678).poisson(2))
# print(RandVarGen(12345678).triangular(100))
print(RandVarGen().normal(size=2))
