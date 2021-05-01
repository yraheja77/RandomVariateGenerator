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
from scipy.stats import norm
from decimal import Decimal
import matplotlib.pyplot as plt


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

# To generate a np.array of X of a Geometric(p) distribution
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

# To generate a np.array of X of a Poisson(lambda) distribution
    def poisson(self, a_lambda, size=None):
        random.seed(self.seed)
        if a_lambda > 2:
            return "Can't process lambda values greater than 2. Please wait for the next version of the library"
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
    def tria(self, a, b, c, unif_no):
        Fc = (c-a)/(b-a)
        if unif_no < Fc:
            return a + ((b-a) * (c-a) * unif_no) ** 0.5
        else:
            return b - ((b-c) * (b-a) * (1-unif_no)) ** 0.5

# To generate a np.array of X of a Triangular(0, 1, 2) distribution
    def triangular(self, a=0, c=1, b=2, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.tria(a, b, c, unif_no))
        else:
            unif_no = random.random()
            return self.tria(a, b, c, unif_no)
        return np.array(out_array)

# To generate X of a Normal(mu, var) distribution
    def norm(self, mu, var, unif_no):
        return norm.ppf(unif_no, loc=mu, scale=var**0.5)

# To generate a np.array of X of a Normal(mu, var) distribution
    def normal(self, mu=0, var=1, size=None):
        random.seed(self.seed)
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
    def weib(self, a_lambda, beta, unif_no):
        return (1/a_lambda) * (-np.log(1-unif_no))**(1/beta)

# To generate a np.array of X of a Weibull(lambda, beta) distribution
    def weibull(self, a_lambda, beta, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.weib(a_lambda=a_lambda, beta=beta, unif_no=unif_no))
        else:
            unif_no = random.random()
            return self.weib(a_lambda=a_lambda, beta=beta, unif_no=unif_no)
        return np.array(out_array)

# To generate X of an Exponential(lambda) distribution
    def exp(self, a_lambda, unif_no):
        return (-1/a_lambda) * np.log(1-unif_no)

# To generate a np.array of X of an Exponential(lambda) distribution
    def exponential(self, a_lambda, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.exp(a_lambda=a_lambda, unif_no=unif_no))
        else:
            unif_no = random.random()
            return self.exp(a_lambda=a_lambda, unif_no=unif_no)
        return np.array(out_array)

# To generate X of a Discrete pmf given any x and f(x) values distribution
    def disc(self, x, fx, unif_no):
        lower_bound = 0
        for i in range(len(x)):
            upper_bound = fx[i] + lower_bound
            if lower_bound <= unif_no <= upper_bound:
                return x[i]
            else:
                lower_bound = upper_bound

# To generate a np.array of X of a Discrete pmf given any x and f(x) distribution
    def discrete(self, x, fx, size=None):
        random.seed(self.seed)

        # To check if the input list lengths match
        if len(x) != len(fx):
            return "Input Error: list of x and f(x) have different lenghts"

        # To check if the pmf sums to 1
        if (1-sum(fx)) > 0.000000001:
            return "Input Error: pmf does not add to 1"

        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.disc(x, fx, unif_no=unif_no))
        else:
            unif_no = random.random()
            return self.disc(x, fx, unif_no=unif_no)
        return np.array(out_array)

# To generate X of a Uniform(a,b) distribution
    def unif(self, a, b, unif_no):
        return a + (b-a) * unif_no

# To generate a np.array of X of a Uniform(a,b) distribution
    def uniform(self, a=0, b=1, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.unif(a=a, b=b, unif_no=unif_no))
        else:
            unif_no = random.random()
            return self.unif(a=a, b=b, unif_no=unif_no)
        return np.array(out_array)

# To generate X of a Discrete Uniform(n) distribution with probability 1/n
    def disc_unif(self, n, unif_no):
        return math.ceil(unif_no * n)

# To generate a np.array of X of a Discrete Uniform(n) distribution with probability 1/n
    def discrete_uniform(self, n, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                unif_no = random.random()
                out_array.append(self.disc_unif(n=n, unif_no=unif_no))
        else:
            unif_no = random.random()
            return self.disc_unif(n=n, unif_no=unif_no)
        return np.array(out_array)

# To generate a single X of a Binomial(n,p) distribution
    def bin(self, n, p):
        n = math.floor(n)
        return np.sum(np.array(self.bernoulli(p=p, size=n)))

# To generate a np.array of X of a Binomial(n,p) distribution
    def binomial(self, n, p, size=None):
        random.seed(self.seed)
        if size:
            # random.seed(random.random())
            size = int(size)
            out_array = []
            for i in range(size):
                self.seed = int(random.random() * 10 ** 20)
                out_array.append(self.bin(n=n, p=p))
        else:
            self.seed = int(random.random() * 10 ** 20)
            return self.bin(n=n, p=p)
        return np.array(out_array)

# To generate a single of X of a Erlang(lambda, n) distribution
    def erl(self, a_lambda, n=1):
        random.seed(self.seed)
        unif_no_product = 1
        for i in range(n):
            unif_no_product *= random.random()
        return (-1/a_lambda) * np.log(unif_no_product)

# To generate a np.array of X of a Erlang(lambda, n) distribution
    def erlang(self, a_lambda, n, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                self.seed = int(random.random() * 10 ** 20)
                out_array.append(self.erl(a_lambda=a_lambda, n=n))
        else:
            self.seed = int(random.random() * 10 ** 20)
            return self.erl(a_lambda=a_lambda, n=n)
        return np.array(out_array)

# To generate a single of X of a NegBin(n, p) distribution
    def NegBin_single(self, n, p):
        return np.sum(np.array(self.geometric(p=p, size=n)))

# To generate a np.array of X of a NegBin(n, p) distribution
    def negative_binomial(self, n, p, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                self.seed = int(random.random() * 10 ** 20)
                out_array.append(self.NegBin_single(p=p, n=n))
        else:
            self.seed = int(random.random() * 10 ** 20)
            return self.NegBin_single(p=p, n=n)
        return np.array(out_array)

# To generate a single of X of a Normal(n, p) distribution
    def chisquare_single(self, n):
        out_value = 0
        for i in range(n):
            out_value += self.normal(mu=0, var=1)**2
        return out_value

# To generate a np.array of X of a Normal(n, p) distribution
    def chisquare(self, ddof, size=None):
        random.seed(self.seed)
        if size:
            size = int(size)
            out_array = []
            for i in range(size):
                self.seed = int(random.random() * 10 ** 20)
                out_array.append(self.chisquare_single(n=ddof))
        else:
            self.seed = int(random.random() * 10 ** 20)
            return self.chisquare_single(n=ddof)
        return np.array(out_array)


# To plot a derived distribution
def plot_cont_dist(x, title):
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("probability")
    plt.hist(x, bins=100, color="blue")
    plt.show()

# print(RandVarGen().bernoulli(0.5))
# print(RandVarGen(10000).geometric(0.5, 10))
# print(RandVarGen(12345678).poisson(2))
# print(RandVarGen(12345678).triangular(100))
# print(RandVarGen(1234).normal(mu=50, size=100))
# print(RandVarGen(123).weibull(4, 10, 100))
# print(RandVarGen().exponential(1, 100))
# print(RandVarGen().discrete([-1, 2.5, 4], [0.6, 0.3, 0.1], 1000))
# print(RandVarGen(123).uniform(4, 5, 100))
# print(RandVarGen(123).discrete_uniform(6, 100))
# print(RandVarGen(1234).binomial(3, 0.4, 100))
# print(RandVarGen(1234).erlang(5, 5, 100))
# print(RandVarGen().negative_binomial(10, 0.5, 100))
# print(RandVarGen().chisquare(5, 100))


size1 = 10000
np.random.seed(0)

# # Plotting Bernoulli(p) distribution in RandVarGen with Python Native
# bernoulli_RandVarGen = RandVarGen().bernoulli(p=0.7, size=size1)
# print(bernoulli_RandVarGen/bernoulli_RandVarGen.size)
# plot_cont_dist(bernoulli_RandVarGen, 'Bernoulli Distribution (RandVarGen)')
#
# # Comparing Binomial distribution in RandVarGen with Python Native
# binomial_RandVarGen = RandVarGen().binomial(5, 0.5,  size=size1)
# binomial_native = np.random.binomial(5, 0.5, size=size1)
# plot_cont_dist(binomial_RandVarGen, 'Binomial Distribution (RandVarGen)')
# plot_cont_dist(binomial_native, 'Binomial Distribution (Numpy native)')

# # Comparing Chi-Square distribution in RandVarGen with Python Native
# chisquare_RandVarGen = RandVarGen().chisquare(1, size=size1)
# chisquare_native = np.random.chisquare(1, size=size1)
# plot_cont_dist(chisquare_RandVarGen, 'Chi-Square Distribution (RandVarGen)')
# plot_cont_dist(chisquare_native, 'Chi-Square Distribution (Numpy native)')

# # Comparing Discrete distribution in RandVarGen with Python Native
# discrete_RandVarGen = RandVarGen().discrete([-1, 2.5, 4], [0.6, 0.3, 0.1],  size=size1)
# plot_cont_dist(discrete_RandVarGen, 'Discrete Distribution (RandVarGen)')
#
# # Comparing Discrete Uniform distribution in RandVarGen with Python Native
# discunif_RandVarGen = RandVarGen().discrete_uniform(12, size=size1)
# plot_cont_dist(discunif_RandVarGen, 'Discrete Uniform Distribution (RandVarGen)')
#
# # Comparing Erlang distribution in RandVarGen with Python Native
# erlang_RandVarGen = RandVarGen().erlang(1, 2,  size=size1)
# plot_cont_dist(erlang_RandVarGen, 'Erlang Distribution (RandVarGen)')
#
# # Comparing Exponential distribution in RandVarGen with Python Native
# exponential_RandVarGen = RandVarGen().exponential(10,  size=size1)
# exponential_native = np.random.exponential(10, size=size1)
# plot_cont_dist(exponential_RandVarGen, 'Exponential Distribution (RandVarGen)')
# plot_cont_dist(exponential_native, 'Exponential Distribution (Numpy native)')
#
# # Comparing Geometric distribution in RandVarGen with Python Native
# geometric_RandVarGen = RandVarGen().geometric(p=0.6, size=size1)
# geometric_native = np.random.geometric(p=0.6, size=size1)
# plot_cont_dist(geometric_RandVarGen, 'Geometric Distribution (RandVarGen)')
# plot_cont_dist(geometric_native, 'Geometric Distribution (Numpy native)')
#
# # Comparing Negative Binomial distribution in RandVarGen with Python Native
# negbinomial_RandVarGen = RandVarGen().negative_binomial(5, 0.5,  size=size1)
# negbinomial_native = np.random.negative_binomial(5, 0.5, size=size1)
# plot_cont_dist(negbinomial_RandVarGen, 'Negative Binomial Distribution (RandVarGen)')
# plot_cont_dist(negbinomial_native, 'Negative Binomial Distribution (Numpy native)')
#
# # Comparing Normal distribution in RandVarGen with Python Native
# normal_RandVarGen = RandVarGen().normal(size=size1)
# normal_native = np.random.normal(size=size1)
# plot_cont_dist(normal_RandVarGen, 'Normal Distribution (RandVarGen)')
# plot_cont_dist(normal_native, 'Normal Distribution (Numpy native)')
#
# # Comparing Poisson distribution in RandVarGen(lambda can be max 2 right now) with Python Native
# poisson_RandVarGen = RandVarGen().poisson(2, size=size1)
# poisson_native = np.random.poisson(2, size=size1)
# plot_cont_dist(poisson_RandVarGen, 'Poisson Distribution (RandVarGen)')
# plot_cont_dist(poisson_native, 'Poisson Distribution (Numpy native)')
#
# # Comparing Triangular distribution in RandVarGen with Python Native
# triangular_RandVarGen = RandVarGen().triangular(size=size1)
# triangular_native = np.random.triangular(0, 1, 2, size=size1)
# plot_cont_dist(triangular_RandVarGen, 'Triangular Distribution (RandVarGen)')
# plot_cont_dist(triangular_native, 'Triangular Distribution (Numpy native)')
#
# # Comparing Uniform distribution in RandVarGen with Python Native
# uniform_RandVarGen = RandVarGen().uniform(9, 27,  size=size1)
# uniform_native = np.random.uniform(9, 27, size=size1)
# plot_cont_dist(uniform_RandVarGen, 'Uniform Distribution (RandVarGen)')
# plot_cont_dist(uniform_native, 'Uniform Distribution (Numpy native)')
#
# # Comparing Weibull distribution in RandVarGen with Python Native
# weibull_RandVarGen = RandVarGen().weibull(1, 3, size=size1)
# weibull_native = np.random.weibull(3, size=size1)
# plot_cont_dist(weibull_RandVarGen, 'Weibull Distribution (RandVarGen)')
# plot_cont_dist(weibull_native, 'Weibull Distribution (Numpy native)')
