# Project Report on Mini-Project2: RandomVariateGenerator (RandVarGen)

by: Yogesh Raheja (yraheja3@gatech.edu)

**Objective:** Make a 'nice' library of Random Variate Generation routines in Python from a given U(0,1) number. Random Variate Generator (RandVarGen) exposes a number of methods for generating random numbers drawn from a variety of probability distributions. 

RandVarGen has the following properties:
1. Output appears to be i.i.d. Unif(0,1). The probability distributions are derived using NumPy's Legacy Random Generation - [RandomState](https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState "RandomState").
2. It is fast. The code uses highly-optimized math formula's to minimize resource intensive functions like log(), etc. However, the speed is slower than NumPy's RandomState and [BitGenerator](https://numpy.org/doc/stable/reference/random/generator.html "BitGenerator") by about 10-25%. 
3. It has the ability to reproduce any sequence it generates. RandVarGen uses a seed and RandomStates [Mersenne Twister pseudo-random number generator (PRNG)](https://en.wikipedia.org/wiki/Mersenne_Twister "Mersenne Twister") to generate a 'sequence' of required probability distribution.

In addition to the distribution-specific arguments, each method takes a keyword argument size that defaults to None. If size is None, then a single value is generated and returned. If size is an integer, then a 1-D array filled with generated values is returned.

This document provides the user's guide, complete source code, and some appropriate examples to show the outcome of the generator.

### Bernoulli Distribution

### Binomial Distribution
Draw samples from a binomial distribution.

$e^{i \pi} = -1$

Samples are drawn from a binomial distribution with specified parameters, n trials and p probability of success where n is an integer >= 0 and p is in the interval [0,1]. (n may be input as a float, but it is truncated to an integer in use)

