#! /usr/bin/python

# A "hello world" level example of using the SciPy optimize.minimize()
# optimizer.

import scipy.optimize as optimize
import scipy.stats    as stats
import numpy          as np
import sys

N     = 100
mu    = 10.0
sigma = 5.0

# Sample a Gaussian-distributed test data set 
# Because it's Gaussian, we know the ML mu, sigma are just the mean, stdev.
#
data = np.random.normal(mu, sigma, N)

# You pass an "objective function" to SciPy's `optimize.minimize` that
# calculates your f(p): the value of your function given your current
# parameters. SciPy `optimize.minimize` will seek a value of `p` that
# minimizes this function.
#
# Because SciPy `optimize.minimize` has to work on a wide variety of functions,
# the calling interface has to be very general.
#
# The first argument is an ndarray containing the current values of
# your parameters (that you're trying to optimize). The first thing
# your f(p) function might do is unpack this ndarray and name the
# different parameters.
#
# The remaining arguments are anything else (fixed, not changing
# during the optimization) that you need to calculate the current f(p)
# objective function. You passed these to `optimize.minimize` as a
# tuple the `args` optional argument, and SciPy passes them
# (separately) to your objective function every time it needs to call
# it. Here, this is just a single extra argument: <data>.
#
# In log likelihood calculations, we want to maximize log likelihood,
# but optimizers are typically written as minimizers; so objective 
# functions typically return a negative log likelihood (NLL).
#
def nll(p, D):
    curr_mu    = p[0]
    curr_sigma = p[1]
    ll = 0.
    for x in D:
        residual = x - curr_mu       
        ll      += stats.norm.logpdf(residual, 0, curr_sigma)    
        # Alternatively, using stats.norm.logpdf(x, curr_mu, curr_sigma) is shorter & equivalent,
        # but doing it in terms of the residual is a more direct analog to this week's homework.
    return -ll



# Besides the objective function, with its lovingly crafted argument
# interface, you also need a `guess`: the initial p0 vector that you'll
# start the optimization from.
#
# If you're in a convex problem (one global optimum) any guess will
# do, within reason. (Like, here, "within reason" includes not giving
# it a 0 stdev.) For problems with multiple local optima, different
# p0's may give you different answers.
#
# SciPy takes the <p0> vector as an ndarray in the same order that it
# passes new guesses of <p> to your objective function.  SciPy doesn't
# itself ascribe any meaning to the parameters: you tell it an initial
# guess, and it's your objective function that calculates f(p). SciPy
# just changes p, trying to make f(p) better.
#
p0 = np.array([ 0.0, 1.0 ])  # start at a standard normal; anything will do.


# Now we can call the minimizer.
# We give it four arguments:
#   - the objective function f(p, (args))
#   - the initial guess p0
#   - any optional (args)
#   - <bounds>... see next.
#
# On the `bounds`. The minimizer expects that all parameters are
# real-valued and can take values -inf to inf. But here sigma has to
# be nonnegative. We can pass an array of (min, max) tuples to specify
# bounds on each parameter, with `None` meaning no bound at that end.
# The mu parameter is unbounded, so it gets `(None,None)`. The sigma
# parameter is bounded >0, so it gets `(0.0,None)`.
#
# The minimizer gets more fast and powerful if you can provide an
# f'(p) function to calculate first derivative information (the
# "Jacobian matrix") and even better if you can provide an f''(p)
# function to calculate second derivative information (the "Hessian
# matrix"). But you don't need either; it will calculate these
# numerically (albeit laboriously).
#
# See http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#
result    = optimize.minimize(nll, p0, (data), bounds = [(None,None), (0.0,None)])
 

# The <result> object is of type OptimizeResult.
# The main things we care about in it:
#   result.success : True | False for whether the minimization succeeded
#   result.x       : The final answer for your optimized <p> vector
#
if result.success != True:
    sys.exit("Maximum likelihood fit failed")
mu_fit    = result.x[0]
sigma_fit = result.x[1]
 




print("mean = {0:.2f}".format(np.mean(data)))
print("sd   = {0:.2f}".format(np.std(data)))

print("ML fit, mu    = {0:.2f}".format(mu_fit))
print("ML fit, sigma = {0:.2f}".format(sigma_fit))



#################################################################
# Bonus section!
#
# Suppose we wanted to optimize our nonnegative sigma using a general
# purpose optimizer that doesn't allow us to specify bounds. A standard
# trick is to use the tranformation
#    sigma = exp(lambda);    lambda = log(sigma)
# and optimize lambda on the full real-valued range -inf..inf, which
# transforms to a nonnegative sigma.
#
# SciPy thinks it is optimizing lambda, but we transform it to sigma
# before calculating the objective function:
def alt_nll(p, D):
    curr_mu    = p[0]
    curr_sigma = np.exp(p[1])   # <=== here's the change of variables
    ll = 0.
    for x in D:
        residual = x - curr_mu       
        ll      += stats.norm.logpdf(residual, 0, curr_sigma)    
        # Alternatively, using stats.norm.logpdf(x, curr_mu, curr_sigma) is shorter & equivalent,
        # but doing it in terms of the residual is a more direct analog to this week's homework.
    return -ll

# similarly our input p0 passes lambda = log(sigma):
alt_p0     = np.array([ 0.0, 0.0 ])

# and now we can call the minimizer without specifying bounds.
# Turns out that the SciPy minimizer will now complain about overflow
# errors ... it's pretty aggressive about exploring values of lambda, and 
# exp(lambda) can overflow easily. But it still works. To avoid
# scaring you with RuntimeWarning's on your screen, I'll shut them 
# off.
old_settings = np.seterr(over='ignore',invalid='ignore')
alt_result = optimize.minimize(alt_nll, alt_p0, (data))
np.seterr(**old_settings)

if alt_result.success != True:
    sys.exit("Alternate maximum likelihood fit failed")
mu_alt    = result.x[0]
sigma_alt = result.x[1]


print("Alternative ML fit, mu    = {0:.2f}".format(mu_alt))
print("Alternative ML fit, sigma = {0:.2f}".format(sigma_alt))
