# This is the proposed script for sampling demand that LLP created. It makes a very skewed distribution

library(MCMCpack)
library(MASS)
# Means per product
means <- c(10, 20)

# Number of commodities
K <- length(means)

# Generating covariance matrix with inverse Wishart distribution. What does that parameter do?
covar <- riwish(K, diag(K))

# Sampling X from a multivariate normal with the covariance from Wishart.
# It's the demand for each commodity at each sample
x <- mvrnorm(50000, rep(0,K), covar)

# Extract the probability density of the sampled values. Is the sqrt(diag(covar)) arbitrary?
px <- t(apply(x, 1, function(w) { pnorm(w, 0, sqrt(diag(covar))) }))

# Take those quantiles and plug them into a geometric. This is going to skew the data and project it into the range that we want starting at 0. 
#qgeom(x,prob). X is a vector of quantiles of the probability of failures in a Bernoulli (shape K). Second param is probabilities.  Why pz(1-pz)?? Something related to MLE?

# The original!!!!
vx <- t(apply(px, 1, function(w) { pz <- 1/means; qgeom(w, pz*(1-pz)) }))

# a test
#vx <- t(apply(px, 1, function(w) { pz <- 1/means; rgeom(w, pz*(1-pz)) }))
plot(vx)