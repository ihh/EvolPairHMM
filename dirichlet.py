import jax
import jax.numpy as jnp

# some non-jax imports for testing
import numpy
import random
import argparse

# Generate sample from Dirichlet distribution "differentiably"
# ... i.e. in a way that can be backpropagated through, using the reparameterization trick.
# Sources:
#  1. The Reparameterization Trick (2018)
#     https://gregorygundersen.com/blog/2018/04/29/reparameterization/
#  2. Auto-encoding Variational Bayes (2013)
#     Kingma, D. P., & Welling, M.
#     https://arxiv.org/abs/1312.6114
#  3. Dirichlet distribution
#     https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_variate_generation
#  4. Choice of Basis for Laplace Approximation (1998)
#     MacKay, D. J. C.
#     https://www.inference.org.uk/mackay/laplace.pdf
#  5. Autoencoding Variational Inference For Topic Models (2017)
#     Srivastava A., Sutton C.
#     https://arxiv.org/abs/1703.01488

# The formula for generating p ~ Dir(alpha) is (per Section 3.2 of Srivastava and Sutton):
#          p = softmax(a)
#        a_i = mu_i + sqrt(Sigma_ii) * n_i
#       mu_i = log(alpha_i) - (1/K) * sum(log(alpha_k))
#   Sigma_ii = (1/alpha_i) * (1 - 2/K) + (1/K^2) * sum(1/alpha_i)
#
# with   n_i ~ Normal(0,1)
#
# This is obtained by fitting a Gaussian to each axis at the mode (MacKay, 1998).

def transform_multivariate_normal_to_dirichlet (n, alpha):
    K = alpha.shape[0]
    assert n.shape[0] == K
    mu = jax.lax.log(alpha) - jnp.sum (jax.lax.log(alpha)) / K
    Sigma = (1/alpha) * (1 - 2/K) + (1/K**2) * jnp.sum(1/alpha)
    a = mu + jnp.sqrt(Sigma) * n
    return jax.nn.softmax(a)

def sample_dirichlet (key, alpha):
    n = jax.random.normal (key, shape=alpha.shape)
    return transform_multivariate_normal_to_dirichlet (n, alpha)

# Non-differentiable reference implementation
def sample_dirichlet_ref (alpha):
    gammas = [numpy.random.gamma(a, 1) for a in alpha]
    return gammas / numpy.sum(gammas)

# Test the implementation by discretizing probability space and comparing counts
# It is presumably possible to get a more accurate estimate of the K-L divergence analytically, but I haven't tried
parser = argparse.ArgumentParser (formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--bins", type=int, default=10, help="Number of bins for discretization")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dims", type=int, default=3, help="Number of dimensions for Dirichlet distribution")
parser.add_argument("--max_count", type=int, default=5, help="Maximum alpha for each dimension")
parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
parser.add_argument("--pseudocount", type=float, default=1e-5, help="Pseudocount for estimating K-L divergence")
parser.add_argument("--verbose", action="store_true", help="Print samples and bins")
args = parser.parse_args()

n_samples = args.samples
dims = args.dims
seed = args.seed
bins = args.bins
max_count = args.max_count
pseudocount = args.pseudocount
verbose = args.verbose

keys = jax.random.split (jax.random.PRNGKey(seed), n_samples)
random.seed(seed)

alpha = [random.random() * max_count for _ in range(dims)]
print ("alpha:", alpha)

jax_samples = [sample_dirichlet (key, jnp.array(alpha)) for key in keys]
python_samples = [sample_dirichlet_ref(alpha) for _ in range(n_samples)]
if verbose:
    print ("JAX:", jax_samples[:5])
    print ("Python:", python_samples[:5])

jax_bins = jnp.histogramdd(jnp.array(jax_samples), bins=bins, range=[(0,1)]*dims)[0]
python_bins = jnp.histogramdd(jnp.array(python_samples), bins=bins, range=[(0,1)]*dims)[0]
if verbose:
    print ("JAX bins:", jax_bins)
    print ("Python bins:", python_bins)

kl = 0
norm = n_samples + pseudocount * bins**dims
for index, pc in numpy.ndenumerate(python_bins):
    jc = jax_bins[index]
    pp = (pc + pseudocount) / norm
    jp = (jc + pseudocount) / norm
    kl += pp * (jnp.log2(pp) - jnp.log2(jp))
    if pc > 0:
        print ([x/bins for x in index], f"{pp:.4f}", f"{jp:.4f}")

print ("Kullback-Leibler divergence:", kl, "bits")
