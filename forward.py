import jax
import jax.numpy as jnp

from jax import grad, value_and_grad
from jax.scipy.special import gammaln, logsumexp
from jax.scipy.linalg import expm

from functools import partial
from jax import jit

import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, ConstantStepSize, SaveAt

import argparse

# We hate NaNs
from jax.config import config
config.update("jax_debug_nans", True)

# We replace zeroes and infinities with small numbers sometimes
min_float32 = jnp.finfo('float32').min
smallest_float32 = jnp.finfo('float32').smallest_normal

# calculate L, M
def lm (t, rate, prob):
  return jnp.exp (-rate * t / (1. - prob))

def indels (t, rate, prob):
  return 1. / lm(t,rate,prob) - 1.

# calculate derivatives of (a,b,u,q)
def derivs (t, counts, indelParams):
  lam,mu,x,y = indelParams
  a,b,u,q = counts
  L = lm (t, lam, x)
  M = lm (t, mu, y)
  num = mu * (b*M + q*(1.-M))
  unsafe_denom = M*(1.-y) + L*q*y + L*M*(y*(1.+b-q)-1.)
  denom = jnp.where (unsafe_denom > 0., unsafe_denom, 1.)   # avoid NaN gradient at zero
  one_minus_m = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
  return jnp.where (unsafe_denom > 0.,
                    jnp.array (((mu*b*u*L*M*(1.-y)/denom - (lam+mu)*a,
                                 -b*num*L/denom + lam*(1.-b),
                                 -u*num*L/denom + lam*a,
                                 ((M*(1.-L)-q*L*(1.-M))*num/denom - q*lam/(1.-y))/one_minus_m))),
                    jnp.array ((-lam-mu,lam,lam,0.)))

# calculate counts (a,b,u,q) by numerical integration
def initCounts(indelParams):
    return jnp.array ((1., 0., 0., 0.))
    
def integrateCounts (t, indelParams, /, step = None, rtol = None, atol = None, **kwargs):
  term = ODETerm(derivs)
  solver = Dopri5()
  if step is None and rtol is None and atol is None:
      raise Exception ("please specify step, rtol, or atol")
  if step is not None:
      stepsize_controller = ConstantStepSize()
  else:
      stepsize_controller = PIDController (rtol, atol)
  y0 = initCounts(indelParams)
  sol = diffeqsolve (term, solver, 0., t, step, y0, args=indelParams,
                     stepsize_controller=stepsize_controller,
                     **kwargs)
  return sol.ys[-1]

# Runge-Kutte (RK4) numerical integration routine
# This is retained solely to have a simpler routine independent of diffrax, if needed for debugging
# Currently we just use integrateCounts instead, so this function is never called
def integrateCounts_RK4 (t, indelParams, /, steps=10, dt0=None):
  lam,mu,x,y = indelParams
  def RK4body (y, t_dt):
    t, dt = t_dt
    k1 = derivs(t, y, indelParams)
    k2 = derivs(t+dt/2, y + dt*k1/2, indelParams)
    k3 = derivs(t+dt/2, y + dt*k2/2, indelParams)
    k4 = derivs(t+dt, y + dt*k3, indelParams)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6, None
  y0 = initCounts (indelParams)
  if dt0 is None:
    dt0 = 0.1 / jnp.maximum (lam, mu)
  ts = jnp.geomspace (dt0, t, num=steps)
  ts_with_0 = jnp.concatenate ([jnp.array([0]), ts])
  dts = jnp.ediff1d (ts_with_0)
  y1, _ = jax.lax.scan (RK4body, y0, (ts_with_0[:-1],dts))
  return y1

# test whether time is past threshold of alignment signal being undetectable
def alignmentIsProbablyUndetectable (t, indelParams, alphabetSize):
    lam,mu,x,y = indelParams
    expectedMatchRunLength = 1. / (1. - jnp.exp(-mu*t))
    expectedInsertions = indels(t,lam,x)
    expectedDeletions = indels(t,mu,y)
    kappa = 2.
    return jnp.where (t > 0.,
                      ((expectedInsertions + 1) * (expectedDeletions + 1)) > kappa * (alphabetSize ** expectedMatchRunLength),
                      False)

# initial transition matrix
def zeroTimeTransitionMatrix (indelParams):
  lam,mu,x,y = indelParams
  return jnp.array ([[1.,0.,0.],
                     [1.-x,x,0.],
                     [1.-y,0.,y]])

# convert counts (a,b,u,q) to transition matrix ((a,b,c),(f,g,h),(p,q,r))
def smallTimeTransitionMatrix (t, indelParams, /, **kwargs):
    lam,mu,x,y = indelParams
    a,b,u,q = integrateCounts(t,indelParams,**kwargs)
# To use the non-diffrax version, comment out the previous line and uncomment the following one:
#    a,b,u,q = integrateCounts_RK4(t,indelParams,dt0=.1/jnp.maximum(lam,mu))
    L = lm(t,lam,x)
    M = lm(t,mu,y)
    one_minus_L = jnp.where (L < 1., 1. - L, smallest_float32)   # avoid NaN gradient at zero
    one_minus_M = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
    return jnp.array ([[a,b,1-a-b],
                      [u*L/one_minus_L,1-(b+q*(1-M)/M)*L/one_minus_L,(b+q*(1-M)/M-u)*L/one_minus_L],
                       [(1-a-u)*M/one_minus_M,q,1-q-(1-a-u)*M/one_minus_M]])

# get limiting transition matrix for large times
def largeTimeTransitionMatrix (t, indelParams):
    lam,mu,x,y = indelParams
    g = 1. - lm(t,lam,x)
    r = 1. - lm(t,mu,y)
    return jnp.array ([[(1-g)*(1-r),g,(1-g)*r],
                       [(1-g)*(1-r),g,(1-g)*r],
                       [(1-r),0,r]])

# get transition matrix for any given time
def transitionMatrix (t, indelParams, /, alphabetSize=20, **kwargs):
    lam,mu,x,y = indelParams
    return jnp.where (t > 0.,
                      jnp.where (alignmentIsProbablyUndetectable(t,indelParams,alphabetSize),
                                 largeTimeTransitionMatrix(t,indelParams),
                                 smallTimeTransitionMatrix(t,indelParams,**kwargs)),
                      zeroTimeTransitionMatrix(indelParams))
  
# get equilibrium of substitution rate matrix by finding an "effectively infinite" time value, multiplying by that, and exponentiating
# this is ugly, but it works...
def get_eqm (submat):
  large_t = 1 / jnp.min(jnp.abs(submat)[jnp.nonzero(submat,size=1)])
  return jnp.matmul (expm (submat * large_t), jnp.ones (submat.shape[0]))

# normalize a substitution rate matrix to have one expected substitution per unit of time
def normalize_rate_matrix (mx):
  mx_abs = jnp.abs (mx)
  mx_diag = jnp.diagonal (mx_abs)
  mx_no_diag = mx_abs - jnp.diag (mx_diag)
  mx_rowsums = mx_no_diag @ jnp.ones_like (mx_diag)
  return mx_no_diag - jnp.diag (mx_rowsums)

# calculate submat=exp(Q*t) and pi=equilibrium distribution of Q
def calc_submat_pi (t, substRateMatrix):
  Q = normalize_rate_matrix (substRateMatrix)
  pi = get_eqm (Q)
  submat = expm (Q * t)
  return submat, pi

# calculate finite-time probability matrices associated with a substitution model
def calc_transmat_submat_pi (t, indelParams, substRateMatrix, /, **kwargs):
  submat, pi = calc_submat_pi (t, substRateMatrix)
  transmat = transitionMatrix (t, indelParams, pi.shape[0], **kwargs)
  return transmat, submat, pi

# safe log wrappers (in the sense that they don't create NaN errors when taking gradients at 0)
def safe_log (x):
  return jnp.where (x > 0., jnp.log (jnp.where (x > 0., x, min_float32)), -jnp.inf)

def safe_logsumexp(a):
  c = jnp.max(a)
  safe = c > min_float32
  a = jnp.where(safe, a, jnp.array([0]))
  return jnp.where (safe, jnp.log(jnp.sum(jnp.exp(a))), -jnp.inf)

logsumexp = safe_logsumexp
log = safe_log

# function to remove -infinity with -1e38 (an icky compromise, to avoid rewriting too much of jax.numpy)
def remove_neginfs (x):
  return jnp.where (x is -jnp.inf, min_float32, x)

# pure, but inefficient, jax implementation of Forward algorithm
# x is the ancestor, laid out horizontally (i.e. each row compares all of x to a single site of y)
# y is the descendant, laid out vertically (i.e. each column compares all of y to a single site of x)
# The states are M(0), I(1), D(2)
def forward_1hot (x, y, transmat, submat, pi, /, **kwargs):
  [[a,b,c],[f,g,h],[p,q,r]] = log (transmat)  # perform calculations in log-space
  lsm = remove_neginfs (log (submat) - log (pi))  # replacing -infinity with -1e38 is a compromise, but avoids having to rewrite jnp.matmul to catch 0*infinity
  def fillCell (cellCarry, col):
    M_src_cell, D_src_cell, yc = cellCarry
    I_src_cell, xc = col[0:3], col[3:]
    cell = jnp.array ([logsumexp (jnp.array ([M_src_cell[0] + a,
                                              M_src_cell[1] + f,
                                              M_src_cell[2] + p])
                        + jnp.matmul (jnp.matmul (xc, lsm), yc)),
                       logsumexp (jnp.array ([I_src_cell[0] + b,
                                              I_src_cell[1] + g,
                                              I_src_cell[2] + q])),
                       logsumexp (jnp.array ([D_src_cell[0] + c,
                                              D_src_cell[1] + h,
                                              D_src_cell[2] + r]))])
    return (I_src_cell, cell, yc), cell
  def fillRow (rowCarry, yc):
    prevRow, x = rowCarry
    firstCellInRow = jnp.array([-jnp.inf,
                                logsumexp (jnp.array ([prevRow[0,0] + b,
                                                      prevRow[0,1] + g])),
                                -jnp.inf])
    _finalCellCarry, restOfRow = jax.lax.scan (fillCell,
                                               (prevRow[0], firstCellInRow, yc),
                                               jnp.concatenate ([prevRow[1:], x], 1))
    row = jnp.concatenate ([jnp.array ([firstCellInRow]), restOfRow])
    return (row, x), row
  def fillFirstRow (prevCell, _xc):
    cell = jnp.array ([-jnp.inf,
                       -jnp.inf,
                       logsumexp (jnp.array ([prevCell[0] + c,
                                              prevCell[2] + r]))])
    return cell, cell
  firstCell = jnp.array ([0,-jnp.inf,-jnp.inf])
  _lastCellInFirstRow, restOfFirstRow = jax.lax.scan (fillFirstRow,
                                                      firstCell,
                                                      x)
  firstRow = jnp.concatenate ([jnp.array ([firstCell]), restOfFirstRow])
  lastRowCarry, _restOfRows = jax.lax.scan (fillRow,
                                            (firstRow,x),
                                            y)
  lastRow = lastRowCarry[0]
  return logsumexp (jnp.array ([lastRow[-1,0] + logsumexp(jnp.array ([a,c])),
                                lastRow[-1,1] + logsumexp(jnp.array ([f,h])),
                                lastRow[-1,2] + logsumexp(jnp.array ([p,r]))]))

# wrapper that computes the finite-time probabilities before calling forward_1hot
def forward_1hot_wrap (x, y, t, indelParams, substRateMatrix, /, **kwargs):
  transmat, submat, pi = calc_transmat_submat_pi (t, indelParams, substRateMatrix, **kwargs)
  return forward_1hot (x, y, transmat, submat, pi, **kwargs)

# null model sequence (log) probability
def null_model_prob_1hot (seq_1hot, pi):
  return jnp.sum (jnp.dot (seq_1hot, log(pi)))

# convert a DNA string to a one-hot encoded array
dna_alphabet = "acgt"
def one_hot_dna (str):
    return jax.nn.one_hot ([dna_alphabet.index(x) for x in str.lower()], 4)

# The Hasegawa-Kishino-Yano (1985) substitution rate matrix
def hky85 (eqm, ti, tv):
  idx = range(4)
  raw = [[eqm[j] * (ti if i & 1 == j & 1 else tv) for j in idx] for i in idx]
  return normalize_rate_matrix (jnp.array (raw))


# parse args
parser = argparse.ArgumentParser(description='Compute logP(descendant|ancestor) under GGI/HKY85 model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ins-rate', metavar='float', dest='lam', type=float, default=.1,
                    help='insertion rate')
parser.add_argument('--del-rate', metavar='float', dest='mu', type=float, default=.1,
                    help='deletion rate')
parser.add_argument('--ins-extend', metavar='float', dest='x', type=float, default=.9,
                    help='insertion extension probability')
parser.add_argument('--del-extend', metavar='float', dest='y', type=float, default=.9,
                    help='deletion extension probability')
parser.add_argument('--time', metavar='float', dest='t', type=float, nargs='*', default=[0.1],
                    help='evolutionary time parameter')
parser.add_argument('--transition', metavar='float', type=float, default=1.,
                    help='relative rate of transition substitutions')
parser.add_argument('--transversion', metavar='float', type=float, default=1.,
                    help='relative rate of transversion substitutions')
parser.add_argument('--gc', metavar='float', type=float, default=.5,
                    help='GC content at equilibrium')
parser.add_argument('--ancestor', metavar='string', type=str, required=True,
                    help='ancestral DNA sequence')
parser.add_argument('--descendant', metavar='string', type=str, required=True,
                    help='descendant DNA sequence')
parser.add_argument('--step', metavar='float', type=float, default=None,
                    help='time step for fixed-step numerical integration')
parser.add_argument('--rtol', metavar='float', type=float, default=1e-3,
                    help='relative tolerance for variable-step numerical integration')
parser.add_argument('--atol', metavar='float', type=float, default=1e-6,
                    help='absolute tolerance for variable-step numerical integration')
parser.add_argument('--derivs', action='store_true',
                    help='show derivatives of log-likelihood')

args = parser.parse_args()

# do integration
params = { "lambda": args.lam,
           "mu": args.mu,
           "x": args.x,
           "y": args.y,
           "gc": args.gc,
           "ti": args.transition,
           "tv": args.transversion }
ancestor = one_hot_dna (args.ancestor)
descendant = one_hot_dna (args.descendant)
diffraxArgs = { "step": args.step,
                "rtol": args.rtol,
                "atol": args.atol }

@jax.jit
def logLikelihood(params):
  indelParams = (params["lambda"], params["mu"], params["x"], params["y"])
  gc, ti, tv = params["gc"], params["ti"], params["tv"]
  substRateMatrix = hky85 ([(1-gc)/2, gc/2, gc/2, (1-gc)/2], ti, tv)
  return forward_1hot_wrap (ancestor, descendant, params["t"], indelParams, substRateMatrix, **diffraxArgs)

ll_grad = jax.jit (value_and_grad (logLikelihood))

for t in args.t:
  params["t"] = t
  if args.derivs:
    logLike, derivs = ll_grad (params)
    print (f"t={t} L={logLike} "
           + " ".join(f"dL/d({x})={derivs[x]}" for x in ["t","lambda","mu","x","y","gc","ti","tv"]))
  else:
    logLike = logLikelihood (params)
    print (f"t={t} L={logLike}")
