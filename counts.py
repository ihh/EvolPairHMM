import numpy as np

import jax
import jax.numpy as jnp

from jax import grad, vmap
from jax.scipy.special import logsumexp, gammaln
from jax.scipy.linalg import expm

from functools import partial
from jax import jit

import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, ConstantStepSize, SaveAt

import argparse

# calculate SI
def insertions (t, lam, x):
    return jnp.exp (lam * t / (1. - x)) - 1.

# calculate derivatives of (A,B,F,Q)
def derivs (t, counts, params):
  lam,mu,x,y = params
  A,B,F,Q = counts
  S = insertions (t, lam, x)
  denom = S - y * (S - B - Q)
  num = mu * (B + Q)
  return jnp.where (t > 0.,
                    jnp.array (((mu*B*F*(1.-y)/denom - (lam+mu)*A,
                                 -B*num/denom + lam*(1.-B),
                                 -F*num/denom + lam*A,
                                 (S-Q)*num/denom))),
                    jnp.array ((-lam-mu,lam,lam,0.)))


def initCounts():
    return jnp.array ((1., 0., 0., 0.))
    
# calculate counts (A,B,F,Q) by numerical integration
def integrateCounts (t, params, /, step = None, rtol = None, atol = None, **kwargs):
  term = ODETerm(derivs)
  solver = Dopri5()
  if step is None and rtol is None and atol is None:
      raise Exception ("please specify step, rtol, or atol")
  if step is not None:
      stepsize_controller = ConstantStepSize()
  else:
      stepsize_controller = PIDController (rtol, atol)
  y0 = initCounts()
  sol = diffeqsolve (term, solver, 0., t, step, y0, args=params,
                     stepsize_controller=stepsize_controller,
                     saveat = SaveAt(steps=True),
                     **kwargs)
  return [(0., y0)] + list (zip (sol.ts, sol.ys))

# parse args
parser = argparse.ArgumentParser(description='Compute GGI counts by numerical integration')
parser.add_argument('--lambda', metavar='float', dest='lam', type=float, default=1.,
                    help='insertion rate')
parser.add_argument('--mu', metavar='float', type=float, default=1.,
                    help='deletion rate')
parser.add_argument('--x', metavar='float', type=float, default=.9,
                    help='insertion extension probability')
parser.add_argument('--y', metavar='float', type=float, default=.9,
                    help='deletion extension probability')
parser.add_argument('--t', metavar='float', type=float, default=1.,
                    help='maximum time')
parser.add_argument('--step', metavar='float', type=float, default=None,
                    help='time step')
parser.add_argument('--rtol', metavar='float', type=float, default=1e-3,
                    help='relative tolerance')
parser.add_argument('--atol', metavar='float', type=float, default=1e-6,
                    help='absolute tolerance')
parser.add_argument('--tderivs', action='store_true', help='show derivatives w.r.t. time')

args = parser.parse_args()

# do integration
params = (args.lam, args.mu, args.x, args.y)
result = integrateCounts (args.t, params, step=args.step, rtol=args.rtol, atol=args.atol)

vars = "A B F Q".split()
labels = "t S".split() + vars
if args.tderivs:
    labels += list (map (lambda v: "d" + v + "/dt", vars))
print (*labels)

for t, counts in result:
    if t <= args.t:
        if args.tderivs:
            d = tuple(derivs(t,counts,params))
        else:
            d = []
        print(t,insertions(t,args.lam,args.x),*counts,*d)
