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

# calculate S, D
def indels (t, rate, prob):
  return jnp.exp (rate * t / (1. - prob)) - 1.

def insertions (t, args):
  return indels(t,args.lam,args.x)

def deletions (t, args):
  return indels(t,args.mu,args.y)

# calculate derivatives of (a,b,u,q)
def derivs (t, counts, params):
  lam,mu,x,y = params
  a,b,u,q = counts
  S = indels (t, lam, x)
  D = indels (t, mu, y)
  denom = S - y * (S - b - q*D)
  num = mu * (b + q*D)
  return jnp.where (t > 0.,
                    jnp.array (((mu*b*u*(1.-y)/denom - (lam+mu)*a,
                                 -b*num/denom + lam*(1.-b),
                                 -u*num/denom + lam*a,
                                 ((S-q*D)*num/denom - q*lam*(D+1)/(1.-y))/D))),
                    jnp.array ((-lam-mu,lam,lam,0.)))


def initCounts(params):
    return jnp.array ((1., 0., 0., 0.))
    
# calculate counts (a,b,u,q) by numerical integration
def integrateCounts (t, params, /, step = None, rtol = None, atol = None, **kwargs):
  term = ODETerm(derivs)
  solver = Dopri5()
  if step is None and rtol is None and atol is None:
      raise Exception ("please specify step, rtol, or atol")
  if step is not None:
      stepsize_controller = ConstantStepSize()
  else:
      stepsize_controller = PIDController (rtol, atol)
  y0 = initCounts(params)
  sol = diffeqsolve (term, solver, 0., t, step, y0, args=params,
                     stepsize_controller=stepsize_controller,
                     saveat = SaveAt(steps=True),
                     **kwargs)
  return [(0., y0)] + list (zip (sol.ts, sol.ys))

# parse args
parser = argparse.ArgumentParser(description='Compute GGI counts by numerical integration',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

vars = "a b u q".split()
labels = "t S D".split() + vars
if args.tderivs:
    labels += list (map (lambda v: "d" + v + "/dt", vars))
print (*labels)

for t, counts in result:
    if t <= args.t:
        if args.tderivs:
            d = tuple(derivs(t,counts,params))
        else:
            d = []
        print(t,insertions(t,args),deletions(t,args),*counts,*d)
