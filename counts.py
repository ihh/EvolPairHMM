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
    return jnp.exp (lam * t / (1. - x)) - 1

# calculate derivatives of (logA,B,F,Q)
def derivs (t, counts, params):
  lam,mu,x,y = params
  A,B,F,Q = counts
  #A = jnp.exp (logA)
  S = insertions (t, lam, x)
  denom = S - y * (S - B - Q)
  num = mu * (B + Q)
  return jnp.where (t > 0.,
                    jnp.array (((mu*B*F*(1.-y)/denom - (lam+mu)*A,
                                 -B*num/denom + lam*(1-B),
                                 -F*num/denom + lam*A,
                                 (S-Q)*num/denom))),
                    jnp.array ((-lam-mu,lam,lam,0.)))


def initCounts():
    return jnp.array ((1., 0., 0., 0.))
    
# calculate counts (logA,B,F,Q) by numerical integration with diffrax
def integrateCounts_diffrax (t, params, /, step = None, rtol = None, atol = None, **kwargs):
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

# calculate counts (logA,B,F,Q) by numerical integration with handrolled RK4
def RK4 (t, params, y0, derivs, steps=10, dt0=0.01):
  def RK4body (y, t_dt):
    t, dt = t_dt
    k1 = derivs(t, y, params)
    k2 = derivs(t+dt/2, y + dt*k1/2, params)
    k3 = derivs(t+dt/2, y + dt*k2/2, params)
    k4 = derivs(t+dt, y + dt*k3, params)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6, None
  ts = jnp.geomspace (dt0, t, num=steps)
  ts_with_0 = jnp.concatenate ([jnp.array([0]), ts])
  dts = jnp.ediff1d (ts_with_0)
  y1, _ = jax.lax.scan (RK4body, y0, (ts_with_0[0:-1],dts))
  return y1

# this is super inefficient, it recalculates the whole range at every step
def integrateCounts_RK4 (t, params, /, step = None, rtol = None, atol = None, **kwargs):
  if step is None:
      raise Exception ("please specify step")
  ts = (jnp.arange((t+1)/step) + 1.) * step
  y0 = initCounts()
  ys = list (map (lambda t: RK4(t,params,y0,derivs), ts))
  return [(0., y0)] + list (zip (ts, ys))

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
result = integrateCounts_diffrax (args.t, params, step=args.step, rtol=args.rtol, atol=args.atol)

vars = "A B F Q".split()
labels = "t S".split() + vars
if args.tderivs:
    labels += list (map (lambda v: "d" + v + "/dt", vars))
print (*labels)

for t, countsTransform in result:
#    (logA,B,F,Q) = countsTransform
 #   A = jnp.exp(logA)
 #   counts = [A,B,F,Q]
    counts = countsTransform
    if t <= args.t:
        if args.tderivs:
            d = tuple(derivs(t,counts,params))
#            dAdt = dlogAdt * A
#            d = [dAdt,dBdt,dFdt,dQdt]
        else:
            d = []
        print(t,insertions(t,args.lam,args.x),*counts,*d)
