indelParams = (0.1, 0.1, 0.9, 0.9)
t = 10
for ti in jnp.geomspace(.1,t):
  print(f"{ti=}")
  %time Ftransitions (ti, indelParams)

%time tm1=Ftransitions_diffrax (t, indelParams)
print(tm1)
%time tm2=Ftransitions_diffrax (t, indelParams, max_steps=10)
print(tm2)
%time tm3=Ftransitions_RK4 (t, indelParams)
print(tm3)




import matplotlib.pyplot as plt

term = ODETerm(Tderivs)
solver = Dopri5()
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
t0 = 0
t1 = 5
dt0 = 0.01
T0 = jnp.array([1,0,0,0])
params = (0.1, 0.1, 0.9, 0.9)
[lam,mu,x,y] = params
saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
sol = diffeqsolve(term, solver, t0, t1, dt0, T0, args=params, saveat=saveat,
                  stepsize_controller=stepsize_controller)

SIs = [SID(t,lam,x) for t in sol.ts]
SDs = [SID(t,mu,y) for t in sol.ts]

#r = [(SDs[n] + sol.ys[0][n] + sol.ys[1][n] - sol.ys[3][n] - 1) / SDs[n] for n in jnp.arange(len(SIs))]
#plt.plot(sol.ts, r, label="r")

plt.plot(sol.ts, sol.ys[:,0], label="TMM")
plt.plot(sol.ts, sol.ys[:,1], label="TMI")
plt.plot(sol.ts, sol.ys[:,2], label="TIM")
plt.plot(sol.ts, sol.ys[:,3], label="TDI")
#plt.plot(sol.ts, SIs, label="SI")
#plt.plot(sol.ts, SDs, label="SD")
plt.legend()
plt.show()



def compareRates (params):
  t0=0
  T0=jnp.array([1,0,0,0])
  #print (Ftransitions_from_counts(t,T0,params))
  Td0 = Tderivs(t0,T0,params) # TMM,TMI,TIM,TDI
  print (f"{Td0=}")

  T0_3x3=jnp.array([1,0,0,0,0,0,0,0,0])
  #print (Ftransitions_from_counts_3x3(t,T0_3x3,params))
  Td0_3x3 = Tderivs_3x3(t0,T0_3x3,params) # TMM,TMI,TMD,TIM,TII,TID,TDM,TDI,TDD
  print (f"{Td0_3x3=}")

  t1=.01
  T1=T0+Td0*(t1-t0)
  T1_3x3=T0_3x3+Td0_3x3*(t1-t0)
  Td1 = Tderivs(t1,T1,params) # TMM,TMI,TIM,TDI
  Td1_3x3 = Tderivs_3x3(t1,T1_3x3,params) # TMM,TMI,TMD,TIM,TII,TID,TDM,TDI,TDD

  print (f"{Td1=}")
  print (f"{Td1_3x3=}")

compareRates ((0.1, 0.1, 0.95, 0.95))




# Some debugging output to check by inspection that results are sensible
def compareGrads (t, params):
  print (Ftransitions_RK4(t,params))

  # we can only autodiff a function returning a scalar, hence this wrapper
  def a_RK4 (t: Float, params: Float[Array, "4"]):
    return Ftransitions_RK4(t,params)[0,0]

  print (grad(a_RK4,[0,1])(t,params))

  # now try diffrax
  print (Ftransitions_diffrax(t,params))

  # we can only autodiff a function returning a scalar, hence this wrapper
  def a_diffrax (t: Float, params: Float[Array, "4"]):
    return Ftransitions_diffrax(t,params)[0,0]

  print (grad(a_diffrax,[0,1])(t,params))

  # now try using numeric version
  def a_3x3 (t: Float, params: Float[Array, "4"]):
    return Ftransitions_3x3(t,params)[0,0]

  print (Ftransitions_3x3(t,params))
  print (grad(a_3x3,[0,1])(t,params))

compareGrads (0.5, (0.1, 0.1, 0.95, 0.95))




def a_TKF91 (t: Float, lam: Float, mu: Float):
  return Ftransitions_TKF91(t,lam,mu)[0,0]

def compareGrads_TKF (t, lam, mu):
  params = lam,mu,0.,0.
  compareGrads (t, params)
  print (Ftransitions_TKF91(t,lam,mu))
  print (grad(a_TKF91,[0,1,2])(t,lam,mu))

compareGrads_TKF (0.5, 0.09, 0.1)




dna = "acgt"
xstr = "aacg"
ystr = "aagt"
x = dsq (xstr, dna)
y = dsq (ystr, dna)
x1 = onehot (xstr, dna)
y1 = onehot (ystr, dna)
t = 1.
indelParams = (.05,.05,.9,.9)
substParams = hky85 ([.25,.25,.25,.25], 10, 1)
%time print ("Impure Forward (1hot):", forward_wrap(forward_impure_1hot) (x1, y1, t, indelParams, substParams))
%time print ("Impure Forward:", forward_impure_wrap (x, y, t, indelParams, substParams))
%time print ("Inefficient Forward:", forward_inefficient_wrap (x, y, t, indelParams, substParams))
%time print ("grad(Inefficient Forward):", forward_inefficient_wrap_grad (x, y, t, indelParams, substParams))
def num_grad_fwd_t (x, y, t, indelParams, substParams):
  dt = .01
  return (forward_impure_wrap (x, y, t + dt, indelParams, substParams) - forward_impure_wrap (x, y, t, indelParams, substParams)) / dt
def num_grad_fwd_lambda (x, y, t, indelParams, substParams):
  dl = .00001
  indelParams_plus_dl = (indelParams[0] + dl, indelParams[1], indelParams[2], indelParams[3])
  return (forward_impure_wrap (x, y, t, indelParams_plus_dl, substParams) - forward_impure_wrap (x, y, t, indelParams, substParams)) / dl
%time print ("numerical gradient_t(Impure Forward):", num_grad_fwd_t (x, y, t, indelParams, substParams))
%time print ("numerical gradient_lambda(Impure Forward):", num_grad_fwd_lambda (x, y, t, indelParams, substParams))




params = (0.1, 0.1, 0.9, 0.9)
t = 0.5
transmat = Ftransitions (t, params)
nDel, nIns = 10,10
%time gp_comb = gap_prob (nDel, nIns, transmat)
%time gp_dp = gap_prob_impure_dp (nDel, nIns, transmat)
print (f"{gp_comb=}")
print (f"{gp_dp=}")




s = summarize_alignment("aaatcc--g",
                        "-gggtccc-",
                        "acgt")
print(s)

t = 1.
indelParams = (.05,.05,.9,.9)
substParams = hky85 ([.25,.25,.25,.25], 10, 1)
print (alignment_likelihood (s, t, indelParams, substParams))

alg = grad(alignment_likelihood,argnums=(1,2,3))
print(alg(s, t, indelParams, substParams))




