import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Implements the Forward algorithm for an HMM as a scan (both generic and associative).
# Inputs:
#    t[i,j] = transition probability from state i to state j
#    e[i,o] = probability of emitting observation o from state i
#  start[i] = initial probability of state i (transitions from start state)
#    end[i] = probability of ending from state i (transitions to end state)

# First using jax.lax.scan
def hmm_forward_scan(params, obs):
    t, e, start, end = params
    def scan_fn(prev, o):
        return jnp.einsum ('i,ij,j->j', prev, t, e[:,o]), None
    return jnp.dot(jax.lax.scan(scan_fn, start, obs, length=obs.size)[0], end)

# Next using jax.lax.associative_scan
def hmm_forward(params, obs):
    t, e, start, end = params
    u = jnp.einsum('ij,jn->nij', t, jnp.take(e, obs, axis=1))
    return start @ jax.lax.associative_scan (jnp.matmul, u)[-1] @ end

# Another associative scan, but in log space
def hmm_forward_log(params, obs):
    t, e, start, end = params
    logu = jnp.log (jnp.einsum('ij,jn->nij', t, jnp.take(e, obs, axis=1)))
    logu_prod = jax.lax.associative_scan (log_matmul, logu)[-1]
    return log_matmul (log_matmul (jnp.log(start)[None,:], logu_prod), jnp.log(end)[:,None])[0,0]

# Log space matrix product via https://stackoverflow.com/a/74409968
# Equivalent of jnp.log (jnp.einsum('...ij,...jk->...ik', jnp.exp(A), jnp.exp(B)))
def log_matmul(A,B):
    assert A.ndim == B.ndim
    Astack = jnp.einsum ('k...ij->...ikj', jnp.stack([A]*B.shape[-1]))
    Bstack = jnp.einsum ('i...jk->...ikj', jnp.stack([B]*A.shape[-2]))
    return logsumexp(Astack+Bstack, axis=-1)

# Finally a reference implementation using a loop
def hmm_forward_ref(params, obs):
    t, e, start, end = params
    F = start
    for o in obs:
        F = jnp.dot(F, t) * e[:,o]
    return jnp.dot(F, end)

# Test the implementations
params = (jnp.array([[0.9, 0.1], [0.2, 0.8]]), jnp.array([[0.7, 0.3], [0.4, 0.6]]), jnp.array([0.55, 0.45]), jnp.array([0.52, 0.48]))
obs = jnp.array([0, 1, 0, 1, 1, 0])
print("Loop:",hmm_forward_ref(params, obs))
print("Scan:",hmm_forward_scan(params, obs))
print("Assoc:",hmm_forward(params, obs))
print("LogAssoc:",jnp.exp(hmm_forward_log(params, obs)))
