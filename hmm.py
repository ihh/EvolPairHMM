import jax
import jax.numpy as jnp

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
    u = jnp.einsum('ij,nj->nij', t, jnp.take(e, obs, axis=1).transpose())
    def associative_scan_fn(a, b):
        return jnp.matmul (b, a)
    return start @ jax.lax.associative_scan (associative_scan_fn, u)[-1] @ end

# Finally a reference implementation using loops
def hmm_forward_ref(params, obs):
    t, e, start, end = params
    F = start
    for o in obs:
        F = jnp.dot(F, t) * e[:,o]
    return jnp.dot(F, end)

# Test the implementations
params = (jnp.array([[0.9, 0.1], [0.2, 0.8]]), jnp.array([[0.7, 0.3], [0.4, 0.6]]), jnp.array([0.55, 0.45]), jnp.array([0.52, 0.48]))
obs = jnp.array([0, 1, 0, 1, 0])
print("Loop:",hmm_forward_ref(params, obs))
print("Scan:",hmm_forward_scan(params, obs))
print("Assoc:",hmm_forward(params, obs))
