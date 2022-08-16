


import jax.numpy as jnp

def fwd_solver(f, z_init):
  z_prev, z = z_init, f(z_init)
  while jnp.linalg.norm(z_prev - z) > 1e-5:
    z_prev, z = z, f(z)
  return z


import jax

def newton_solver(f, z_init):
  f_root = lambda z: f(z) - z
  g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
  return fwd_solver(g, z_init)


def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
  x0 = z_init
  x1 = f(x0)
  x2 = f(x1)
  X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
  F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

  res = []
  for k in range(2, max_iter):
    n = min(k, m)
    G = F[:n] - X[:n]
    GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
    H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
                   [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
    alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]

    xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
    X = X.at[k % m].set(xk)
    F = F.at[k % m].set(f(xk))

    res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))
    if res < tol:
      break
  return xk


def fixed_point_layer(solver, f, params, x):
  z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
  return z_star


f = lambda W, x, z: jnp.tanh(jnp.dot(W, z) + x)

from jax import random

ndim = 10
W = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
x = random.normal(random.PRNGKey(1), (ndim,))

z_star = fixed_point_layer(fwd_solver, f, W, x)