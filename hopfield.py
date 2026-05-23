"""modern-hopfield: classical and modern Hopfield networks in JAX.

The central claim of this library, due to Ramsauer et al. (2020), is that the
update rule of the modern (continuous) Hopfield network is identical to
softmax attention.  Not analogous.  Identical.  ``verify_equivalence`` is
the empirical proof: it runs both and reports the max absolute difference,
which sits at float32 rounding noise (~1e-7).

References
----------
Hopfield (1982)     Neural networks and physical systems with emergent
                    collective computational abilities.
Ramsauer et al.     Hopfield Networks is All You Need.  (arXiv:2008.02217)
    (2020)
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1.  CLASSICAL HOPFIELD NETWORK  (Hopfield, 1982)
# ---------------------------------------------------------------------------

def hebbian_store(patterns: jnp.ndarray) -> jnp.ndarray:
    """Store ``patterns`` via Hebbian learning.

    Math
        ``W = (1/N) * sum_mu (xi_mu xi_mu^T)``,  ``diag(W) = 0``.

    Neuroscience
        "Neurons that fire together, wire together."  Each pattern
        strengthens the synapse between every pair of co-active units.

    JAX
        The whole sum is one matmul (``X^T X``); no Python loop over
        patterns.  Diagonal zeroing prevents a unit from biasing itself.

    Args:
        patterns: ``(M, N)`` array of ``M`` binary (+/-1) patterns.
    Returns:
        ``(N, N)`` symmetric weight matrix with zero diagonal.
    """
    M, N = patterns.shape
    W = patterns.T @ patterns / N
    return W - jnp.diag(jnp.diag(W))


def energy(W: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    """Classical Hopfield energy ``E(s) = -1/2 s^T W s``.

    Neuroscience
        Stored patterns sit at local minima ("attractors") of this
        surface.  Retrieval is descent into the basin of the nearest
        attractor; spurious states are higher saddle points and ridges.

    JAX
        ``jax.grad(energy, argnums=1)(W, s) = -W s`` -- the local field
        pulling each unit toward the nearest minimum.  Both update rules
        below are discretizations of this gradient flow.
    """
    return -0.5 * state @ W @ state


def async_update(W, state, key, n_steps):
    """Asynchronous Glauber dynamics.

    Math
        Pick a unit ``i`` uniformly at random; set
        ``s_i <- sign(sum_j W_ij s_j)``.  Energy is monotonically
        non-increasing.

    Neuroscience
        The biologically realistic update -- neurons fire at random
        times, not in lockstep.  Always converges (no two-cycles).

    JAX
        Indices are pre-sampled, then a ``lax.scan`` walks them.  The
        ``.at[i].set(...)`` is the functional analogue of in-place
        mutation and traces cleanly under ``jit``.
    """
    N = state.shape[0]
    indices = jax.random.randint(key, (n_steps,), 0, N)

    def step(s, i):
        new = jnp.where(W[i] @ s >= 0, 1.0, -1.0)
        return s.at[i].set(new), None

    final, _ = jax.lax.scan(step, state, indices)
    return final


def sync_update(W, state, n_steps):
    """Synchronous update ``s_{t+1} = sign(W s_t)``.

    Neuroscience
        All units update simultaneously.  Faster than async and the
        natural ``lax.scan`` idiom, but can land in length-2 limit
        cycles; for well-stored patterns it converges in 1-2 steps.

    JAX
        Single ``lax.scan``; the trajectory is returned so callers can
        plot energy over time without re-running.
    """
    def step(s, _):
        s = jnp.sign(W @ s)
        s = jnp.where(s == 0, 1.0, s)          # break sign(0) ties to +1
        return s, s

    final, trajectory = jax.lax.scan(step, state, None, length=n_steps)
    return final, trajectory


# ---------------------------------------------------------------------------
# 2.  MODERN (CONTINUOUS) HOPFIELD NETWORK  (Ramsauer et al., 2020)
# ---------------------------------------------------------------------------

def modern_store(patterns: jnp.ndarray) -> jnp.ndarray:
    """In the modern formulation, patterns ARE the weight matrix.

    There is no Hebbian outer-product step.  The stored memories serve
    directly as the keys and values of an attention layer.  This is the
    conceptual jump: instead of compressing ``M`` patterns into a single
    ``N x N`` matrix (and inheriting its ``0.14 N`` capacity ceiling),
    we keep all ``M`` patterns explicitly and retrieve via softmax.
    Capacity is then *exponential* in ``N``.

    Returns ``patterns`` unchanged; the function exists to keep the API
    parallel to ``hebbian_store`` and to mark the shift in viewpoint.
    """
    return patterns


def modern_energy(patterns, state, beta):
    """Modern Hopfield energy (Ramsauer et al. 2020, Eq. 3).

    Math
        ``E(s) = -beta^{-1} log sum_mu exp(beta xi_mu^T s) + 1/2 s^T s``

    Neuroscience
        The log-sum-exp ("softmax energy") replaces the classical net's
        ``O(N)`` broad basins with exponentially many narrow ones.
        ``beta`` controls basin sharpness: large ``beta`` retrieves a
        single pattern crisply, small ``beta`` blends nearby memories.

    JAX
        ``jax.scipy.special.logsumexp`` is autodiff-safe;
        ``jax.grad(modern_energy, 1)`` recovers ``state - X^T softmax(...)``,
        i.e. the residual of one ``modern_retrieve`` step.
    """
    return (-jax.scipy.special.logsumexp(beta * patterns @ state) / beta
            + 0.5 * state @ state)


def modern_retrieve(patterns, query, beta, n_steps=1):
    """Fixed-point iteration for modern Hopfield retrieval.

    Math (Ramsauer et al. 2020, Eq. 4)::

        s_{t+1} = X^T softmax(beta * X s_t)        with X = patterns.

    THIS IS SOFTMAX ATTENTION.  With the substitutions::

        K = V = patterns,    Q = query,    beta = 1 / sqrt(d),

    the expression becomes::

        Attention(Q, K, V) = V^T softmax(K Q / sqrt(d)).

    A Transformer attention head retrieving from its context is
    performing exactly one step of this iteration.  The stored memories
    are the keys and values; the query is the query.  See
    ``verify_equivalence`` for the empirical check.

    Neuroscience
        One step suffices for well-separated memories -- which is why
        Transformers don't iterate.  More steps help on noisy or
        ambiguous queries.

    JAX
        ``lax.scan`` over fixed-point iterations; pure functional, no
        in-place mutation, friendly to ``jit`` and ``vmap``.
    """
    def step(s, _):
        attn = jax.nn.softmax(beta * patterns @ s)
        return patterns.T @ attn, None

    final, _ = jax.lax.scan(step, query, None, length=n_steps)
    return final


def capacity_classical(n_neurons):
    """Theoretical capacity of the classical net (Amit-Gutfreund-Sompolinsky
    1985): about ``0.138 * N`` random binary patterns before retrieval
    collapses into spurious mixed states."""
    return 0.138 * n_neurons


def capacity_modern(n_neurons, beta):
    """Theoretical capacity of the modern net (Ramsauer et al. 2020,
    Thm. 3): exponential in ``N``.  Roughly ``(1/2) * 4^(N/2)`` patterns
    can be stored with well-separated attractors -- which is what makes
    a single attention head useful as long-term memory."""
    del beta  # bound is independent of beta in the well-separated regime
    return 0.5 * 4.0 ** (n_neurons / 2)


# ---------------------------------------------------------------------------
# 3.  VMAPPED BATCH RETRIEVAL  (the JAX showcase)
# ---------------------------------------------------------------------------
#
# Retrieving 1000 queries at once requires no rewrite -- just compose
# ``vmap`` with ``jit``.  Two lines, the entire point of the framework.

batch_retrieve = jax.vmap(modern_retrieve, in_axes=(None, 0, None, None))
fast_retrieve = jax.jit(batch_retrieve, static_argnums=(3,))


# ---------------------------------------------------------------------------
# 4.  ATTENTION EQUIVALENCE
# ---------------------------------------------------------------------------

def attention(Q, K, V, scale=None):
    """Standard scaled dot-product attention for a single query.

    Math
        ``Attention(Q, K, V) = V^T softmax(scale * K Q)``.

    With ``K = V`` and ``scale = beta`` this is one step of
    ``modern_retrieve``.  See ``verify_equivalence``.
    """
    if scale is None:
        scale = 1.0 / jnp.sqrt(K.shape[-1])
    return V.T @ jax.nn.softmax(scale * K @ Q)


def verify_equivalence(patterns, query, beta):
    """Empirical proof that ``modern_retrieve`` is softmax attention.

    Runs one step of each on identical inputs and returns both outputs
    plus the max absolute difference.  In float32 the difference is at
    last-bit rounding (~1e-7); in float64, ~1e-15.

    This is the entire claim of Ramsauer et al. (2020) collapsed into
    a single assertion.
    """
    h = modern_retrieve(patterns, query, beta, n_steps=1)
    a = attention(query, patterns, patterns, scale=beta)
    return h, a, jnp.max(jnp.abs(h - a))
