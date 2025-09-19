# mala_pyopencl.py
import math, time, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# -------------------- Configuration --------------------
N_TOTAL          = 1_000_000      # steps per chain (set 10_000_000 for the big run)
BURNIN           = 100
N_CHAINS         = 50_000         # bump to saturate GPU (e.g., 100k–200k)
STEP_SIZE        = 0.55
STEPS_PER_LAUNCH = 1000           # inner steps per kernel launch (amortizes overhead)
SEED             = 123456789      # change for different random streams

# -------------------- OpenCL setup ---------------------
def pick_amd_device():
    # Prefer an AMD GPU; fall back to any GPU; then CPU as last resort.
    for plat in cl.get_platforms():
        for dev in plat.get_devices():
            name = dev.name.lower()
            if "advanced micro devices" in plat.name.lower() or "amd" in name or "radeon" in name:
                if dev.type & cl.device_type.GPU:
                    return cl.Context([dev])
    # Fallbacks:
    for plat in cl.get_platforms():
        gpus = [d for d in plat.get_devices() if d.type & cl.device_type.GPU]
        if gpus:
            return cl.Context([gpus[0]])
    # CPU fallback (just for sanity checks)
    return cl.create_some_context(interactive=False)

ctx = pick_amd_device()
queue = cl.CommandQueue(ctx)
dev = ctx.devices[0]
print("Using device:", dev.platform.name, "|", dev.name)

# -------------------- OpenCL kernel --------------------
# One work-item = one chain. Each call advances K steps.
# RNG: xorshift32 + Box–Muller. State/logp in float; sums in double.
KERNEL_SRC = r"""
// Enable double precision
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline uint xorshift32(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline float u01(uint *state) {
    uint x = xorshift32(*state);
    *state = x;
    // (x + 1) * 2^-32, avoid 0
    return fmax((x + 1u) * 2.3283064365386963e-10f, 1.0e-7f);
}

// Box–Muller: returns one normal, discards the second
inline float randn(uint *state) {
    float u1 = u01(state);
    float u2 = u01(state);
    float r = sqrt(-2.0f * log(u1));
    float ang = 6.283185307179586f * u2; // 2*pi
    return r * cos(ang);
}

inline float log_unnorm_t4(float x) {
    // -log(1 + x^4 / 4)
    float x2 = x * x;
    float x4 = x2 * x2;
    return -log(1.0f + 0.25f * x4);
}

inline float grad_log_unnorm_t4(float x) {
    // -x / (2*(1 + x^4/4))
    float x2 = x * x;
    float x4 = x2 * x2;
    return -x / (2.0f * (1.0f + 0.25f * x4));
}

__kernel void mala_chunk(
    __global float  *x,           // [n]
    __global float  *logp,        // [n]
    __global double *sum_abs,     // [n] (f64)
    __global double *sum_ind,     // [n] (f64)
    __global int    *burn_left,   // [n]
    __global uint   *rng_state,   // [n]
    const float step_size,
    const float ss2,
    const float inv2s,
    const int   K
) {
    int i = get_global_id(0);
    float xx = x[i];
    float lp = logp[i];
    int   b  = burn_left[i];
    uint  s  = rng_state[i];

    for (int t = 0; t < K; ++t) {
        // proposal
        float g = grad_log_unnorm_t4(xx);
        float mean_fwd = xx + 0.5f * ss2 * g;
        float z = randn(&s);
        float xprop = mean_fwd + step_size * z;

        float lp_prop = log_unnorm_t4(xprop);
        float gp = grad_log_unnorm_t4(xprop);
        float mean_bwd = xprop + 0.5f * ss2 * gp;

        float df = xprop - mean_fwd;
        float db = xx    - mean_bwd;
        float lq_fwd = -inv2s * df * df;
        float lq_bwd = -inv2s * db * db;

        float loga = lp_prop + lq_bwd - (lp + lq_fwd);

        float u = u01(&s);
        if (log(u) < loga) {
            xx = xprop;
            lp = lp_prop;
        }

        if (b > 0) {
            --b;
        } else {
            double ax = (double)fabs(xx);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }
    }

    x[i] = xx;
    logp[i] = lp;
    burn_left[i] = b;
    rng_state[i] = s;
}
"""

prg = cl.Program(ctx, KERNEL_SRC).build(options=["-cl-fast-relaxed-math"])

# -------------------- Host buffers ---------------------
n = N_CHAINS
x = cl_array.zeros(queue, n, dtype=np.float32)
logp = cl_array.empty_like(x); logp.fill(0.0)   # log p(0) = -log(1+0)=0
sum_abs = cl_array.zeros(queue, n, dtype=np.float64)
sum_ind = cl_array.zeros(queue, n, dtype=np.float64)
burn_left = cl_array.to_device(queue, np.full(n, BURNIN, dtype=np.int32))

# simple per-chain RNG seed: splitmix-like
rng_h = np.empty(n, dtype=np.uint32)
state = np.uint64(SEED)
for i in range(n):
    state = (state + 0x9E3779B97F4A7C15) & ((1<<64)-1)
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1<<64)-1)
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1<<64)-1)
    z = z ^ (z >> 31)
    rng_h[i] = np.uint32(z & 0xFFFFFFFF)
rng_state = cl_array.to_device(queue, rng_h)

ss2 = np.float32(STEP_SIZE * STEP_SIZE)
inv2s = np.float32(1.0 / (2.0 * ss2))

# -------------------- Run ------------------------------
def run():
    kept_steps = max(0, (N_TOTAL - 1) - BURNIN)  # same for all chains

    # warm-up (JIT + caches)
    evt = prg.mala_chunk(queue, (n,), None,
                         x.data, logp.data, sum_abs.data, sum_ind.data,
                         burn_left.data, rng_state.data,
                         np.float32(STEP_SIZE), ss2, inv2s, np.int32(1))
    evt.wait()

    t0 = time.time()
    steps_done = 1  # we started at t=1 (like your Julia)
    while steps_done < (N_TOTAL - 1):
        K = min(STEPS_PER_LAUNCH, (N_TOTAL - 1) - steps_done)
        evt = prg.mala_chunk(queue, (n,), None,
                             x.data, logp.data, sum_abs.data, sum_ind.data,
                             burn_left.data, rng_state.data,
                             np.float32(STEP_SIZE), ss2, inv2s, np.int32(K))
        evt.wait()
        steps_done += K

    cl.enqueue_barrier(queue)
    elapsed = time.time() - t0

    erg_abs = (sum_abs.get() / max(kept_steps, 1))
    erg_ind = (sum_ind.get() / max(kept_steps, 1))

    print(f"Elapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,} | Kept: {kept_steps:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

if __name__ == "__main__":
    run()
