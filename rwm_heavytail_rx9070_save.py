# rwm_heavytail_rx9070_save.py
# Random-Walk Metropolis with symmetric double-Pareto (mirrored Lomax) proposals
# on AMD RX 9070 (gfx1201) via OpenCL. Saves CSVs and reports acceptance
# probability for chain 0.

import time, os, csv, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ================== TUNABLES ==================
N_TOTAL           =  2_000_000    # steps per chain (raise to 30_000_000 for big runs)
BURNIN            =    500_000
N_CHAINS          =     50_000    # increase (100k–300k) to feed the GPU
PROPOSAL_SCALE    =        5.0     # proposal scale 's' (like your Julia code)
ALPHA_TAIL        =        0.10    # tail exponent α; set 0.05 to match t(0.05) tails

STEPS_PER_LAUNCH  =      4000     # inner-iter does 2 steps => 8000 steps per launch
DESIRED_LOCAL     =       256     # 128/256/512; auto-clamped safely
SEED              = 123456789

# Lock to your RX 9070 based on your listing: Platform 0, Device 1
PLATFORM_INDEX    =         0
DEVICE_INDEX      =         1

# ================== KERNEL ==================
# - Symmetric double-Pareto (mirrored Lomax) proposal:
#     U1,U2 ~ Uniform(0,1], sign = ±1 (from U1), magnitude = s*(U2^{-1/α}-1)
#   => proposal is symmetric => Hastings ratio = target ratio.
# - Two RWM steps per inner loop (reduces launch overhead).
KERNEL = r"""
// We keep fp64 for accumulators (safer for very long runs).
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline uint xorshift32(uint x){ x^=x<<13; x^=x>>17; x^=x<<5; return x; }

inline float u01(uint *s){
    uint x = xorshift32(*s); *s = x;
    // (x+1)*2^-32 in (≈0,1], avoid 0 precisely
    return fmax((x + 1u) * 2.3283064365386963e-10f, 1.0e-7f);
}

// Heavy-tailed symmetric step ~ double-Pareto with tail exponent (1+alpha)
// magnitude = scale*(U^{-1/alpha}-1), sign = ±1
inline float heavy_step(uint *state, float scale, float alpha){
    float u1 = u01(state);
    float sign = (u1 < 0.5f) ? -1.0f : 1.0f;
    float u  = u01(state);                   // (0,1]
    // u^{-1/alpha} = exp( (-1/alpha) * log(u) )
    float mag = scale * (native_exp((-1.0f/alpha) * native_log(u)) - 1.0f);
    return sign * mag;
}

// Target (unnormalised): 1 / (1 + |x|^{4.5} / 9.0)
inline float unnorm_target(float x){
    float ax = fabs(x);
    // pow(ax,4.5) = ax^4 * sqrt(ax)
    float x2 = ax*ax;
    float x4 = x2*x2;
    float x45 = x4 * native_sqrt(ax);
    return 1.0f / (1.0f + (x45 * (1.0f/9.0f)));
}

__kernel void rwm_heavy_chunk2(
    __global float  *x, __global double *sum_abs, __global double *sum_ind,
    __global int    *burn_left, __global uint   *rng_state,
    // acceptance tracking for chain 0:
    __global uint   *acc_count_out, __global uint *prop_count_out,
    const int n, const float scale, const float alpha, const int K)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float xi = x[i];
    float p_current = unnorm_target(xi);

    int   b  = burn_left[i];
    uint  s  = rng_state[i];

    // Local acceptance counters for chain 0
    uint acc0 = 0, prop0 = 0;

    for (int it=0; it<K; ++it){
        // ---- step A ----
        float step = heavy_step(&s, scale, alpha);
        float xprop = xi + step;
        float pprop = unnorm_target(xprop);
        float a = pprop / p_current;                 // symmetric proposal
        float u = u01(&s);
        // accept?
        if (u < fmin(a, 1.0f)){ xi = xprop; p_current = pprop; if (i==0) acc0++; }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double ax = (double)fabs(xi);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }

        // ---- step B ----
        step  = heavy_step(&s, scale, alpha);
        xprop = xi + step;
        pprop = unnorm_target(xprop);
        a     = pprop / p_current;
        u     = u01(&s);
        if (u < fmin(a, 1.0f)){ xi = xprop; p_current = pprop; if (i==0) acc0++; }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double ax = (double)fabs(xi);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }
    }

    x[i] = xi; burn_left[i] = b; rng_state[i] = s;

    // Only chain 0 writes acceptance counters (no races)
    if (i == 0){
        acc_count_out[0]  += acc0;
        prop_count_out[0] += prop0;
    }
}
"""

# ================== UTIL ==================
def build_program(ctx):
    return cl.Program(ctx, KERNEL).build(options=[
        "-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros"
    ])

def splitmix64_seed_array(n, seed):
    """SplitMix64 in pure Python ints (no NumPy overflow warnings)."""
    out = np.empty(n, dtype=np.uint32)
    mask = (1 << 64) - 1
    x = int(seed) & mask
    for i in range(n):
        x = (x + 0x9E3779B97F4A7C15) & mask
        z = x
        z ^= (z >> 30); z = (z * 0xBF58476D1CE4E5B9) & mask
        z ^= (z >> 27); z = (z * 0x94D049BB133111EB) & mask
        z ^= (z >> 31)
        out[i] = np.uint32(z & 0xFFFFFFFF)
    return out

def choose_local_size(dev, kernel, desired):
    dev_max = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    ker_max = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
    pref    = kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev)
    lsz = int(min(desired, dev_max, ker_max))
    if pref and lsz >= pref:
        lsz = (lsz // pref) * pref
    lsz = max(1, lsz)
    return lsz

def round_up(x, m):
    return ((x + m - 1) // m) * m

# ================== RUN ==================
def run():
    # Select RX 9070 (P0:D1)
    plats = cl.get_platforms()
    dev = plats[PLATFORM_INDEX].get_devices()[DEVICE_INDEX]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    print("Using device:", dev.platform.name, "|", dev.name)

    prg = build_program(ctx)
    kernel = cl.Kernel(prg, "rwm_heavy_chunk2")

    n = int(N_CHAINS)

    # Device buffers
    x = cl_array.zeros(queue, n, dtype=np.float32)       # start at 0
    sum_abs = cl_array.zeros(queue, n, dtype=np.float64) # ergodic sums
    sum_ind = cl_array.zeros(queue, n, dtype=np.float64)
    burn_left = cl_array.to_device(queue, np.full(n, BURNIN, dtype=np.int32))
    rng_state = cl_array.to_device(queue, splitmix64_seed_array(n, SEED))

    # Acceptance counters for chain 0
    acc_count = cl_array.zeros(queue, 1, dtype=np.uint32)
    prop_count= cl_array.zeros(queue, 1, dtype=np.uint32)

    # Kernel args (static)
    kernel.set_arg(0,  x.data)
    kernel.set_arg(1,  sum_abs.data)
    kernel.set_arg(2,  sum_ind.data)
    kernel.set_arg(3,  burn_left.data)
    kernel.set_arg(4,  rng_state.data)
    kernel.set_arg(5,  acc_count.data)
    kernel.set_arg(6,  prop_count.data)
    kernel.set_arg(7,  np.int32(n))
    kernel.set_arg(8,  np.float32(PROPOSAL_SCALE))
    kernel.set_arg(9,  np.float32(ALPHA_TAIL))
    # arg 10 = K (set per launch)

    # Launch geometry
    local_size = choose_local_size(dev, kernel, DESIRED_LOCAL)
    gsize = round_up(n, local_size)
    global_size = (gsize,)
    local_size  = (local_size,)

    # Warmup (K=1 -> 2 steps)
    kernel.set_arg(10, np.int32(1))
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    # Main loop (each inner-iter does 2 steps -> advance 2*K per launch)
    steps_done = 0
    t0 = time.time()
    while steps_done < (N_TOTAL - 1):
        K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
        kernel.set_arg(10, np.int32(K))
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        steps_done += 2*K

    elapsed = time.time() - t0

    # Collect results
    kept = max(0, (N_TOTAL - 1) - BURNIN)
    erg_abs = (sum_abs.get()/max(kept,1))
    erg_ind = (sum_ind.get()/max(kept,1))

    acc = int(acc_count.get()[0])
    props = int(prop_count.get()[0])
    acc_rate = (acc / props) if props > 0 else float('nan')

    print(f"\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])
    print(f"Chain 0 acceptance: {acc} / {props} = {acc_rate:.4f}")

    # ========== CSV OUTPUT ==========
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(scriptdir, "ergodic_average_abs_RWM_heavy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for val in erg_abs:
            w.writerow([val])

    with open(os.path.join(scriptdir, "ergodic_average_indicator_RWM_heavy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for val in erg_ind:
            w.writerow([val])

    with open(os.path.join(scriptdir, "acceptance_chain0_RWM_heavy.txt"), "w") as f:
        f.write(f"accepts={acc}, proposals={props}, accept_rate={acc_rate:.6f}\n")

    print("\nCSV files written:")
    print("  ergodic_average_abs_RWM_heavy.csv")
    print("  ergodic_average_indicator_RWM_heavy.csv")
    print("  acceptance_chain0_RWM_heavy.txt")

if __name__ == "__main__":
    run()
