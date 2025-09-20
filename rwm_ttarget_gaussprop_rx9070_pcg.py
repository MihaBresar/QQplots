# rwm_ttarget_gaussprop_rx9070_pcg.py
# RWM targeting Student-t (nu_target arbitrary).
# Proposals: symmetric Gaussian via Box–Muller.
# RNG: PCG32 with independent per-chain streams (state + increment, 64-bit).
# OpenCL on AMD RX 9070. ASCII-only; CSV outputs and chain-0 acceptance.

import os, time, csv, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# Optional: disable PyOpenCL cache if you ever hit encoding issues
# os.environ["PYOPENCL_NO_CACHE"] = "1"

# ================== TUNABLES ==================
N_TOTAL           =  5_0000_000     # steps per chain
BURNIN            =    2_0000_000
N_CHAINS          =     150_000     # raise to 100k–300k to feed the GPU

# Target: Student-t with (nu, tau)
NU_TARGET         =        1.0
TAU_TARGET        =        1.0

# Proposal: Gaussian N(0, prop_scale^2)
PROPOSAL_SCALE    =        1.0      # proposal std dev; tune for ~10–40% accept

STEPS_PER_LAUNCH  =      4000       # inner-iter does 2 steps => 8000 steps/launch
DESIRED_LOCAL     =       256       # 128/256/512; auto-clamped
SEED              = 123456789

# Lock to RX 9070 based on your listing: Platform 0, Device 1
PLATFORM_INDEX    =         0
DEVICE_INDEX      =         1

# ================== KERNEL (ASCII ONLY) ==================
KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef ulong u64;
typedef uint  u32;

inline u32 rotr32(u32 x, u32 r) { return (x >> r) | (x << ((32 - r) & 31)); }

// PCG32: per-thread stream using 64-bit state and 64-bit odd increment "inc"
inline u32 pcg32_next(u64 *state, u64 inc) {
    u64 old = *state;
    *state = old * (u64)6364136223846793005UL + (inc | 1UL);
    u32 xorshifted = (u32)(((old >> 18) ^ old) >> 27);
    u32 rot = (u32)(old >> 59);
    return rotr32(xorshifted, rot);
}

// Uniform (0,1] from PCG32 (avoid exact 0)
inline float pcg_u01(u64 *state, u64 inc) {
    u32 x = pcg32_next(state, inc);
    return fmax(((float)(x + 1u)) * 2.3283064365386963e-10f, 1.0e-7f);
}

// Standard normal via Box–Muller (one draw)
// z ~ N(0,1)
inline float sample_standard_normal(u64 *state, u64 inc){
    const float pi = 3.14159265358979323846f;
    float u1 = pcg_u01(state, inc);
    float u2 = pcg_u01(state, inc);
    float r  = native_sqrt(-2.0f * native_log(u1));
    float th = 2.0f * pi * u2;
    return r * native_cos(th); // discard the sine mate to keep code simple
}

// Student-t target (unnormalised) with (nu, tau):
// log p(x) = -0.5 * (nu + 1) * log( 1 + (x/tau)^2 / nu )
inline float log_unnorm_t_target(float x, float nu, float tau){
    float z  = x / tau;
    float q  = 1.0f + (z*z) / nu;
    return -0.5f * (nu + 1.0f) * native_log(q);
}

// Two RWM steps per inner loop to reduce launch overhead.
__kernel void rwm_gauss_chunk2_pcg(
    __global float  *x, __global double *sum_abs, __global double *sum_ind,
    __global int    *burn_left,
    __global ulong  *rng_state_hi,   // PCG state (updated)
    __global ulong  *rng_inc_lo,     // PCG stream increment (constant, odd)
    // acceptance tracking for chain 0:
    __global uint   *acc_count_out, __global uint *prop_count_out,
    const int n,
    const float prop_scale,
    const float nu_target, const float tau_target,
    const int K)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float xi = x[i];
    float lp_current = log_unnorm_t_target(xi, nu_target, tau_target);

    int  b  = burn_left[i];
    u64  st = rng_state_hi[i];
    u64  inc= rng_inc_lo[i];

    uint acc0 = 0, prop0 = 0; // for chain 0 only

    for (int it=0; it<K; ++it){
        // ---- step A ----
        float step = prop_scale * sample_standard_normal(&st, inc);
        float xprop = xi + step;
        float lp_prop = log_unnorm_t_target(xprop, nu_target, tau_target);
        float loga = lp_prop - lp_current;  // symmetric proposal
        float u = pcg_u01(&st, inc);
        if (native_log(u) < loga){ xi = xprop; lp_current = lp_prop; if (i==0) acc0++; }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double ax = (double)fabs(xi);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }

        // ---- step B ----
        step  = prop_scale * sample_standard_normal(&st, inc);
        xprop = xi + step;
        lp_prop = log_unnorm_t_target(xprop, nu_target, tau_target);
        loga = lp_prop - lp_current;
        u    = pcg_u01(&st, inc);
        if (native_log(u) < loga){ xi = xprop; lp_current = lp_prop; if (i==0) acc0++; }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double ax = (double)fabs(xi);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }
    }

    x[i] = xi; burn_left[i] = b;
    rng_state_hi[i] = st; // persist PCG state

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

# SplitMix64 for seeding 64-bit values
def splitmix64(x):
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    z = x
    z ^= (z >> 30); z = (z * 0xBF58476D1CE4E5B9) & mask
    z ^= (z >> 27); z = (z * 0x94D049BB133111EB) & mask
    z ^= (z >> 31)
    return z

def make_pcg_state(n, seed):
    out = np.empty(n, dtype=np.uint64)
    x = seed & ((1<<64)-1)
    for i in range(n):
        x = splitmix64(x)
        out[i] = x
    return out

def make_pcg_inc(n, seed):
    out = np.empty(n, dtype=np.uint64)
    x = (~seed) & ((1<<64)-1)
    for i in range(n):
        # mix index so each stream gets a distinct odd increment
        x = splitmix64((x ^ (i * 0x9E3779B97F4A7C15)) & ((1<<64)-1))
        out[i] = x | 1  # PCG requires odd increment
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
    plats = cl.get_platforms()
    dev = plats[PLATFORM_INDEX].get_devices()[DEVICE_INDEX]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    print("Using device:", dev.platform.name, "|", dev.name)

    prg = build_program(ctx)
    kernel = cl.Kernel(prg, "rwm_gauss_chunk2_pcg")

    n = int(N_CHAINS)

    # Device buffers
    x = cl_array.zeros(queue, n, dtype=np.float32)
    sum_abs = cl_array.zeros(queue, n, dtype=np.float64)
    sum_ind = cl_array.zeros(queue, n, dtype=np.float64)
    burn_left = cl_array.to_device(queue, np.full(n, BURNIN, dtype=np.int32))

    # PCG32 RNG buffers: state (hi) and increment (lo, constant odd)
    rng_state_hi = cl_array.to_device(queue, make_pcg_state(n, SEED).view(np.uint64))
    rng_inc_lo   = cl_array.to_device(queue, make_pcg_inc  (n, SEED).view(np.uint64))

    # Acceptance counters for chain 0
    acc_count = cl_array.zeros(queue, 1, dtype=np.uint32)
    prop_count= cl_array.zeros(queue, 1, dtype=np.uint32)

    # Kernel args (static)
    kernel.set_arg(0,  x.data)
    kernel.set_arg(1,  sum_abs.data)
    kernel.set_arg(2,  sum_ind.data)
    kernel.set_arg(3,  burn_left.data)
    kernel.set_arg(4,  rng_state_hi.data)
    kernel.set_arg(5,  rng_inc_lo.data)
    kernel.set_arg(6,  acc_count.data)
    kernel.set_arg(7,  prop_count.data)
    kernel.set_arg(8,  np.int32(n))
    kernel.set_arg(9,  np.float32(PROPOSAL_SCALE))    # <— alpha removed
    kernel.set_arg(10, np.float32(NU_TARGET))
    kernel.set_arg(11, np.float32(TAU_TARGET))
    # arg 12 = K, set per launch

    # Launch geometry
    local_size = choose_local_size(dev, kernel, DESIRED_LOCAL)
    gsize = round_up(n, local_size)
    global_size = (gsize,)
    local_size  = (local_size,)

    # Warmup (2 steps)
    kernel.set_arg(12, np.int32(1))
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    # Main loop: each inner-iter does 2 steps -> advance 2*K per launch
    steps_done = 0
    t0 = time.time()
    while steps_done < (N_TOTAL - 1):
        K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
        kernel.set_arg(12, np.int32(K))
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

    print(f\"\\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,}\")
    print(\"Last 5 ergodic |x| averages:\", erg_abs[-5:])
    print(\"Last 5 ergodic indicators :\", erg_ind[-5:])
    print(f\"Chain 0 acceptance: {acc} / {props} = {acc_rate:.4f}\")

    # Save CSVs next to this script
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(scriptdir, \"ergodic_average_abs_RWM_ttarget_gauss_pcg.csv\"), \"w\", newline=\"\") as f:
        w = csv.writer(f); w.writerows([[v] for v in erg_abs])

    with open(os.path.join(scriptdir, \"ergodic_average_indicator_RWM_ttarget_gauss_pcg.csv\"), \"w\", newline=\"\") as f:
        w = csv.writer(f); w.writerows([[v] for v in erg_ind])

    with open(os.path.join(scriptdir, \"acceptance_chain0_RWM_ttarget_gauss_pcg.txt\"), \"w\") as f:
        f.write(f\"accepts={acc}, proposals={props}, accept_rate={acc_rate:.6f}\\n\")

    print(\"\\nCSV files written:\")
    print(\"  ergodic_average_abs_RWM_ttarget_gauss_pcg.csv\")
    print(\"  ergodic_average_indicator_RWM_ttarget_gauss_pcg.csv\")
    print(\"  acceptance_chain0_RWM_ttarget_gauss_pcg.txt\")

if __name__ == \"__main__\":
    run()
