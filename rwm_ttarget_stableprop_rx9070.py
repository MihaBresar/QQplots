# rwm_ttarget_stableprop_rx9070.py
# Random-Walk Metropolis targeting Student-t(ν_target=5) (finite 3 moments),
# proposals from symmetric alpha-stable (alpha=0.1) via CMS method.
# OpenCL on AMD RX 9070 (gfx1201). Writes CSVs and acceptance for chain 0.

import os, time, csv, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ================== TUNABLES ==================
N_TOTAL           =  2_000_000     # steps per chain (raise as needed)
BURNIN            =    500_000
N_CHAINS          =     50_000     # increase (100k–300k) to boost GPU utilization

# Target: Student-t with nu_target > 3 (we use 5) and scale tau
NU_TARGET         =        5.0
TAU_TARGET        =        1.0      # scale of the target (like standard t when =1)

# Proposal: symmetric alpha-stable (very heavy-tailed) via CMS
PROPOSAL_ALPHA    =        0.10     # alpha in (0,2]; 0.1 is ultra heavy-tailed
PROPOSAL_SCALE    =        1.0      # overall proposal step scale (tune this)

STEPS_PER_LAUNCH  =      4000       # inner-iter does 2 steps => 8000 steps per launch
DESIRED_LOCAL     =       256       # 128/256/512; auto-clamped to valid
SEED              = 123456789

# Lock to RX 9070 based on your listing: Platform 0, Device 1
PLATFORM_INDEX    =         0
DEVICE_INDEX      =         1

# Optional: disable PyOpenCL kernel caching if your system still warns
# os.environ["PYOPENCL_NO_CACHE"] = "1"

# ================== KERNEL (ASCII ONLY) ==================
KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline uint xorshift32(uint x){ x^=x<<13; x^=x>>17; x^=x<<5; return x; }

inline float u01(uint *s){
    uint x = xorshift32(*s); *s = x;
    // (x+1)*2^-32 in (0,1], avoid exact 0
    return fmax((x + 1u) * 2.3283064365386963e-10f, 1.0e-7f);
}

// Symmetric alpha-stable sample (beta=0) via Chambers-Mallows-Stuck.
// U ~ Uniform(-pi/2, pi/2), W ~ Exp(1).
// For beta=0, formula simplifies to:
// X = [ sin(alpha*U) / (cos U)^(1/alpha) ] * [ cos( (1-alpha)*U ) / W ]^((1-alpha)/alpha)
// To avoid overflow/NaN, clamp cos(...) and W away from 0, and compute via logs.
inline float sample_symmetric_stable(uint *state, float alpha){
    // Map u in (0,1] to U in (-pi/2, pi/2)
    float u = u01(state);
    // avoid endpoints that make cos(U)=0
    const float pi = 3.14159265358979323846f;
    float U = pi*(u - 0.5f);                 // (-pi/2, pi/2)

    // Exponential(1): W = -log(u2), clamp away from 0
    float u2 = u01(state);
    float W  = -native_log(u2);
    W = fmax(W, 1e-12f);

    float cU   = native_cos(U);
    float sA   = native_sin(alpha*U);
    float cA   = native_cos((1.0f - alpha)*U);

    // clamp cosines to avoid division by 0
    cU = fmax(cU, 1e-12f);
    cA = fmax(cA, 1e-12f);

    float inv_alpha = 1.0f / alpha;
    float one_minus_over_alpha = (1.0f - alpha) * inv_alpha;

    // term1 = sin(alpha*U) / (cos U)^(1/alpha)
    float term1 = sA * native_exp( -inv_alpha * native_log(cU) );

    // term2 = [ cA / W ]^((1-alpha)/alpha) = exp( ((1-alpha)/alpha) * (log(cA) - log(W)) )
    float term2 = native_exp( one_minus_over_alpha * ( native_log(cA) - native_log(W) ) );

    return term1 * term2;
}

// Student-t target (unnormalised) with nu_target and scale tau:
// p(x) ∝ [ 1 + (x/tau)^2 / nu ]^{-(nu+1)/2}
// We use log form for stability.
inline float log_unnorm_t_target(float x, float nu, float tau){
    float z  = x / tau;
    float q  = 1.0f + (z*z) / nu;
    return -0.5f * (nu + 1.0f) * native_log(q);
}

// Two RWM steps per inner loop (reduces launch overhead).
__kernel void rwm_stable_chunk2(
    __global float  *x, __global double *sum_abs, __global double *sum_ind,
    __global int    *burn_left, __global uint   *rng_state,
    // acceptance tracking for chain 0:
    __global uint   *acc_count_out, __global uint *prop_count_out,
    const int n,
    const float alpha, const float prop_scale,
    const float nu_target, const float tau_target,
    const int K)
{
    int i = get_global_id(0);
    if (i >= n) return;

    // Initialize state
    float xi = x[i];
    float lp_current = log_unnorm_t_target(xi, nu_target, tau_target);

    int   b  = burn_left[i];
    uint  s  = rng_state[i];

    // Local acceptance counters for chain 0 only
    uint acc0 = 0, prop0 = 0;

    for (int it=0; it<K; ++it){
        // ---- step A ----
        float step = prop_scale * sample_symmetric_stable(&s, alpha);
        float xprop = xi + step;
        float lp_prop = log_unnorm_t_target(xprop, nu_target, tau_target);

        float loga = lp_prop - lp_current;   // symmetric proposal
        float u = u01(&s);
        if (native_log(u) < loga){ xi = xprop; lp_current = lp_prop; if (i==0) acc0++; }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double ax = (double)fabs(xi);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }

        // ---- step B ----
        step  = prop_scale * sample_symmetric_stable(&s, alpha);
        xprop = xi + step;
        lp_prop = log_unnorm_t_target(xprop, nu_target, tau_target);

        loga = lp_prop - lp_current;
        u    = u01(&s);
        if (native_log(u) < loga){ xi = xprop; lp_current = lp_prop; if (i==0) acc0++; }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double ax = (double)fabs(xi);
            sum_abs[i] += ax;
            sum_ind[i] += (ax >= 2.0);
        }
    }

    x[i] = xi; burn_left[i] = b; rng_state[i] = s;

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
    # RX 9070 (P0:D1)
    plats = cl.get_platforms()
    dev = plats[PLATFORM_INDEX].get_devices()[DEVICE_INDEX]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    print("Using device:", dev.platform.name, "|", dev.name)

    prg = build_program(ctx)
    kernel = cl.Kernel(prg, "rwm_stable_chunk2")

    n = int(N_CHAINS)
    # Device buffers
    x = cl_array.zeros(queue, n, dtype=np.float32)
    sum_abs = cl_array.zeros(queue, n, dtype=np.float64)
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
    kernel.set_arg(8,  np.float32(PROPOSAL_ALPHA))
    kernel.set_arg(9,  np.float32(PROPOSAL_SCALE))
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

    print(f"\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])
    print(f"Chain 0 acceptance: {acc} / {props} = {acc_rate:.4f}")

    # Save CSVs next to this script
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(scriptdir, "ergodic_average_abs_RWM_ttarget_stable.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows([[v] for v in erg_abs])

    with open(os.path.join(scriptdir, "ergodic_average_indicator_RWM_ttarget_stable.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows([[v] for v in erg_ind])

    with open(os.path.join(scriptdir, "acceptance_chain0_RWM_ttarget_stable.txt"), "w") as f:
        f.write(f"accepts={acc}, proposals={props}, accept_rate={acc_rate:.6f}\n")

    print("\nCSV files written:")
    print("  ergodic_average_abs_RWM_ttarget_stable.csv")
    print("  ergodic_average_indicator_RWM_ttarget_stable.csv")
    print("  acceptance_chain0_RWM_ttarget_stable.txt")

if __name__ == "__main__":
    run()
