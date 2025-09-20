# rwm_ttarget_stableprop_rx9070_pcg_20d_fix.py
# 20-D RWM targeting independent Student-t per dimension.
# Proposal: i.i.d. symmetric alpha-stable (alpha=0.1) via CMS (per-dim),
# with dimension-aware scaling: eff_scale = PROPOSAL_SCALE * D^(-1/alpha).
# RNG: PCG32 per chain (64-bit state + 64-bit odd increment).
# Correct MH: cache proposed vector once and reuse on accept (no re-sampling).
# Writes CSVs; tracks chain-0 acceptance.

import os, time, csv, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ================== TUNABLES ==================
DIM               =        20      # dimensionality
N_TOTAL           =  2_000_000     # steps per chain
BURNIN            =    500_000
N_CHAINS          =     50_000     # raise to 100k–300k as needed

# Target: Student-t per dimension
NU_TARGET         =        5.0
TAU_TARGET        =        1.0

# Proposal: symmetric alpha-stable via CMS (i.i.d. across dims)
PROPOSAL_ALPHA    =        0.10     # tune 0.1 .. 0.5; heavier -> lower acceptance
PROPOSAL_SCALE    =        1.0      # base scale; effective scale uses D^(-1/alpha)

STEPS_PER_LAUNCH  =      3000       # inner-iter does 2 steps => 6000 steps/launch
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

// ===== PCG32 =====
inline u32 rotr32(u32 x, u32 r) { return (x >> r) | (x << ((32 - r) & 31)); }
inline u32 pcg32_next(u64 *state, u64 inc) {
    u64 old = *state;
    *state = old * (u64)6364136223846793005UL + (inc | 1UL);
    u32 xorshifted = (u32)(((old >> 18) ^ old) >> 27);
    u32 rot = (u32)(old >> 59);
    return rotr32(xorshifted, rot);
}
inline float pcg_u01(u64 *state, u64 inc) {
    u32 x = pcg32_next(state, inc);
    return fmax(((float)(x + 1u)) * 2.3283064365386963e-10f, 1.0e-7f);
}

// Symmetric alpha-stable sample (beta=0) via CMS; numerically stable with clamping.
inline float sample_symmetric_stable(u64 *state, u64 inc, float alpha){
    const float pi = 3.14159265358979323846f;
    float u  = pcg_u01(state, inc);
    float U  = pi*(u - 0.5f);              // (-pi/2, pi/2)
    float u2 = pcg_u01(state, inc);
    float W  = -native_log(u2);
    W = fmax(W, 1e-12f);
    float cU = native_cos(U);
    float sA = native_sin(alpha*U);
    float cA = native_cos((1.0f - alpha)*U);
    cU = fmax(cU, 1e-12f);
    cA = fmax(cA, 1e-12f);
    float inv_alpha = 1.0f / alpha;
    float one_minus_over_alpha = (1.0f - alpha) * inv_alpha;
    float term1 = sA * native_exp( -inv_alpha * native_log(cU) );
    float term2 = native_exp( one_minus_over_alpha * ( native_log(cA) - native_log(W) ) );
    return term1 * term2;
}

// Student-t log unnormalised density per dimension
inline float log_unnorm_t_target(float x, float nu, float tau){
    float z  = x / tau;
    float q  = 1.0f + (z*z) / nu;
    return -0.5f * (nu + 1.0f) * native_log(q);
}

#define MAX_D 64  // enough room for DIM=20

// Two RWM steps per inner loop, cache proposed vector once, reuse on accept.
__kernel void rwm_stable_chunk2_pcg_20d_fix(
    __global float  *x,                     // length n*D (Structure-of-Arrays per chain)
    __global double *sum_abs,               // per-chain accumulator of mean |x|
    __global double *sum_ind,               // per-chain accumulator of indicator any(|x|>=thr)
    __global int    *burn_left,             // per-chain burnin counter
    __global ulong  *rng_state_hi,          // per-chain PCG state
    __global ulong  *rng_inc_lo,            // per-chain PCG increment (odd)
    __global uint   *acc_count_out,         // chain 0 acceptance count
    __global uint   *prop_count_out,        // chain 0 proposal count
    const int n,
    const int D,
    const float alpha, const float prop_scale_eff,   // effective scale already D^{-1/alpha}
    const float nu_target, const float tau_target,
    const float thr,                        // threshold for indicator (e.g., 2.0)
    const int K)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (D > MAX_D) return; // safety

    int  base = i * D;
    int  b    = burn_left[i];
    u64  st   = rng_state_hi[i];
    u64  inc  = rng_inc_lo[i];

    // current log target
    float lp_current = 0.0f;
    for (int d=0; d<D; ++d){
        lp_current += log_unnorm_t_target(x[base + d], nu_target, tau_target);
    }

    uint acc0 = 0, prop0 = 0; // chain 0 counters
    __private float prop_buf[MAX_D]; // proposed vector cache

    for (int it=0; it<K; ++it){
        // ----- step A: propose once, cache, compute lp_prop -----
        float lp_prop = 0.0f;
        for (int d=0; d<D; ++d){
            float step = prop_scale_eff * sample_symmetric_stable(&st, inc, alpha);
            float xprop = x[base + d] + step;
            prop_buf[d] = xprop;
            lp_prop += log_unnorm_t_target(xprop, nu_target, tau_target);
        }
        // MH accept
        float u = pcg_u01(&st, inc);
        if (native_log(u) < (lp_prop - lp_current)){
            // accept: write cached proposal and update lp_current
            lp_current = 0.0f;
            for (int d=0; d<D; ++d){
                float xnew = prop_buf[d];
                x[base + d] = xnew;
                lp_current += log_unnorm_t_target(xnew, nu_target, tau_target);
            }
            if (i==0) acc0++;
        }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            // accumulate diagnostics from current state
            double sum_abs_d = 0.0;
            int any_large = 0;
            for (int d=0; d<D; ++d){
                float ax = fabs(x[base + d]);
                sum_abs_d += (double)ax;
                any_large |= (ax >= thr) ? 1 : 0;
            }
            sum_abs[i] += sum_abs_d / (double)D;
            sum_ind[i] += (double)any_large;
        }

        // ----- step B: same pattern -----
        lp_prop = 0.0f;
        for (int d=0; d<D; ++d){
            float step = prop_scale_eff * sample_symmetric_stable(&st, inc, alpha);
            float xprop = x[base + d] + step;
            prop_buf[d] = xprop;
            lp_prop += log_unnorm_t_target(xprop, nu_target, tau_target);
        }
        u = pcg_u01(&st, inc);
        if (native_log(u) < (lp_prop - lp_current)){
            lp_current = 0.0f;
            for (int d=0; d<D; ++d){
                float xnew = prop_buf[d];
                x[base + d] = xnew;
                lp_current += log_unnorm_t_target(xnew, nu_target, tau_target);
            }
            if (i==0) acc0++;
        }
        if (i==0) prop0++;
        if (b > 0) --b; else {
            double sum_abs_d = 0.0;
            int any_large = 0;
            for (int d=0; d<D; ++d){
                float ax = fabs(x[base + d]);
                sum_abs_d += (double)ax;
                any_large |= (ax >= thr) ? 1 : 0;
            }
            sum_abs[i] += sum_abs_d / (double)D;
            sum_ind[i] += (double)any_large;
        }
    }

    burn_left[i]   = b;
    rng_state_hi[i]= st;
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

# SplitMix64 helpers to seed PCG streams
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
        x = splitmix64((x ^ (i * 0x9E3779B97F4A7C15)) & ((1<<64)-1))
        out[i] = x | 1  # odd
    return out

def choose_local_size(dev, kernel, desired):
    dev_max = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    ker_max = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
    pref    = kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev)
    lsz = int(min(desired, dev_max, ker_max))
    if pref and lsz >= pref:
        lsz = (lsz // pref) * pref
    return max(1, lsz)

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
    kernel = cl.Kernel(prg, "rwm_stable_chunk2_pcg_20d_fix")

    n = int(N_CHAINS)
    D = int(DIM)

    # Effective per-dimension scale: D^{-1/alpha}
    eff_scale = PROPOSAL_SCALE * (D ** (-1.0 / PROPOSAL_ALPHA))

    # Device buffers
    x = cl_array.zeros(queue, n*D, dtype=np.float32)            # start at 0
    sum_abs = cl_array.zeros(queue, n, dtype=np.float64)        # mean |x| over dims
    sum_ind = cl_array.zeros(queue, n, dtype=np.float64)        # indicator any(|x|>=thr)
    burn_left = cl_array.to_device(queue, np.full(n, BURNIN, dtype=np.int32))

    # PCG32 RNG: per-chain state and increments
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
    kernel.set_arg(9,  np.int32(D))
    kernel.set_arg(10, np.float32(PROPOSAL_ALPHA))
    kernel.set_arg(11, np.float32(eff_scale))
    kernel.set_arg(12, np.float32(NU_TARGET))
    kernel.set_arg(13, np.float32(TAU_TARGET))
    kernel.set_arg(14, np.float32(2.0))       # threshold for indicator |x|>=2
    # arg 15 = K set per launch

    # Launch geometry
    local_size = choose_local_size(dev, kernel, DESIRED_LOCAL)
    gsize = round_up(n, local_size)
    global_size = (gsize,)
    local_size  = (local_size,)

    # Warmup (2 steps)
    kernel.set_arg(15, np.int32(1))
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    # Main loop: each inner-iter does 2 steps -> advance 2*K per launch
    steps_done = 0
    t0 = time.time()
    while steps_done < (N_TOTAL - 1):
        K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
        kernel.set_arg(15, np.int32(K))
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        steps_done += 2*K

    elapsed = time.time() - t0

    kept = max(0, (N_TOTAL - 1) - BURNIN)
    erg_abs = (sum_abs.get()/max(kept,1))
    erg_ind = (sum_ind.get()/max(kept,1))

    acc = int(acc_count.get()[0])
    props = int(prop_count.get()[0])
    acc_rate = (acc / props) if props > 0 else float('nan')

    print(f"\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,} | Dim: {D}")
    print("Last 5 ergodic mean |x| over dims:", erg_abs[-5:])
    print("Last 5 ergodic indicators any(|x|>=2):", erg_ind[-5:])
    print(f"Chain 0 acceptance: {acc} / {props} = {acc_rate:.4f}")

    # Save CSVs next to this script
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(scriptdir, "ergodic_average_abs_RWM_20d_pcg_fix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows([[v] for v in erg_abs])

    with open(os.path.join(scriptdir, "ergodic_average_indicator_RWM_20d_pcg_fix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows([[v] for v in erg_ind])

    with open(os.path.join(scriptdir, "acceptance_chain0_RWM_20d_pcg_fix.txt"), "w") as f:
        f.write(f"accepts={acc}, proposals={props}, accept_rate={acc_rate:.6f}\n")

    print("\nCSV files written:")
    print("  ergodic_average_abs_RWM_20d_pcg_fix.csv")
    print("  ergodic_average_indicator_RWM_20d_pcg_fix.csv")
    print("  acceptance_chain0_RWM_20d_pcg_fix.txt")

if __name__ == "__main__":
    run()
