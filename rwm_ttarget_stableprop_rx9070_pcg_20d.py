# rwm_ttarget_dpareto_pcg_20d_gpu_f32.py
# 20-D RWM targeting independent Student-t per dimension.
# Proposal: symmetric double-Pareto (mirrored Lomax), i.i.d. across dims.
# RNG: PCG32 per chain (64-bit state + 64-bit odd increment).
# Correct MH: propose once, cache, reuse on accept.
# Accumulators: float32 (no FP64 in hot loop) for better RX utilization.
# Indicator: any(|x_j| > 5) per step.
# CSV outputs and acceptance tracking for chain 0. ASCII-only kernel.

import os
import time
import csv
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ================== TUNABLES ==================
DIM               =        20      # dimensionality (keep <= 64 with current kernel buffer)
N_TOTAL           =  20_000_000     # steps per chain (>= 2)
BURNIN            =    9500_000
N_CHAINS          =    120_000     # raise to 300k if VRAM allows (for higher utilization)

# Target: Student-t per dimension
NU_TARGET         =        1.0     # try 1.5, 2.0, 5.0, etc.
TAU_TARGET        =        1.0

# Proposal: symmetric double-Pareto (mirrored Lomax), i.i.d. across dims
PROPOSAL_ALPHA    =        0.10    # heavier tails -> lower acceptance
PROPOSAL_SCALE    =        1.0     # base scale; effective scale uses D^(-1/alpha)

STEPS_PER_LAUNCH  =      8000      # inner-iter K does 2*K steps => 16k steps/launch
DESIRED_LOCAL     =       256      # 128/256/512; auto-clamped
SEED              = 123456789

# ================== Robust AMD GPU picker ==================
def pick_best_gpu(prefer_name_contains=("gfx1201", "Radeon RX", "RX 9", "7900")):
    plats = cl.get_platforms()
    best = None
    best_score = -1
    for p in plats:
        try:
            devs = p.get_devices(device_type=cl.device_type.GPU)
        except cl.LogicError:
            continue
        for d in devs:
            name = d.get_info(cl.device_info.NAME)
            vendor = d.get_info(cl.device_info.VENDOR)
            cu = d.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            mem = d.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            score = cu + (mem / float(8 << 30))  # weight mem in ~8 GiB units
            if "Advanced Micro Devices" in vendor:
                score += 100
            if any(tag.lower() in name.lower() for tag in prefer_name_contains):
                score += 50
            if score > best_score:
                best = d
                best_score = score
    if best is None:
        # fallback: any GPU
        for p in plats:
            try:
                devs = p.get_devices(device_type=cl.device_type.GPU)
                if devs:
                    return devs[0]
            except cl.LogicError:
                pass
        raise RuntimeError("No OpenCL GPU device found.")
    return best

# ================== OpenCL KERNEL (ASCII ONLY) ==================
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
    // (0,1]; avoid exact 0 to keep logs safe
    return fmax(((float)(x + 1u)) * 2.3283064365386963e-10f, 1.0e-7f);
}

// ===== Double-Pareto (mirrored Lomax) proposal =====
// magnitude = scale * (U^(-1/alpha) - 1), sign = +/- with prob 1/2
// Numerically safe: cap y = -(1/alpha)*log(u) to <= 80 to avoid overflow.
inline float heavy_step_dp(u64 *state, u64 inc, float alpha, float scale){
    float u1  = pcg_u01(state, inc);
    float sgn = (u1 < 0.5f) ? -1.0f : 1.0f;

    float u   = pcg_u01(state, inc);      // (0,1]
    float y   = (-native_log(u)) / alpha; // may be large for small u
    if (y > 80.0f) y = 80.0f;

    float mag = scale * (native_exp(y) - 1.0f);
    return sgn * mag;
}

// Student-t log unnormalised density per dimension
inline float log_unnorm_t_target(float x, float nu, float tau){
    float z  = x / tau;
    float q  = 1.0f + (z*z) / nu;
    return -0.5f * (nu + 1.0f) * native_log(q);
}

#define MAX_D 64  // enough room for DIM up to 64

// Two RWM steps per inner loop; cache proposed vector once; reuse on accept.
__kernel void rwm_dpareto_chunk2_pcg_20d(
    __global float  *x,                     // length n*D (structure-of-arrays per chain)
    __global float  *sum_abs,               // per-chain accumulator of mean |x|
    __global float  *sum_ind,               // per-chain accumulator of indicator any(|x|>thr)
    __global int    *burn_left,             // per-chain burnin counter
    __global ulong  *rng_state_hi,          // per-chain PCG state (updated)
    __global ulong  *rng_inc_lo,            // per-chain PCG increment (constant, odd)
    __global uint   *acc_count_out,         // chain 0 acceptance count
    __global uint   *prop_count_out,        // chain 0 proposal count
    const int n,
    const int D,
    const float alpha, const float prop_scale_eff,   // effective scale already D^{-1/alpha}
    const float nu_target, const float tau_target,
    const float thr,                        // threshold for indicator (e.g., 5.0)
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
            float step = heavy_step_dp(&st, inc, alpha, prop_scale_eff);
            float xprop = x[base + d] + step;
            prop_buf[d] = xprop;
            lp_prop += log_unnorm_t_target(xprop, nu_target, tau_target);
        }
        // MH accept
        float u = pcg_u01(&st, inc);
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
            float sum_abs_d = 0.0f;
            int any_large = 0;
            for (int d=0; d<D; ++d){
                float ax = fabs(x[base + d]);
                sum_abs_d += ax;
                any_large |= (ax > thr) ? 1 : 0;
            }
            sum_abs[i] += sum_abs_d / (float)D;
            sum_ind[i] += (float)any_large;
        }

        // ----- step B -----
        lp_prop = 0.0f;
        for (int d=0; d<D; ++d){
            float step = heavy_step_dp(&st, inc, alpha, prop_scale_eff);
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
            float sum_abs_d = 0.0f;
            int any_large = 0;
            for (int d=0; d<D; ++d){
                float ax = fabs(x[base + d]);
                sum_abs_d += ax;
                any_large |= (ax > thr) ? 1 : 0;
            }
            sum_abs[i] += sum_abs_d / (float)D;
            sum_ind[i] += (float)any_large;
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

# ================== Build helpers ==================
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
    # Pick the best AMD GPU (RX 9070 should be selected)
    dev = pick_best_gpu()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    name = dev.get_info(cl.device_info.NAME)
    vendor = dev.get_info(cl.device_info.VENDOR)
    cu = dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
    gmem = dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    print("Using device:", vendor, "|", name, "| CUs:", cu, "| Mem GiB:", round(gmem/2**30,1))

    prg = build_program(ctx)
    kernel = cl.Kernel(prg, "rwm_dpareto_chunk2_pcg_20d")

    n = int(N_CHAINS)
    D = int(DIM)

    # Effective per-dimension scale: D^{-1/alpha} to keep typical jump roughly dim-invariant
    eff_scale = PROPOSAL_SCALE * (D ** (-1.0 / PROPOSAL_ALPHA))

    # Device buffers
    x = cl_array.zeros(queue, n*D, dtype=np.float32)            # start at 0
    sum_abs = cl_array.zeros(queue, n, dtype=np.float32)        # mean |x| over dims
    sum_ind = cl_array.zeros(queue, n, dtype=np.float32)        # indicator any(|x|>thr)
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
    kernel.set_arg(14, np.float32(5.0))     # indicator threshold |x| > 5
    # arg 15 = K set per launch below

    # Launch geometry
    local_size = choose_local_size(dev, kernel, DESIRED_LOCAL)
    gsize = round_up(n, local_size)
    global_size = (gsize,)
    local_size  = (local_size,)
    print("Global size:", global_size, " Local size:", local_size,
          "| Chains:", n, " Dim:", D, " Steps/launch (effective):", 2*STEPS_PER_LAUNCH)

    # Warmup (2 steps total)
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

    # Collect results
    kept = max(0, (N_TOTAL - 1) - BURNIN)
    erg_abs = (sum_abs.get()/max(kept,1))
    erg_ind = (sum_ind.get()/max(kept,1))

    acc = int(acc_count.get()[0])
    props = int(prop_count.get()[0])
    acc_rate = (acc / props) if props > 0 else float('nan')

    print(f"\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,} | Dim: {D}")
    print("Last 5 ergodic mean |x| over dims:", erg_abs[-5:])
    print("Last 5 ergodic indicators any(|x|>5):", erg_ind[-5:])
    print(f"Chain 0 acceptance: {acc} / {props} = {acc_rate:.4f}")

    # Save CSVs next to this script
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(scriptdir, "ergodic_average_abs_RWM_20d_dpareto_pcg_f32.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows([[float(v)] for v in erg_abs])

    with open(os.path.join(scriptdir, "ergodic_average_indicator_RWM_20d_dpareto_pcg_f32.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows([[float(v)] for v in erg_ind])

    with open(os.path.join(scriptdir, "acceptance_chain0_RWM_20d_dpareto_pcg_f32.txt"), "w") as f:
        f.write(f"accepts={acc}, proposals={props}, accept_rate={acc_rate:.6f}\n")

    print("\nCSV files written:")
    print("  ergodic_average_abs_RWM_20d_dpareto_pcg_f32.csv")
    print("  ergodic_average_indicator_RWM_20d_dpareto_pcg_f32.csv")
    print("  acceptance_chain0_RWM_20d_dpareto_pcg_f32.txt")

if __name__ == "__main__":
    run()
