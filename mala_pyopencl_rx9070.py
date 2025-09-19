# mala_pyopencl_rx9070.py
# Runs many MALA chains on your AMD RX 9070 via OpenCL.
# Forces device [P0:D1] based on your listing:
#   [P0:D0] GPU gfx1036 (iGPU, 1 CU)
#   [P0:D1] GPU gfx1201 (RX 9070, 32 CUs)

import math, time, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# -------------------- CONFIG --------------------
# Big job (tune as needed)
N_TOTAL          = 30_000_000   # steps per chain
BURNIN           = 100
N_CHAINS         = 50_000       # total chains on the RX 9070
STEP_SIZE        = 0.55
STEPS_PER_LAUNCH = 4000         # kernel does 2 steps/iter -> 8000 steps per launch
SEED             = 123456789

# Device selection (YOUR RX 9070 = P0:D1)
PLATFORM_INDEX   = 0
DEVICE_INDEX     = 1            # <-- forces RX 9070 (gfx1201)
USE_BOTH_DEVICES = False        # True to also use P0:D0 (not recommended: only 1 CU)

# -------------------- KERNEL --------------------
# 2 MALA steps per inner-iter (uses both Box–Muller normals)
KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline uint xorshift32(uint x){ x^=x<<13; x^=x>>17; x^=x<<5; return x; }
inline float u01(uint *s){
    uint x = xorshift32(*s); *s = x;
    return fmax((x + 1u) * 2.3283064365386963e-10f, 1.0e-7f);
}
inline void randn_pair(uint *s, float *z0, float *z1){
    float u1 = u01(s), u2 = u01(s);
    float r = sqrt(-2.0f * log(u1));
    float a = 6.283185307179586f * u2;
    *z0 = r * cos(a); *z1 = r * sin(a);
}
inline float log_unnorm_t4(float x){
    float x2=x*x, x4=x2*x2;
    return -log(1.0f + 0.25f*x4);
}
inline float grad_log_unnorm_t4(float x){
    float x2=x*x, x4=x2*x2;
    return -x / (2.0f*(1.0f + 0.25f*x4));
}

__kernel void mala_chunk2(
    __global float  *x, __global float  *logp,
    __global double *sum_abs, __global double *sum_ind,
    __global int    *burn_left, __global uint *rng_state,
    const float step, const float ss2, const float inv2s, const int K)
{
    int i = get_global_id(0);
    float xx = x[i];
    float lp = logp[i];
    int   b  = burn_left[i];
    uint  s  = rng_state[i];

    for (int it=0; it<K; ++it){
        float z0, z1; randn_pair(&s, &z0, &z1);

        // ---- step A ----
        float g  = grad_log_unnorm_t4(xx);
        float mf = xx + 0.5f*ss2*g;
        float xprop = mf + step*z0;
        float lp_prop = log_unnorm_t4(xprop);
        float gp = grad_log_unnorm_t4(xprop);
        float mb = xprop + 0.5f*ss2*gp;
        float df = xprop - mf, db = xx - mb;
        float loga = lp_prop - lp - inv2s*(df*df - db*db);
        if (log(u01(&s)) < loga){ xx = xprop; lp = lp_prop; }
        if (b > 0) --b; else {
            double ax = (double)fabs(xx);
            sum_abs[i] += ax; sum_ind[i] += (ax >= 2.0);
        }

        // ---- step B ----
        g  = grad_log_unnorm_t4(xx);
        mf = xx + 0.5f*ss2*g;
        xprop = mf + step*z1;
        lp_prop = log_unnorm_t4(xprop);
        gp = grad_log_unnorm_t4(xprop);
        mb = xprop + 0.5f*ss2*gp;
        df = xprop - mf; db = xx - mb;
        loga = lp_prop - lp - inv2s*(df*df - db*db);
        if (log(u01(&s)) < loga){ xx = xprop; lp = lp_prop; }
        if (b > 0) --b; else {
            double ax = (double)fabs(xx);
            sum_abs[i] += ax; sum_ind[i] += (ax >= 2.0);
        }
    }
    x[i]=xx; logp[i]=lp; burn_left[i]=b; rng_state[i]=s;
}
"""

# -------------------- UTIL --------------------
def build_program(ctx):
    return cl.Program(ctx, KERNEL).build(options=["-cl-fast-relaxed-math"])

def seed_array(n, seed):
    rng_h = np.empty(n, dtype=np.uint32)
    st = np.uint64(seed)
    for i in range(n):
        st = (st + 0x9E3779B97F4A7C15) & ((1<<64)-1)
        z = st
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1<<64)-1)
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1<<64)-1)
        z = z ^ (z >> 31)
        rng_h[i] = np.uint32(z & 0xFFFFFFFF)
    return rng_h

# -------------------- SINGLE-DEVICE (RX 9070) --------------------
def run_single_device():
    plats = cl.get_platforms()
    dev = plats[PLATFORM_INDEX].get_devices()[DEVICE_INDEX]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    print("Using device:", dev.platform.name, "|", dev.name)

    prg = build_program(ctx)

    n = N_CHAINS
    x = cl_array.zeros(queue, n, dtype=np.float32)
    logp = cl_array.zeros(queue, n, dtype=np.float32)  # log p(0)=0
    sum_abs = cl_array.zeros(queue, n, dtype=np.float64)
    sum_ind = cl_array.zeros(queue, n, dtype=np.float64)
    burn_left = cl_array.to_device(queue, np.full(n, BURNIN, dtype=np.int32))
    rng_state = cl_array.to_device(queue, seed_array(n, SEED))

    ss2 = np.float32(STEP_SIZE*STEP_SIZE)
    inv2s = np.float32(1.0/(2.0*ss2))

    # warmup
    prg.mala_chunk2(queue, (n,), None,
        x.data, logp.data, sum_abs.data, sum_ind.data,
        burn_left.data, rng_state.data,
        np.float32(STEP_SIZE), ss2, inv2s, np.int32(1)).wait()

    steps_done = 0
    t0 = time.time()
    # each inner-iter does 2 steps -> total advanced per launch = 2*K
    while steps_done < (N_TOTAL - 1):
        K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
        prg.mala_chunk2(queue, (n,), None,
            x.data, logp.data, sum_abs.data, sum_ind.data,
            burn_left.data, rng_state.data,
            np.float32(STEP_SIZE), ss2, inv2s, np.int32(K)).wait()
        steps_done += 2*K

    elapsed = time.time() - t0
    kept = max(0, (N_TOTAL - 1) - BURNIN)
    erg_abs = (sum_abs.get()/max(kept,1))
    erg_ind = (sum_ind.get()/max(kept,1))

    print(f"\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

# -------------------- OPTIONAL: MULTI-DEVICE --------------------
# Will use both P0:D1 (RX 9070) and P0:D0 (iGPU). RX will get ~all chains.
import threading
def run_multi_device():
    plats = cl.get_platforms()
    all_devs = plats[0].get_devices()  # your listing shows everything on P0
    rx = all_devs[1]   # D1: RX 9070
    ig = all_devs[0]   # D0: iGPU (1 CU)

    # weight by compute units; give tiny slice to iGPU
    cu_rx = max(rx.max_compute_units, 1)
    cu_ig = max(ig.max_compute_units, 1)
    w_rx, w_ig = cu_rx, max(1, min(2, cu_ig))  # cap iGPU weight ~1–2
    total_w = w_rx + w_ig
    n_rx = int(N_CHAINS * w_rx / total_w)
    n_ig = N_CHAINS - n_rx

    def worker(dev, n_local, seed, out):
        ctx = cl.Context([dev])
        queue = cl.CommandQueue(ctx)
        prg = build_program(ctx)

        x = cl_array.zeros(queue, n_local, dtype=np.float32)
        logp = cl_array.zeros(queue, n_local, dtype=np.float32)
        sum_abs = cl_array.zeros(queue, n_local, dtype=np.float64)
        sum_ind = cl_array.zeros(queue, n_local, dtype=np.float64)
        burn_left = cl_array.to_device(queue, np.full(n_local, BURNIN, dtype=np.int32))
        rng_state = cl_array.to_device(queue, seed_array(n_local, seed))

        ss2 = np.float32(STEP_SIZE*STEP_SIZE)
        inv2s = np.float32(1.0/(2.0*ss2))

        prg.mala_chunk2(queue, (n_local,), None,
            x.data, logp.data, sum_abs.data, sum_ind.data,
            burn_left.data, rng_state.data,
            np.float32(STEP_SIZE), ss2, inv2s, np.int32(1)).wait()

        steps_done = 0
        while steps_done < (N_TOTAL - 1):
            K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
            prg.mala_chunk2(queue, (n_local,), None,
                x.data, logp.data, sum_abs.data, sum_ind.data,
                burn_left.data, rng_state.data,
                np.float32(STEP_SIZE), ss2, inv2s, np.int32(K)).wait()
            steps_done += 2*K

        kept = max(0, (N_TOTAL - 1) - BURNIN)
        out.append((sum_abs.get()/max(kept,1), sum_ind.get()/max(kept,1)))

    print("Devices used:")
    print(f"  RX 9070: {rx.name}  <= {n_rx} chains")
    print(f"  iGPU   : {ig.name}  <= {n_ig} chains")

    out = []
    t0 = time.time()
    th1 = threading.Thread(target=worker, args=(rx, n_rx, SEED+101, out), daemon=True)
    th1.start()
    ths = [th1]
    if n_ig > 0:
        th2 = threading.Thread(target=worker, args=(ig, n_ig, SEED+202, out), daemon=True)
        th2.start()
        ths.append(th2)
    for t in ths: t.join()
    elapsed = time.time() - t0

    erg_abs = np.concatenate([o[0] for o in out]) if out else np.zeros(0)
    erg_ind = np.concatenate([o[1] for o in out]) if out else np.zeros(0)
    print(f"\nElapsed: {elapsed:.2f}s | Chains: {len(erg_abs):,} | Steps/chain: {N_TOTAL:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

# -------------------- MAIN --------------------
if __name__ == "__main__":
    if USE_BOTH_DEVICES:
        run_multi_device()
    else:
        run_single_device()
