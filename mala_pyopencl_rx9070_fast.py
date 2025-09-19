# mala_pyopencl_rx9070_fast.py
# AMD RX 9070 (gfx1201) OpenCL MALA, tuned for higher utilization.

import time, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ======== TUNABLES ========
N_TOTAL          = 30_000_000   # steps per chain
BURNIN           = 100
N_CHAINS         = 200_000      # try 100k–300k to load the GPU
STEP_SIZE        = 0.55
STEPS_PER_LAUNCH = 8000         # inner iter does 2 steps -> 16k steps/launch
LOCAL_SIZE       = 256          # work-group size (try 128/256/512)
SEED             = 123456789

# Force RX 9070 from your listing: Platform 0, Device 1
PLATFORM_INDEX   = 0
DEVICE_INDEX     = 1

# ======== KERNEL (2 steps per inner loop) ========
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

# ======== UTIL ========
def build_program(ctx):
    # extra fast-math flags
    return cl.Program(ctx, KERNEL).build(options=[
        "-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros"
    ])

def seed_array(n, seed):
    """SplitMix64 → xorshift32 seeds, without NumPy overflow warnings."""
    out = np.empty(n, dtype=np.uint32)
    mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)
    st = np.uint64(seed)
    for i in range(n):
        st = (st + np.uint64(0x9E3779B97F4A7C15)) & mask64
        z = st
        z ^= (z >> np.uint64(30))
        z = (z * np.uint64(0xBF58476D1CE4E5B9)) & mask64
        z ^= (z >> np.uint64(27))
        z = (z * np.uint64(0x94D049BB133111EB)) & mask64
        z ^= (z >> np.uint64(31))
        out[i] = np.uint32(z & np.uint64(0xFFFFFFFF))
    return out

# ======== RUN (single device RX 9070) ========
def run():
    plats = cl.get_platforms()
    dev = plats[PLATFORM_INDEX].get_devices()[DEVICE_INDEX]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    print("Using device:", dev.platform.name, "|", dev.name)

    prg = build_program(ctx)
    kernel = cl.Kernel(prg, "mala_chunk2")  # reuse kernel object (no warning)

    n = N_CHAINS
    x = cl_array.zeros(queue, n, dtype=np.float32)
    logp = cl_array.zeros(queue, n, dtype=np.float32)  # log p(0)=0
    sum_abs = cl_array.zeros(queue, n, dtype=np.float64)
    sum_ind = cl_array.zeros(queue, n, dtype=np.float64)
    burn_left = cl_array.to_device(queue, np.full(n, BURNIN, dtype=np.int32))
    rng_state = cl_array.to_device(queue, seed_array(n, SEED))

    ss2 = np.float32(STEP_SIZE*STEP_SIZE)
    inv2s = np.float32(1.0/(2.0*ss2))

    # Set static args once; K (arg #10) will be updated per launch
    kernel.set_arg(0,  x.data)        ; kernel.set_arg(1,  logp.data)
    kernel.set_arg(2,  sum_abs.data)  ; kernel.set_arg(3,  sum_ind.data)
    kernel.set_arg(4,  burn_left.data); kernel.set_arg(5,  rng_state.data)
    kernel.set_arg(6,  np.float32(STEP_SIZE))
    kernel.set_arg(7,  ss2)
    kernel.set_arg(8,  inv2s)
    # arg 9 = K, set in the loop

    # Warmup (K=1)
    kernel.set_arg(9, np.int32(1))
    global_size = (n,)
    local_size  = (LOCAL_SIZE,)
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    steps_done = 0
    t0 = time.time()
    while steps_done < (N_TOTAL - 1):
        # each inner-iter does 2 steps -> we advance 2*K steps per launch
        K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
        kernel.set_arg(9, np.int32(K))
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        steps_done += 2*K

    elapsed = time.time() - t0
    kept = max(0, (N_TOTAL - 1) - BURNIN)
    erg_abs = (sum_abs.get()/max(kept,1))
    erg_ind = (sum_ind.get()/max(kept,1))

    print(f"\nElapsed: {elapsed:.2f}s | Chains: {n:,} | Steps/chain: {N_TOTAL:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

if __name__ == "__main__":
    run()
