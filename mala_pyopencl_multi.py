# mala_pyopencl_multi.py
import math, time, numpy as np, threading
import pyopencl as cl
import pyopencl.array as cl_array

# ---------------- config ----------------
N_TOTAL          = 30_000_000   # steps per chain (yep: 30 million)
BURNIN           = 10_000_000
N_CHAINS         = 50_000       # total chains across all devices
STEP_SIZE        = 0.55
STEPS_PER_LAUNCH = 4000         # each inner iter does 2 steps => 8000 steps/launch
USE_CPU_DEVICE   = True         # include an OpenCL CPU device if available

SEED             = 123456789

# -------------- kernel ------------------
KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline uint xorshift32(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
inline float u01(uint *s) {
    uint x = xorshift32(*s); *s = x;
    return fmax((x + 1u) * 2.3283064365386963e-10f, 1.0e-7f);
}
inline void randn_pair(uint *s, float *z0, float *z1) {
    float u1 = u01(s), u2 = u01(s);
    float r = sqrt(-2.0f * log(u1));
    float a = 6.283185307179586f * u2; // 2*pi
    *z0 = r * cos(a);
    *z1 = r * sin(a);
}
inline float log_unnorm_t4(float x) {
    float x2 = x*x, x4 = x2*x2;
    return -log(1.0f + 0.25f*x4);
}
inline float grad_log_unnorm_t4(float x) {
    float x2 = x*x, x4 = x2*x2;
    return -x / (2.0f*(1.0f + 0.25f*x4));
}

// Each inner loop performs TWO MALA steps (z0, z1) to halve RNG overhead.
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

    for (int it=0; it<K; ++it) {
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
        if (log(u01(&s)) < loga) { xx = xprop; lp = lp_prop; }
        if (b > 0) { --b; } else {
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
        if (log(u01(&s)) < loga) { xx = xprop; lp = lp_prop; }
        if (b > 0) { --b; } else {
            double ax = (double)fabs(xx);
            sum_abs[i] += ax; sum_ind[i] += (ax >= 2.0);
        }
    }
    x[i] = xx; logp[i] = lp; burn_left[i] = b; rng_state[i] = s;
}
"""

# -------------- device picking --------------
def list_devices():
    devs = []
    for plat in cl.get_platforms():
        for d in plat.get_devices():
            devs.append((plat, d))
    return devs

def choose_devices(include_cpu=True):
    all_devs = list_devices()
    gpus = [(p,d) for (p,d) in all_devs if d.type & cl.device_type.GPU]
    cpus = [(p,d) for (p,d) in all_devs if (d.type & cl.device_type.CPU)]
    chosen = gpus[:] + (cpus if include_cpu else [])
    if not chosen:
        raise RuntimeError("No OpenCL devices found.")
    return chosen

# -------------- worker (per device) --------------
def worker(plat, dev, n_local, seed, results, key):
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL).build(options=["-cl-fast-relaxed-math"])

    # device buffers
    x = cl_array.zeros(queue, n_local, dtype=np.float32)
    logp = cl_array.zeros(queue, n_local, dtype=np.float32)  # log p(0)=0
    sum_abs = cl_array.zeros(queue, n_local, dtype=np.float64)
    sum_ind = cl_array.zeros(queue, n_local, dtype=np.float64)
    burn_left = cl_array.to_device(queue, np.full(n_local, BURNIN, dtype=np.int32))

    # per-chain seeds (splitmix64 -> xorshift32)
    rng_h = np.empty(n_local, dtype=np.uint32)
    st = np.uint64(seed)
    for i in range(n_local):
        st = (st + 0x9E3779B97F4A7C15) & ((1<<64)-1)
        z = st
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1<<64)-1)
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1<<64)-1)
        z = z ^ (z >> 31)
        rng_h[i] = np.uint32(z & 0xFFFFFFFF)
    rng_state = cl_array.to_device(queue, rng_h)

    ss2 = np.float32(STEP_SIZE*STEP_SIZE)
    inv2s = np.float32(1.0/(2.0*ss2))

    # warmup
    prg.mala_chunk2(queue, (n_local,), None,
        x.data, logp.data, sum_abs.data, sum_ind.data,
        burn_left.data, rng_state.data,
        np.float32(STEP_SIZE), ss2, inv2s, np.int32(1)).wait()

    kept_target = max(0, (N_TOTAL-1) - BURNIN)  # per chain
    steps_done = 0
    # NOTE: each inner iter does **2 steps**, so total steps advanced per launch = 2*K
    while steps_done < (N_TOTAL - 1):
        K = min(STEPS_PER_LAUNCH, ((N_TOTAL - 1) - steps_done + 1)//2)
        prg.mala_chunk2(queue, (n_local,), None,
            x.data, logp.data, sum_abs.data, sum_ind.data,
            burn_left.data, rng_state.data,
            np.float32(STEP_SIZE), ss2, inv2s, np.int32(K)).wait()
        steps_done += 2*K

    # gather
    results[key] = (sum_abs.get(), sum_ind.get(), kept_target)

# -------------- driver --------------
def run():
    chosen = choose_devices(include_cpu=USE_CPU_DEVICE)
    # spread chains as evenly as possible
    chunks = []
    remaining = N_CHAINS
    per = max(1, N_CHAINS // len(chosen))
    for idx,(p,d) in enumerate(chosen):
        n_local = per if idx < len(chosen)-1 else remaining
        remaining -= n_local
        chunks.append((p,d,n_local))
    print("Devices used:")
    for (p,d,nl) in chunks:
        print(f"  - {p.name} | {d.name}  <= {nl} chains")

    # launch threads
    threads = []
    results = {}
    t0 = time.time()
    for j,(p,d,nl) in enumerate(chunks):
        t = threading.Thread(
            target=worker,
            args=(p,d,nl, SEED + 77777*j, results, j),
            daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    # merge
    erg_abs = np.concatenate([results[k][0]/max(results[k][2],1) for k in sorted(results)])
    erg_ind = np.concatenate([results[k][1]/max(results[k][2],1) for k in sorted(results)])

    print(f"\nElapsed: {elapsed:.2f}s | Total chains: {N_CHAINS:,} | Steps/chain: {N_TOTAL:,}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

if __name__ == "__main__":
    run()
