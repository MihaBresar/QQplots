# mala_arrayfire.py
import arrayfire as af
import math, time

af.set_backend('opencl')  # AMD on Windows
af.info()

N = 10_000_00             # per-chain steps (set 10_000_000 for big run)
burnin = 100
n_chains = 50_000
step = 0.55

def log_unnorm_t4(x):  return -af.log1p((x**4)/4)
def grad_log_unnorm_t4(x): return -x / (2*(1 + (x**4)/4))

def run():
    x  = af.constant(0, n_chains, dtype=af.Dtype.f32)
    lp = log_unnorm_t4(x)
    sum_abs = af.constant(0, n_chains, dtype=af.Dtype.f64)
    sum_ind = af.constant(0, n_chains, dtype=af.Dtype.f64)

    ss2 = step*step
    inv2s = 1.0/(2*ss2)

    t0 = time.time()
    kept = 0
    for t in range(1, N):
        z  = af.randn(n_chains, dtype=af.Dtype.f32)
        g  = grad_log_unnorm_t4(x)
        mean_fwd = x + (ss2/2)*g
        xprop = mean_fwd + step*z

        lp_prop = log_unnorm_t4(xprop)
        gp = grad_log_unnorm_t4(xprop)
        mean_bwd = xprop + (ss2/2)*gp

        lq_fwd = -inv2s*(xprop-mean_fwd)**2
        lq_bwd = -inv2s*(x-mean_bwd)**2

        loga = lp_prop + lq_bwd - (lp + lq_fwd)

        u = af.randu(n_chains, dtype=af.Dtype.f32)
        accept = af.log(u) < loga  # boolean mask

        x  = af.select(accept, xprop, x)
        lp = af.select(accept, lp_prop, lp)

        if t > burnin:
            ax = af.abs(x).as_type(af.Dtype.f64)
            sum_abs = sum_abs + ax
            sum_ind = sum_ind + (ax >= 2)
            kept += 1

    af.sync()
    elapsed = time.time() - t0
    erg_abs = (sum_abs/kept).to_ndarray()
    erg_ind = (sum_ind/kept).to_ndarray()

    print(f"Elapsed: {elapsed:.2f}s | Chains: {n_chains} | Steps/chain: {N}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

if __name__ == "__main__":
    run()
