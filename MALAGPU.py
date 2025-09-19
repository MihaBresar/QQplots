# mala_taichi.py
import math, time
import taichi as ti

# Try DX12 first on Windows; fall back to Vulkan if needed
try:
    ti.init(arch=ti.dx12)
except Exception:
    ti.init(arch=ti.vulkan)

# ---------- Parameters ----------
N_total         = 10_000_00   # total steps per chain (set 10_000_000 for your big run)
burnin          = 100
n_chains        = 50_000      # increase to saturate GPU
step_size       = 0.55
steps_per_launch= 1000        # do 1000 MALA steps per kernel launch to cut overhead

# ---------- Fields ----------
n = n_chains
x      = ti.field(dtype=ti.f32, shape=n)        # chain states
logp   = ti.field(dtype=ti.f32, shape=n)        # log target at x
sum_abs= ti.field(dtype=ti.f64, shape=n)        # running |x| (use f64 to reduce drift)
sum_ind= ti.field(dtype=ti.f64, shape=n)        # running 1{|x|>=2}
kept   = ti.field(dtype=ti.i32, shape=())       # how many post-burnin steps accumulated
burn   = ti.field(dtype=ti.i32, shape=())       # mutable burnin counter

# ---------- Helpers ----------
@ti.func
def log_unnorm_t4(x):
    # -log1p(x^4/4)
    return -ti.log(1.0 + (x*x*x*x)*0.25)

@ti.func
def grad_log_unnorm_t4(x):
    # -x / (2*(1 + x^4/4))
    return -x / (2.0 * (1.0 + (x*x*x*x)*0.25))

@ti.func
def randn_pair():
    # Box–Muller using two uniforms in (0,1]
    u1 = ti.max(1e-7, ti.random(dtype=ti.f32))
    u2 = ti.random(dtype=ti.f32)
    r  = ti.sqrt(-2.0 * ti.log(u1))
    ang= 2.0 * math.pi * u2
    return r * ti.cos(ang), r * ti.sin(ang)

# ---------- Kernel: do K steps per launch ----------
@ti.kernel
def mala_steps(K: ti.i32, step_size: ti.f32):
    ss2 = step_size * step_size
    inv2s = 1.0 / (2.0 * ss2)
    for i in range(n):
        xx = x[i]
        lp = logp[i]
        b  = burn[None]
        kept_local = 0

        for _ in range(K):
            # proposal
            g  = grad_log_unnorm_t4(xx)
            mean_fwd = xx + 0.5 * ss2 * g
            z0, _ = randn_pair()
            xprop = mean_fwd + step_size * z0

            lp_prop = log_unnorm_t4(xprop)
            gp      = grad_log_unnorm_t4(xprop)
            mean_bwd = xprop + 0.5 * ss2 * gp

            lq_fwd = -inv2s * (xprop - mean_fwd) * (xprop - mean_fwd)
            lq_bwd = -inv2s * (xx    - mean_bwd) * (xx    - mean_bwd)

            loga = lp_prop + lq_bwd - (lp + lq_fwd)

            # accept/reject
            u = ti.random(dtype=ti.f32)
            accept = ti.log(u) < loga
            if accept:
                xx = xprop
                lp = lp_prop

            # online stats after burn-in
            if b > 0:
                b -= 1
            else:
                ax = ti.abs(xx)
                # Kahan/Neumaier not needed if we use f64 sums; good enough for 1e7 steps
                sum_abs[i] += ti.cast(ax, ti.f64)
                sum_ind[i] += ti.cast(1.0 if ax >= 2.0 else 0.0, ti.f64)
                kept_local += 1

        x[i]   = xx
        logp[i]= lp
        burn[None] = b
        ti.atomic_add(kept[None], kept_local)

# ---------- Driver ----------
def run():
    # init
    x.fill(0.0)
    for i in range(n):
        logp[i] = float(log_unnorm_t4(0.0))
    sum_abs.fill(0.0)
    sum_ind.fill(0.0)
    kept.fill(0)
    burn.fill(burnin)

    t0 = time.time()
    steps_done = 0
    while steps_done < N_total - 1:
        K = min(steps_per_launch, (N_total - 1) - steps_done)
        mala_steps(K, float(step_size))
        steps_done += K
    ti.sync()
    elapsed = time.time() - t0

    kept_steps = max(kept.to_numpy().item(), 1)
    erg_abs = (sum_abs.to_numpy() / kept_steps)
    erg_ind = (sum_ind.to_numpy() / kept_steps)

    print(f"Elapsed: {elapsed:.2f}s | Chains: {n} | Steps/chain: {N_total} | Kept/chain: ~{kept_steps}")
    print("Last 5 ergodic |x| averages:", erg_abs[-5:])
    print("Last 5 ergodic indicators :", erg_ind[-5:])

if __name__ == "__main__":
    run()
