# ------------------------------------------------------------------
#  ULA for heavy–tailed 1/(1+|x|³)   or   1/(1+x⁴)
# ------------------------------------------------------------------
using Random, DelimitedFiles
import Base.Threads                  # gives Threads.@threads

# ──────────────────────────────────────────────────────────────────
# 1 ▸ CHOOSE TARGET DISTRIBUTION
#    :abs3   →  π̃(x) ∝ 1/(1+|x|³)
#    :x4     →  π̃(x) ∝ 1/(1+x⁴)
# ──────────────────────────────────────────────────────────────────
const TARGET = :x4          # ← change to :abs3 if you want the |x|³ model

# ──────────────────────────────────────────────────────────────────
# 2 ▸ GRADIENT OF log π̃  (selected automatically)
# ──────────────────────────────────────────────────────────────────
@inline function grad_log_pi(x::Float64)
    if TARGET === :abs3
        # d/dx log 1/(1+|x|³) = −3 |x|² sign(x)/(1+|x|³)
        if x == 0
            return 0.0
        else
            return -3 * abs(x)^2 * sign(x) / (1 + abs(x)^3)
        end
    elseif TARGET === :x4
        # d/dx log 1/(1+x⁴) = −4 x³ /(1+x⁴)
        return -4 * x^3 / (1 + x^4)
    else
        error("Unsupported TARGET = $TARGET")
    end
end

# ──────────────────────────────────────────────────────────────────
# 3 ▸ ULA kernel (unchanged)
# ──────────────────────────────────────────────────────────────────
function ula_generic(n_iter::Int, initial::Float64, step_size::Float64,
                     burnin::Int, rng::AbstractRNG)
    chain        = Vector{Float64}(undef, n_iter)
    chain[1]     = initial
    ss2          = step_size^2
    half_ss2     = ss2/2
    erg_abs      = 0.0
    erg_indicator = 0.0

    for i in 2:n_iter
        x        = chain[i-1]
        x_next   = x + half_ss2 * grad_log_pi(x) + step_size * randn(rng)
        chain[i] = x_next

        if i > burnin
            z = abs(x_next)
            erg_abs      += z
            erg_indicator += (z ≥ 2.0) ? 1.0 : 0.0
        end
    end
    return chain, erg_abs, erg_indicator
end

# ──────────────────────────────────────────────────────────────────
# 4 ▸ Driver
# ──────────────────────────────────────────────────────────────────
function main()
    N, burnin        = 10_000_000, 1000_000
    n_simulations    = 11_000
    step_size        = TARGET === :abs3 ? 0.07 : 0.05   # heuristics

    erg_abs  = zeros(Float64, n_simulations)
    erg_ind  = zeros(Float64, n_simulations)
    last5    = zeros(Float64, 5)

    t0 = time()
    Threads.@threads for i in 1:n_simulations
        rng = MersenneTwister(1234 + i)
        ch, s_abs, s_ind = ula_generic(N, 0.0, step_size, burnin, rng)
        erg_abs[i] = s_abs / (N - burnin)
        erg_ind[i] = s_ind / (N - burnin)
        if i == 1
            last5 .= ch[end-4:end]
        end
    end
    println("Target: ", TARGET)
    println("Last 5 samples of first chain: ", last5)
    println("Last 5 ergodic averages |x|:   ", erg_abs[end-4:end])
    println("Last 5 ergodic 𝟙{|x|≥2}:        ", erg_ind[end-4:end])
    println("Elapsed ", round(time()-t0, digits=2), " s")

    root = @__DIR__
    writedlm(joinpath(root, "ergodic_abs_ULA_$TARGET.csv"), erg_abs, ',')
    writedlm(joinpath(root, "ergodic_ind_ULA_$TARGET.csv"), erg_ind, ',')
end

main()
