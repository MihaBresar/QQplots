##############################################################################
#  SPS simulation for the unnormalised density  π(x) ∝ 1 / (1 + x² / 4)      #
#  (heavy-tailed “t₄-like” target)                                           #
#                                                                            #
#  Start Julia with threads, e.g.                                            #
#      julia --threads=8 sps_simulation.jl                                   #
##############################################################################

using Random, Distributions, DelimitedFiles, Statistics, Dates,
      LinearAlgebra, StaticArrays

# ---------------------------------------------------------------------------
#  Target density  (unnormalised)  π(x) = 1 / (1 + x²/4)
# ---------------------------------------------------------------------------
@inline function unnormalized_t4_like_pdf(x::Float64)
    return 1.0 / (1.0 + 0.25 * abs(x)^(4))
end

# ---------------------------------------------------------------------------
#  Stereographic projection helpers (ℝ¹  ↔  S¹ ⊂ ℝ²)
# ---------------------------------------------------------------------------
@inline function stereographic_inverse_1d(x::Float64)
    denom = 1.0 + x^2
    y     = 2.0 * x / denom
    z     = (x^2 - 1.0) / denom
    return y, z                    # returns coordinates on S¹
end

@inline function stereographic_projection_1d(y::Float64, z::Float64)
    return y / (1.0 - z)           # maps back to ℝ
end

# ---------------------------------------------------------------------------
#  Stereographic Projection Sampler (Algorithm 1, d = 1)
# ---------------------------------------------------------------------------
function sps_t4_like(n_iter::Int, initial::Float64, step_size::Float64,
                     burnin::Int, rng::AbstractRNG)

    chain   = Vector{Float64}(undef, n_iter)
    chain[1] = initial
    p_curr  = unnormalized_t4_like_pdf(initial)

    ergodic_sum_abs        = 0.0
    ergodic_sum_indicator  = 0.0
    n_accept               = 0

    for i in 2:n_iter
      # 1. current point on the circle
        y, z = stereographic_inverse_1d(chain[i-1])

# 2. isotropic Gaussian in the tangent plane
        tilde_y = step_size * randn(rng)
        tilde_z = step_size * randn(rng)

        dotprod  = y*tilde_y + z*tilde_z
        z_prime_y = tilde_y - dotprod*y          # projection
        z_prime_z = tilde_z - dotprod*z

        # normalise back to S¹
        len      = sqrt((y + z_prime_y)^2 + (z + z_prime_z)^2)
        hat_y    = (y + z_prime_y) / len
        hat_z    = (z + z_prime_z) / len
        hat_x    = stereographic_projection_1d(hat_y, hat_z)

        # 3. MH acceptance probability  α = π(x̂)(1+x̂²) / π(x)(1+x²)
        num = unnormalized_t4_like_pdf(hat_x) * (1.0 + hat_x^2)
        den = p_curr                       * (1.0 + chain[i-1]^2)
        α   = min(1.0, num / den)

        if rand(rng) < α
            chain[i] = hat_x
            p_curr   = unnormalized_t4_like_pdf(hat_x)
            n_accept += 1
        else
            chain[i] = chain[i-1]
        end

        if i > burnin
            abs_val = abs(chain[i])
            ergodic_sum_abs       += abs_val
            ergodic_sum_indicator += (abs_val >= 2.0) ? 1.0 : 0.0
        end
    end

    accept_rate = n_accept / (n_iter - 1)
    return chain, ergodic_sum_abs, ergodic_sum_indicator, accept_rate
end

# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
function main()
    # --- simulation parameters
    N              = 5_000_00     # total iterations
    burnin         = 500_00
    n_simulations  = 11_000        # independent chains
    step_size      = 0.2          # SPS Gaussian scale  (tune for ≈35 % acceptance)

    erg_abs        = zeros(Float64, n_simulations)
    erg_ind        = zeros(Float64, n_simulations)
    accept_rates   = zeros(Float64, n_simulations)
    last5_chain1   = zeros(Float64, 5)

    start_time = time()

    Threads.@threads for i in 1:n_simulations
        rng = MersenneTwister(1234 + i)                 # per-thread RNG
        chain, sum_abs, sum_ind, acc = sps_t4_like(N, 0.0, step_size,
                                                   burnin, rng)
        erg_abs[i]      = sum_abs / (N - burnin)
        erg_ind[i]      = sum_ind / (N - burnin)
        accept_rates[i] = acc

        if i == 1
            last5_chain1 .= chain[end-4:end]
        end
    end

    elapsed = time() - start_time

    println("Last 5 samples of the first SPS chain:")
    println(last5_chain1)
    println("Last 5 ergodic averages for |x| (chains $(n_simulations-4):$n_simulations):")
    println(erg_abs[end-4:end])
    println("Last 5 ergodic averages for 1(|x| ≥ 2) (chains $(n_simulations-4):$n_simulations):")
    println(erg_ind[end-4:end])
    println("Mean acceptance rate across chains: $(round(mean(accept_rates) * 100, digits=2)) %")
    println("Total elapsed time: $(elapsed) seconds.")

    # --- write CSVs
    scriptdir = @__DIR__
    writedlm(joinpath(scriptdir, "ergodic_average_abs_SPS_v=3.csv"),      erg_abs, ',')
    writedlm(joinpath(scriptdir, "ergodic_average_indicator_SPS_v=3.csv"), erg_ind, ',')
    println("CSV files written.")
end

main()
