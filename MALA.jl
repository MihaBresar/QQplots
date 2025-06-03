using Random
using Distributions
using DelimitedFiles
using Statistics
using Dates


# ----------  target density, log-density, and gradient ----------
@inline function unnormalized_t4_like_pdf(x)
    # 1 / (1 + x²/4)
    return inv(1 + x^4 / 4)
end

@inline function log_unnormalized_t4_like_pdf(x)
    # log of the above (negative)
    return -log1p(x^4 / 4)
end

@inline function grad_log_unnormalized_t4_like_pdf(x)
    # ∂/∂x log π̃(x)  =  −x / (2(1 + x²/4))
    return -x / (2 * (1 + x^4 / 4))
end

# ----------  MALA kernel ---------------------------------------
"""
    mala_t4_like(
        n_iter::Int,
        initial::Float64,
        step_size::Float64,
        burnin::Int,
        rng::AbstractRNG)

Run a univariate MALA chain targeting π̃(x) ∝ 1/(1+x²/4).

Returns `(chain, ergodic_sum_abs, ergodic_sum_indicator)`.
"""
function mala_t4_like(n_iter::Int, initial::Float64, step_size::Float64,
                      burnin::Int, rng::AbstractRNG)

    chain = Vector{Float64}(undef, n_iter)
    chain[1] = initial

    log_p_current = log_unnormalized_t4_like_pdf(initial)

    ergodic_sum_abs      = 0.0
    ergodic_sum_indicator = 0.0

    ss2   = step_size^2
    inv2s = 1.0 / (2 * ss2)      # 1 / (2 σ²) for Gaussian kernel

    for i in 2:n_iter
        x_curr = chain[i-1]
        g_curr = grad_log_unnormalized_t4_like_pdf(x_curr)

        # Langevin drift: x + (σ²/2) ∇ log π̃(x)
        mean_fwd = x_curr + (ss2 / 2) * g_curr
        x_prop   = mean_fwd + step_size * randn(rng)

        # log π̃ at proposal
        log_p_prop = log_unnormalized_t4_like_pdf(x_prop)

        # gradients for backward transition
        g_prop     = grad_log_unnormalized_t4_like_pdf(x_prop)
        mean_bwd   = x_prop + (ss2 / 2) * g_prop

        # Gaussian proposal logs (constants cancel)
        log_q_fwd = -inv2s * (x_prop - mean_fwd)^2
        log_q_bwd = -inv2s * (x_curr - mean_bwd)^2

        # MH acceptance
        log_α = log_p_prop + log_q_bwd - (log_p_current + log_q_fwd)
        if log(rand(rng)) < log_α
            chain[i]      = x_prop
            log_p_current = log_p_prop
        else
            chain[i] = x_curr
        end

        if i > burnin
            abs_val = abs(chain[i])
            ergodic_sum_abs      += abs_val
            ergodic_sum_indicator += abs_val >= 2.0 ? 1.0 : 0.0
        end
    end

    return chain, ergodic_sum_abs, ergodic_sum_indicator
end

# ----------  driver -------------------------------------------
function main()
    # Simulation parameters
    N              = 10_000_00     # samples per chain
    burnin         = 100_000
    n_simulations  = 11_000
    step_size      = 0.55          # MALA step-size  (tune for ≈ 57 % accept.)

    erg_abs        = zeros(Float64, n_simulations)
    erg_indicator  = zeros(Float64, n_simulations)
    last5_chain1   = zeros(Float64, 5)

    start_time = time()

    Threads.@threads for i in 1:n_simulations
        rng = MersenneTwister(1234 + i)
        chain, s_abs, s_ind = mala_t4_like(N, 0.0, step_size, burnin, rng)
        erg_abs[i]       = s_abs / (N - burnin)
        erg_indicator[i] = s_ind / (N - burnin)

        if i == 1                                # stash last 5 from first chain
            last5_chain1 .= chain[end-4:end]
        end
    end

    elapsed = time() - start_time

    println("Last 5 samples of the first MALA chain:")
    println(last5_chain1)
    println("Last 5 ergodic averages for |x| (from the last 5 chains):")
    println(erg_abs[end-4:end])
    println("Last 5 ergodic averages for indicator 1(|x| ≥ 2) (from the last 5 chains):")
    println(erg_indicator[end-4:end])
    println("Total elapsed time: $(elapsed) seconds.")

    # --- write CSV ---
    scriptdir = @__DIR__

    writedlm(joinpath(scriptdir, "ergodic_average_abs_MALA.csv"),       erg_abs, ',')
    writedlm(joinpath(scriptdir, "ergodic_average_indicator_MALA.csv"), erg_indicator, ',')
    println("CSV files written.")
end

main()
