using Random
using DelimitedFiles
using Statistics
using Base.Threads
using BenchmarkTools

# --- Optimized versions with @fastmath, @inbounds, preallocation ---
@inline @fastmath function unnormalized_t4_like_pdf(x)
    return inv(1 + x^4 / 4)
end

@inline @fastmath function log_unnormalized_t4_like_pdf(x)
    return -log1p(x^4 / 4)
end

@inline @fastmath function grad_log_unnormalized_t4_like_pdf(x)
    return -x / (2 * (1 + x^4 / 4))
end

function mala_t4_like(n_iter::Int, initial::Float64, step_size::Float64,
                      burnin::Int, rng::AbstractRNG)

    chain = Vector{Float64}(undef, n_iter)
    chain[1] = initial

    log_p_current = log_unnormalized_t4_like_pdf(initial)

    ergodic_sum_abs      = 0.0
    ergodic_sum_indicator = 0.0

    ss2   = step_size^2
    inv2s = 1.0 / (2 * ss2)

    # Pre-generate all random numbers for this chain
    rand_vals = randn(rng, n_iter - 1)

    @inbounds for i in 2:n_iter
        x_curr = chain[i-1]
        g_curr = grad_log_unnormalized_t4_like_pdf(x_curr)

        mean_fwd = x_curr + (ss2 / 2) * g_curr
        x_prop   = mean_fwd + step_size * rand_vals[i-2]

        log_p_prop = log_unnormalized_t4_like_pdf(x_prop)

        g_prop     = grad_log_unnormalized_t4_like_pdf(x_prop)
        mean_bwd   = x_prop + (ss2 / 2) * g_prop

        log_q_fwd = -inv2s * (x_prop - mean_fwd)^2
        log_q_bwd = -inv2s * (x_curr - mean_bwd)^2

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

function main_cpu()
    println("Number of threads available: ", Threads.nthreads())
    println("Number of CPU cores: ", Sys.CPU_THREADS)

    N              = 10_000
    burnin         = 100
    n_simulations  = 11_000
    step_size      = 0.55

    erg_abs        = zeros(Float64, n_simulations)
    erg_indicator  = zeros(Float64, n_simulations)
    last5_chain1   = zeros(Float64, 5)

    # Thread-local RNGs (critical!)
    rngs = [MersenneTwister(rand(UInt32)) for _ in 1:nthreads()]

    start_time = time()

    Threads.@threads for i in 1:n_simulations
        tid = Threads.threadid()
        local_rng = rngs[tid]

        chain, s_abs, s_ind = mala_t4_like(N, 0.0, step_size, burnin, local_rng)
        erg_abs[i]       = s_abs / (N - burnin)
        erg_indicator[i] = s_ind / (N - burnin)

        if i == 1
            last5_chain1 .= chain[end-4:end]
        end
    end

    elapsed = time() - start_time

    println("Total elapsed time: $(round(elapsed, digits=2)) seconds")
    println("Throughput: $(round(n_simulations / elapsed, digits=1)) chains/second")

    println("\nLast 5 samples of first chain:")
    println(last5_chain1)
    println("Last 5 ergodic averages for |x|:")
    println(erg_abs[end-4:end])
    println("Last 5 ergodic averages for indicator (|x| ≥ 2):")
    println(erg_indicator[end-4:end])

    scriptdir = @__DIR__
    writedlm(joinpath(scriptdir, "ergodic_average_abs_MALA.csv"), erg_abs, ',')
    writedlm(joinpath(scriptdir, "ergodic_average_indicator_MALA.csv"), erg_indicator, ',')
    println("\nCSV files written.")
end

# Run!
main_cpu()