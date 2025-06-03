using Random
using Distributions
using DelimitedFiles
using Statistics
using Dates

# Unnormalized PDF: 1/(1 + x²/4)
@inline function unnormalized_t4_like_pdf(x)
    tmp = 1.0 + (x^2)/4.0
    return 1.0 / tmp
end

# Random Walk Metropolis using a Cauchy proposal.
# Returns the chain (as a vector), the accumulated sum of |x| (post burn-in)
# and the accumulated sum of indicator{ |x| >= 2 }.
function rwm_t4_like_normal(n_iter::Int, initial::Float64, proposal_scale::Float64,
                            burnin::Int, rng::AbstractRNG)
    chain = Vector{Float64}(undef, n_iter)
    chain[1] = initial
    p_current = unnormalized_t4_like_pdf(chain[1])
    # Create a Cauchy distribution with location 0 and scale 1.
    cauchy_dist = Cauchy(0, 1)
    
    ergodic_sum_abs = 0.0
    ergodic_sum_indicator = 0.0

    for i in 2:n_iter
        # Draw a proposal step from the Cauchy distribution and scale it.
        step = rand(rng, cauchy_dist) * proposal_scale
        x_prop = chain[i - 1] + step
        p_prop = unnormalized_t4_like_pdf(x_prop)
        α = p_prop / p_current
        if rand(rng) < α
            chain[i] = x_prop
            p_current = p_prop
        else
            chain[i] = chain[i - 1]
        end

        if i > burnin
            abs_val = abs(chain[i])
            ergodic_sum_abs += abs_val
            ergodic_sum_indicator += (abs_val >= 2.0) ? 1.0 : 0.0
        end
    end

    return chain, ergodic_sum_abs, ergodic_sum_indicator
end

function main()
    # Simulation parameters
    N = 1_000_000              # Total samples per chain
    burnin = 100_000           # Burn-in period
    n_simulations = 11_000     # Number of independent chains
    proposal_scale = 5.0       # Scale parameter for the proposal

    ergodic_averages_abs = zeros(Float64, n_simulations)
    ergodic_averages_indicator = zeros(Float64, n_simulations)
    last5_samples_chain_1 = zeros(Float64, 5)

    # Record the start time.
    start_time = time()

    # Run simulations in parallel. Make sure to start Julia with the desired
    # number of threads (e.g. julia --threads=8 rwm_simulation.jl).
    Threads.@threads for i in 1:n_simulations
        # Each thread gets its own RNG.
        # (For reproducibility you might use a seed based on i.)
        rng = MersenneTwister(1234 + i)
        chain, sum_abs, sum_indicator = rwm_t4_like_normal(N, 0.0, proposal_scale, burnin, rng)
        ergodic_averages_abs[i] = sum_abs / (N - burnin)
        ergodic_averages_indicator[i] = sum_indicator / (N - burnin)

        # Save the last 5 samples of the first chain (if i == 1)
        if i == 1
            last5_samples_chain_1 .= chain[end-4:end]
        end
    end

    elapsed = time() - start_time

    println("Last 5 samples of the first RWM chain:")
    println(last5_samples_chain_1)
    println("Last 5 ergodic averages for |x| (from the last 5 chains):")
    println(ergodic_averages_abs[end-4:end])
    println("Last 5 ergodic averages for indicator 1(|x| >= 2) (from the last 5 chains):")
    println(ergodic_averages_indicator[end-4:end])
    println("Total elapsed time: $(elapsed) seconds.")

    # Write the ergodic averages to CSV files.
scriptdir = @__DIR__

writedlm(joinpath(scriptdir,
                  "ergodic_average_abs_v=1_Cauch.csv"),
         ergodic_averages_abs, ',')

writedlm(joinpath(scriptdir,
                  "ergodic_average_indicator_v=1_Cauch.csv"),
         ergodic_averages_indicator, ',')
    println("CSV files written.")
end

main()
