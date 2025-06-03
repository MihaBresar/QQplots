using Random
using Distributions           # <— still needed
using DelimitedFiles
using Statistics
using Dates

# ---------------------------
# Target density (unnormalised)
# ---------------------------
@inline function unnormalized_t4_like_pdf(x)
    tmp = 1.0 + (abs(x)^(4.5)) / 9    #  ~ (1 + x²/4)⁻¹
    return 1.0 / tmp
end

# ---------------------------
# Random–Walk Metropolis with  
# Student-t(ν=0.1) proposals
# ---------------------------
function rwm_t4_like_normal(n_iter::Int, initial::Float64,
                            proposal_scale::Float64, burnin::Int,
                            rng::AbstractRNG)

    chain = Vector{Float64}(undef, n_iter)
    chain[1] = initial
    p_current = unnormalized_t4_like_pdf(chain[1])

    # Heavy-tailed proposal: ν = 0.1   (tail ~ r^{-0.1})
    proposal_dist = TDist(0.1)

    ergodic_sum_abs       = 0.0
    ergodic_sum_indicator = 0.0

    for i in 2:n_iter
        # propose a move
        step   = rand(rng, proposal_dist) * proposal_scale
        x_prop = chain[i-1] + step

        p_prop = unnormalized_t4_like_pdf(x_prop)
        α      = p_prop / p_current               # since target is un-normalised

        if rand(rng) < α
            chain[i]  = x_prop
            p_current = p_prop
        else
            chain[i] = chain[i-1]
        end

        if i > burnin
            abs_val = abs(chain[i])
            ergodic_sum_abs       += abs_val
            ergodic_sum_indicator += (abs_val ≥ 2.0) ? 1.0 : 0.0
        end
    end

    return chain, ergodic_sum_abs, ergodic_sum_indicator
end

# ---------------------------
# Driver
# ---------------------------
function main()
    # Simulation parameters
    N               = 2_000_000     # samples / chain
    burnin          = 500_000
    n_simulations   = 10_000
    proposal_scale  = 5.0           # keep the same scale for comparability

    erg_abs        = zeros(Float64, n_simulations)
    erg_indicator  = zeros(Float64, n_simulations)
    last5_chain1   = zeros(Float64, 5)

    start_time = time()

    Threads.@threads for i in 1:n_simulations
        rng   = MersenneTwister(1234 + i)
        ch, s_abs, s_ind = rwm_t4_like_normal(N, 0.0, proposal_scale, burnin, rng)

        erg_abs[i]       = s_abs / (N - burnin)
        erg_indicator[i] = s_ind / (N - burnin)

        if i == 1
            last5_chain1 .= ch[end-4:end]
        end
    end

    elapsed = time() - start_time

    println("Last 5 samples of the first RWM chain:")
    println(last5_chain1)
    println("Last 5 ergodic averages for |x| (chains $(n_simulations-4):$n_simulations):")
    println(erg_abs[end-4:end])
    println("Last 5 ergodic averages for indicator 1(|x| ≥ 2):")
    println(erg_indicator[end-4:end])
    println("Total elapsed time: $(elapsed) seconds.")

    # --- CSV output ---------------------------------------------------------
    scriptdir = @__DIR__

    writedlm(joinpath(scriptdir,
                      "ergodic_average_abs_v=3_0.1Student.csv"),
             erg_abs, ',')

    writedlm(joinpath(scriptdir,
                      "ergodic_average_indicator_v=3_0.1Student.csv"),
             erg_indicator, ',')

    println("CSV files written.")
end

main()
