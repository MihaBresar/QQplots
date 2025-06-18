using Random, LinearAlgebra, Statistics, DelimitedFiles, StaticArrays


# -----------------------------------------------------------------------------
# Unnormalised heavy-tailed target: π(x) ∝ 1 / (1 + ||x||^{5/4}), x ∈ ℝ⁴
# -----------------------------------------------------------------------------
@inline function unnormalized_heavy_pdf(x::SVector{4,Float64})
    return 1.0 / (1.0 + norm(x)^6)
end

# -----------------------------------------------------------------------------
# Stereographic projection ℝ⁴ ↔ S⁴ ⊂ ℝ⁵
# -----------------------------------------------------------------------------
@inline function stereographic_inverse_4d(x::SVector{4,Float64})
    denom = 1.0 + dot(x, x)
    y = 2.0 * x / denom
    z = (dot(x, x) - 1.0) / denom
    return SVector{5,Float64}(y..., z)
end

@inline function stereographic_projection_4d(s::SVector{5,Float64})
    y = SVector{4,Float64}(s[1:4])
    z = s[5]
    return y / (1.0 - z)
end

# -----------------------------------------------------------------------------
# SPS for 4D heavy-tailed target
# -----------------------------------------------------------------------------
function sps_heavy_4d(n_iter::Int, initial::SVector{4,Float64}, step_size::Float64,
                      burnin::Int, rng::AbstractRNG)

    chain = Vector{SVector{4,Float64}}(undef, n_iter)
    chain[1] = initial
    p_curr = unnormalized_heavy_pdf(initial)

    ergodic_sum_norm = 0.0
    ergodic_sum_x1 = 0.0
    ergodic_sum_indicator = 0.0
    n_accept = 0

    for i in 2:n_iter
        s = stereographic_inverse_4d(chain[i-1])

        ξ = step_size * randn(rng, 5)
        dotprod = dot(s, ξ)
        z_prime = ξ - dotprod * s

        s_prop = s + z_prime
        s_prop = s_prop / norm(s_prop)

        x_prop = stereographic_projection_4d(s_prop)

        p_prop = unnormalized_heavy_pdf(x_prop)
        num = p_prop * (1 + dot(x_prop, x_prop))
        den = p_curr * (1 + dot(chain[i-1], chain[i-1]))
        α = min(1.0, num / den)

        if rand(rng) < α
            chain[i] = x_prop
            p_curr = p_prop
            n_accept += 1
        else
            chain[i] = chain[i-1]
        end

        if i > burnin
            x = chain[i]
            ergodic_sum_norm += norm(x)
            ergodic_sum_x1  += x[1]
            ergodic_sum_indicator += x[1] ≥ 1.0 ? 1.0 : 0.0
        end
    end

    T = n_iter - burnin
    accept_rate = n_accept / (n_iter - 1)
    return chain, ergodic_sum_norm / T, ergodic_sum_x1 / T, ergodic_sum_indicator / T, accept_rate
end


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function main()
    N = 1000_000
    burnin = 300_000
    n_chains = 7000
    step_size = 0.15

    erg_norm = zeros(n_chains)
    erg_x1   = zeros(n_chains)
    erg_ind  = zeros(n_chains)
    accept_rates = zeros(n_chains)

    Threads.@threads for i in 1:n_chains
        rng = MersenneTwister(1000 + i)
        init = @SVector randn(rng, 4)
        chain, avg_norm, avg_x1, avg_ind, acc = sps_heavy_4d(N, init, step_size, burnin, rng)
        erg_norm[i] = avg_norm
        erg_x1[i]   = avg_x1
        erg_ind[i]  = avg_ind
        accept_rates[i] = acc
    end

    scriptdir = @__DIR__
    writedlm(joinpath(scriptdir, "ergodic_avg_norm_SPS.csv"), erg_norm, ',')
    writedlm(joinpath(scriptdir, "ergodic_avg_x1_SPS.csv"),   erg_x1, ',')
    writedlm(joinpath(scriptdir, "ergodic_avg_indicator_SPS.csv"), erg_ind, ',')
    writedlm(joinpath(scriptdir, "acceptance_rates_SPS.csv"), accept_rates, ',')

    println("Means over all chains:")
    println("⟨‖x‖⟩ = ", mean(erg_norm))
    println("⟨x₁⟩   = ", mean(erg_x1))
    println("⟨1(x₁ ≥ 1)⟩ = ", mean(erg_ind))
    println("Acceptance rate = ", round(mean(accept_rates)*100, digits=2), "%")
end


main()
