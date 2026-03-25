using LinearAlgebra
using Plots

μ = [4.0, 8.0, 8.0]

Σ = [2.0 1.0 0.0;
    1.0 2.0 1.0;
    0.0 1.0 2.0]

n = length(μ)
N = 100000

means = zeros(N)
stds = zeros(N)

## a) Plotting the feasible set of portfolios
for i in 1:N
    w = rand(n)
    w = w/sum(w)

    means[i] = μ' * w
    stds[i] = sqrt(w' * Σ * w)
end

scatter(stds, means,
    xlabel = "Standard Deviation",
    ylabel = "Expected Return",
    title = "Feasible set of portfolios",
    color = :lightblue,
    markerstrokewidth = 0,
    legend = false)


## b) Plotting the efficient frontier

# Minimum Variance Portfolio
e = ones(n)
w_min = inv(Σ) * e / (e' * inv(Σ) * e)

μ_min = μ' * w_min
σ_min = sqrt(w_min' * Σ * w_min)

# Another Efficient Portfolio
w_max = [0, 0.5, 0.5]

μ_max = μ' * w_max
σ_max = sqrt(w_max' * Σ * w_max)

# Two fund theorem
α_vals = range(0, 1, length = 100)

frontier_std = Float64[]
frontier_mean = Float64[]

for α in α_vals
    w = α * w_min + (1-α) * w_max

    push!(frontier_std, sqrt(w' * Σ * w))
    push!(frontier_mean, μ' * w)
end

plot!(frontier_std, frontier_mean,
    label = "Efficient Frontier",
    linewidth = 5,
    color = :maroon)


## c) Indicating the global minimum variance portfolio

scatter!([σ_min], [μ_min],
    label = "Global Minimum Variance Portfolio",
    color = :yellow,
    markersize = 5)

## d) Plotting the capital allocation line

rf = 2.0

w_tangent = (inv(Σ)*(μ - rf*e))/(e' * inv(Σ) * (μ - rf*e))
μ_tangent = μ' * w_tangent
σ_tangent = sqrt(w_tangent' * Σ * w_tangent)

α_vals = range(0, 1.2, length=200)

cal_std = α_vals .* σ_tangent
cal_mean = rf .+ α_vals .* (μ_tangent - rf)

plot!(cal_std, cal_mean,
    label = "Capital Allocation Line",
    linewidth = 2,
    color = :black)

scatter!([σ_tangent], [μ_tangent],
    label = "Tangency Portfolio",
    color = :pink,
    markersize = 5)


    