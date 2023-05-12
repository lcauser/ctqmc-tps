include("sampling/bridges.jl")
include("sampling/spin.jl")
include("sampling/spins.jl")
include("sampling/interactions.jl")
using HDF5

# Effective constraint for the abolsute magnetization
function effectiveConstraint(I::Interactions, spins::Spins, s::Float64, Jval::Float64, idx::Int)
    # Find the ising bonds which interact with ising
    idxs = interactionSpins(I, spins, idx)

    # Reconstruct the trajectory for all spins
    idxs_all = collect(1:length(spins.spins))
    deleteat!(idxs_all, idx)
    times, states = reconstruct(spins, idxs_all)

    # Convert to magnetizations
    mzs = 2 .* states .- 1

    # Find the effective J values
    Js = zeros(Float64, length(times))
    for i = 1:length(times)
        J = 0.0

        # Absolute magnetizion
        sum_mz = sum(mzs[i, :]) 
        if sum_mz > 0
            J += s
        elseif sum_mz < 0
            J -= s 
        end

        # Ising bonds
        for j = idxs[2:end]
            J += -Jval .* mzs[i, j > idx ? j - 1 : j]
        end

         Js[i] = J
    end
    return times, Js
end

# Update the trajectory
function update_periodic!(spins::Spins, I::Interactions, s::Float64, J::Float64, idx::Int)
    # Find the effective constraints and update the spin
    times, Js = effectiveConstraint(I, spins, s, J, idx)

    # Update the spin 
    update_periodic!(spins.spins[idx], times, Js)

    return idx
end
update_periodic!(spins::Spins, I::Interactions, s::Real, J::Real) = update_periodic!(spins, I, s, J, rand(1:length(spins.spins)))


# System parameters
N = 4
beta = 64.0
J = 1.0
s = 1.0
Jc = 1 / 3.04438
save_file = "observables.h5"

# Simulation parameters
num_sims = Int(ceil(10^5 * N^2))
checkpoint = N^2
mag_bias = -5.0


function main(N, beta, J, s, num_sims, checkpoint, filesave=false)
    # Simulate a trajectory
    mag_bias = s <= 0.0 ? -5.0 : 0.0
    traj = simulateSpinsPeriodic(N^2, beta, mag_bias)
    if J < -Jc
        for idx1 = 1:N
            idxs = isodd(idx1) ? (1:2:N-1) : (2:2:N)
            for idx2 = idxs
                idx = (idx1-1) * N + idx2
                traj.spins[idx].initial = !traj.spins[idx].initial
            end
        end
    end

    # Create ising bonds and measure observables
    bonds = Ising2D(N, N, J, 0.0)
    magZs = magnetization(traj)
    magXs = [length(traj.spins[i].times) for i = 1:N^2]
    magZZs = zeros(Float64, 2, N, N)
    for idx1 = 1:N
        idxs = isodd(idx1) ? (1:2:N-1) : (2:2:N)
        for idx2 = idxs
            idx = (idx1-1) * N + idx2
            measurements = calculateInteraction(bonds, traj, idx)
            magZZs[1, idx1, idx2 - 1 == 0 ? N : idx2 - 1] = measurements[1]
            magZZs[1, idx1, idx2] = measurements[2]
            magZZs[2, idx1, idx2 - 1 == 0 ? N : idx2 - 1] = measurements[3]
            magZZs[2, idx1, idx2] = measurements[4]
        end
    end

    # Do updates
    Z = 0
    X = 0
    ZZ = 0
    Z_abs = 0
    for i = 1:num_sims
        idx = update_periodic!(traj, bonds, s, J)
        magZs[idx] = magnetization(traj.spins[idx])
        magXs[idx] = length(traj.spins[idx].times)

        idx1 = Int(ceil(idx / N))
        idx2 = idx - (idx1-1)*N
        measurements = calculateInteraction(bonds, traj, idx)
        magZZs[1, idx1, idx2 - 1 == 0 ? N : idx2 - 1] = measurements[1]
        magZZs[1, idx1, idx2] = measurements[2]
        magZZs[2, idx1, idx2 - 1 == 0 ? N : idx2 - 1] = measurements[3]
        magZZs[2, idx1, idx2] = measurements[4]
        Z += sum(magZs)
        X += sum(magXs)
        ZZ += sum(magZZs)

        # Absolute magnetization is highly correlated and expensive; only calculate every N updates
        if i % N^2 == 0
            Z_abs += absolute_magnetization(traj)
        end

        # Save the averages
        if i % checkpoint == 0
            println(string(i, "/", num_sims, " X=", round(X / (N^2*beta*i), digits=5), " ZZ=", round(ZZ / (N^2*beta*i), digits=5), " Z=", round(Z_abs / (beta*i), digits=5)))
            
            if filesave != false
                f = h5open(filesave, "w")
                write(f, "x", X / (N^2*beta*i))
                write(f, "zz", ZZ / (N^2*beta*i))
                write(f, "z_abs", Z_abs / (beta*i))
                write(f, "num_sims", i)
                close(f)
            end
        end
    end
end

main(N, beta, J, s, num_sims, checkpoint, save_file)