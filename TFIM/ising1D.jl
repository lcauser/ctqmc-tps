include("sampling/bridges.jl")
include("sampling/spin.jl")
include("sampling/spins.jl")
include("sampling/interactions.jl")
using HDF5

# System parameters
N = 128
beta = 128
J = 0.98
g = 0
save_file = "observables.h5"

# Simulation parameters
num_sims = N * 10^5
checkpoint = Int(ceil(num_sims / 1000))

function main(N, beta, J, g, num_sims, checkpoint, filesave=false)
    # Simulate an initial trajectory
    mag_bias = abs(J) >= 1 ? -5.0 : 0.0 # Bias so the seed trajectory is in the correct phase
    traj = simulateSpinsPeriodic(N, beta, mag_bias)
    if J < -1.0
        for i = 2:2:N
            traj.spins[i].initial = !traj.spins[i].initial
        end
    end

    # Create ising bonds and measure observables
    bonds = Ising1D(N, J, g)
    magZs = magnetization(traj)
    magXs = [length(traj.spins[i].times) for i = 1:N]
    magZZs = zeros(Float64, N)
    for idx = 2:2:N
        magZZs[idx-1:idx] = calculateInteraction(bonds, traj, idx)[1:2]
    end

    # Updates
    Z = 0
    X = 0
    ZZ = 0
    Z_abs = 0
    for i = 1:num_sims
        # Update the trajectory
        idx = update_periodic!(traj, bonds)

        # Calculate observables
        magXs[idx] = length(traj.spins[idx].times)
        inters = calculateInteraction(bonds, traj, idx)
        magZZs[[idx - 1 == 0 ? N : idx - 1, idx]] = inters[1:2]
        magZs[idx] = inters[3]
        Z += sum(magZs)
        X += sum(magXs)
        ZZ += sum(magZZs)

        # Absolute magnetization is highly correlated and expensive; only calculate every N updates
        if i % N == 0
            Z_abs += absolute_magnetization(traj)
        end

        # Save the averages
        if i % checkpoint == 0
            println(string(i, "/", num_sims, " X=", round(X / (N*beta*i), digits=5), " ZZ=", round(ZZ / (N*beta*i), digits=5), " Z=", round(Z_abs / (beta*i), digits=5)))
            
            if filesave != false
                f = h5open(filesave, "w")
                write(f, "x", X / (N*beta*i))
                write(f, "zz", ZZ / (N*beta*i))
                write(f, "z_abs", Z_abs / (beta*i))
                close(f)
            end
        end
    end
end

main(N, beta, J, g, num_sims, checkpoint, save_file)