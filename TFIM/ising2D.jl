include("sampling/bridges.jl")
include("sampling/spin.jl")
include("sampling/spins.jl")
include("sampling/interactions.jl")
using HDF5

# System parameters
N = 4
J = 1.0
beta = 128.0
Jc = 1 / 3.04438 # critical
save_file = "observables.h5"

# Simulation parameters
num_sims = Int(ceil(10^5 * N^2))
checkpoint = Int(ceil(num_sims / 1000))
mag_bias = -5.0

function main(N, beta, J, num_sims, checkpoint, filesave=false)
    # Simulate a trajectory
    mag_bias = abs(J) >= Jc ? -5.0 : 0.0
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
        idx = update_periodic!(traj, bonds)
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

main(N, beta, J,  num_sims, checkpoint, save_file)