include("sampling/plaquettes.jl")
include("sampling/bridges.jl")
using HDF5

# System properties
N = 4
beta = 128.0
J = 1.0
num_sims = 10^5
checkpoint = 10^2
filesave = "observables.h5"

# Warm up properties through thermal annealing
initial_beta = 0.1
increment_small = 0.1
increment_large = 1.0
increment_change = 10.0
sims_increment = 1

function main()
    # Warm up through thermal annealing
    Nx = N 
    Ny = N
    traj = simulatePlaquettes(Nx, Ny, initial_beta)
    while traj.time <= beta
        for _ = 1:sims_increment*Nx*Ny
            idx, pidxs = updatePlaquette!(traj, J)
        end

        println(string("beta=", round(traj.time, digits=5)))
        traj.time == beta && break

        # Increase beta 
        inc = (traj.time +1e-5) >= increment_change ? increment_large : increment_small
        traj.times .*= (traj.time + inc) / (traj.time)
        traj.time += inc
    end

    # Measure observables
    Zs = [magnetization(traj, i) for i = 1:Nx*Ny]
    Xs = [length(traj.times[i]) for i = 1:Nx*Ny]

    # Updates
    X = 0
    Z = 0
    for i = 1:num_sims * Nx * Ny
        # Update plaquette
        idx, pidxs = updatePlaquette!(traj, J)

        # Measure observables
        Xs[idx] = length(traj.times[idx])
        Zs[pidxs[1]] = magnetization(traj, pidxs[1])
        Zs[pidxs[2]] = magnetization(traj, pidxs[2])
        Zs[pidxs[3]] = magnetization(traj, pidxs[3])
        X += sum(Xs) / (Nx * Ny * beta)
        Z += sum(Zs) / (Nx * Ny * beta)
        
        # Save
        if i % (checkpoint * Nx * Ny) == 0
            println(string("sim=", i, "/", num_sims*Nx*Ny, ", Z=", round(Z / i, digits=5), ", X=", round(X / i, digits=5)))
            f = h5open(filesave, "w")
            write(f, "X", X / (N^2*beta*i))
            write(f, "Z", Z / (N^2*beta*i))
            write(f, "num_sims", i)
            close(f)
        end
    end
end

main()