#= 
    The Plaquettes structure contains the struct to store plaquette trajectories,
    and functions to simulate / manipulate them.
=#

using LinearAlgebra

"""
    Plaquettes(Nx::Int, Ny::Int, time::Real, initial::Array{Bool}, times::Array{Vector{Real}}})

Stores the trajectory for a plaquette lattice.
"""
mutable struct Plaquettes{T<:Real}
    Nx::Int 
    Ny::Int
    time::T
    initial::Array{Bool}
    times::Array{Vector{T}}
end

function Plaquettes(Nx::Int, Ny::Int, time::Real, initial::Array{Bool})
    return Plaquettes(Nx, Ny, time, initial, [Float64[] for i = 1:Nx*Ny])
end

function simulatePlaquettes(Nx::Int, Ny::Int, time::Real)
    initial = rand(Bool, Nx*Ny)
    times = [bridge(0, time, false, false) for _ = 1:Nx*Ny]
    return Plaquettes(Nx, Ny, time, initial, times)
end

"""
    plaquettePosition(P::Plaquettes, idx::Int)

Find the x and y coordinates of a plaquette.
"""
function plaquettePosition(P::Plaquettes, idx::Int)
    # Ensure the index is within the range
    idx = idx > (P.Nx * P.Ny) ? idx - (P.Nx * P.Ny) : idx
    idx = idx < 1 ? (P.Nx * P.Ny) + idx : idx

    # Find the coordinates of the plaquette
    ypos = Int(ceil(idx / P.Nx))
    xpos = idx - ((ypos - 1) * P.Nx)

    return xpos, ypos
end

"""
    plaquetteIndex(P::Plaquettes, xpos::Int, ypos::Int)

Calculate the index of a plaquette given its lattice position.
"""
function plaquetteIndex(P::Plaquettes, xpos::Int, ypos::Int)
    # Ensure the positions are within range 
    xpos = xpos > P.Nx ? xpos - P.Nx : xpos
    xpos = xpos < 1 ? P.Nx + xpos : xpos 
    ypos = ypos > P.Ny ? ypos - P.Ny : ypos
    ypos = ypos < 1 ? P.Ny + ypos : ypos 

    # Convert to an index 
    return (ypos - 1) * P.Nx + xpos

end


"""
    spinPlaquetteIndexs(P::Plaquettes, sidx::Int)

Returns the indexs of the plaquettes assoiated with a spin at sidx.
"""
function spinPlaquetteIndexs(P::Plaquettes, sidx::Int)
    # Find the coordinates of the first site
    xpos, ypos = plaquettePosition(P, sidx)

    # Find the other plaquette indexs
    idx2 = plaquetteIndex(P, xpos, ypos+1)
    idx3 = plaquetteIndex(P, xpos+1, ypos+1)

    return sidx, idx2, idx3    
end


"""
    plaquetteSpinIndexs(P::Plaquettes, idx::Int)

Return the indexs of spins contained in a plaquette.
"""
function plaquetteSpinIndexs(P::Plaquettes, idx::Int)
    # Find the coordinates of the plaquette index 
    xpos, ypos = plaquettePosition(P, idx)

    # Find the other spin indexs 
    sidx2 = plaquetteIndex(P, xpos, ypos-1)
    sidx3 = plaquetteIndex(P, xpos-1, ypos-1)

    return idx, sidx2, sidx3
end


"""
    updatePlaquette!(P::Plaquettes, J::Real)

Update a plaquette.
"""
function updatePlaquette!(P::Plaquettes, J::Real)
    # Choose an index 
    i = rand(1:P.Nx*P.Ny)

    # Sample and update trajectory
    pidxs, initial, times = samplePlaquette(P, i, J)
    P.initial[[pidxs...]] = initial 
    P.times[i] = times

    return i, pidxs
end

"""
    samplePlaquette(P::Plaquettes, sidx::Int, J::Real)  

Sample a new trajectory for a plaquette
"""
function samplePlaquette(P::Plaquettes, sidx::Int, J::Real)
    # Find the plaquettes with the spin 
    pidxs = spinPlaquetteIndexs(P, sidx)
    
    # Find the reconstruction of other neighbouring spins flipping 
    times, transitions = reconstructSpin(P, sidx)

    # Create the list of possible plaquettes states and their Z-mags
    states = [[1, 1, 1], [0, 0, 0],
              [0, 1, 1], [1, 0, 0],
              [1, 0, 1], [0, 1, 0],
              [1, 1, 0], [0, 0, 1]]
    Zs = J .* [3, 1, 1, 1]

    # The evolution operator (normalization adjusted to not make too large)
    function evolution(t::Real)
        U = zeros(Float64, 8, 8)

        # First matrix 
        Jprime = sqrt(1+Zs[1]^2)
        ex = exp(-2*t*Jprime)
        U[1, 1] = 0.5 * (1 + (Zs[1] / Jprime) + ex * (1 - (Zs[1] / Jprime)))
        U[2, 2] = 0.5 * (1 - (Zs[1] / Jprime) + ex * (1 + (Zs[1] / Jprime)))
        U[1, 2] = (0.5 / Jprime) * (1 - ex)
        U[2, 1] = U[1, 2]

        # Second matrix
        Jprime2 = sqrt(1+Zs[2]^2)
        ex = exp(-2*t*Jprime2)
        ex2 = exp(t*(Jprime2-Jprime))
        U[3, 3] = 0.5 * ex2 * (1 + (Zs[2] / Jprime2) + ex * (1 - (Zs[2] / Jprime2)))
        U[4, 4] = 0.5 * ex2 * (1 - (Zs[2] / Jprime2) + ex * (1 + (Zs[2] / Jprime2)))
        U[3, 4] = (0.5 / Jprime2) * ex2 * (1 - ex)
        U[4, 3] = U[3, 4]

        # Repeat
        U[5:6, 5:6] = U[3:4, 3:4]
        U[7:8, 7:8] = U[3:4, 3:4]
        return U
    end

    # Create the flip matrices
    flips = zeros(Float64, 3, 8, 8)
    idxs = [[3, 4, 1, 2, 8, 7, 6, 5],
            [5, 6, 8, 7, 1, 2, 4, 3],
            [7, 8, 6, 5, 4, 3, 1, 2]]
    for i = 1:3
        for j = 1:8
            flips[i, j, idxs[i][j]] = 1.0
        end
    end
    
    # Create idenity matrix
    Q = diagm(ones(Float64, 8))

    Ps = zeros(Float64, length(times), 8, 8)
    Us = zeros(Float64, length(times)+1, 8, 8)
    # Evolve the matrix 
    for i = 1:length(times)
        # Find evolution matrix
        dt = i == 1 ? times[1] : times[i] - times[i-1]
        U = evolution(dt)

        # Perform the required flip
        idx = findfirst(pidxs .== transitions[i])
        U = flips[idx, :, :] * U
        Us[i, :, :] = U

        # Evolve 
        Q = U * Q
        
        # Normalize
        Q ./= sum(Q)
        Ps[i, :, :] = deepcopy(Q)
    end
    
    # Do the final evolution 
    dt = length(times) > 0 ? P.time - times[end] : P.time
    U = (evolution(dt))
    Us[end, :, :] = U
    Q = U * Q
    Q ./= sum(Q[i, i] for i = 1:8)

    # Sample the initial / final configuration
    configs = zeros(Int, length(times)+2)
    idx = findfirst(cumsum(diag(Q)) .>= rand())
    configs[1] = idx 
    configs[end] = idx

    # Back-sample configurations 
    for i = 1:length(times)
        # Fetch the evolution until the required time
        Q = Ps[end-i+1, :, configs[1]]

        # Fetch the next evolution matrix 
        U = Us[end-i+1, configs[end-i+1], :]

        # Find probabilities 
        Q = U .* Q 
        Q ./= sum(Q)

        # Sample 
        idx = findfirst(cumsum(Q) .>= rand())
        configs[end-i] = deepcopy(idx) 
    end

    # Find the configuration before the forced transitions
    configs_ends = zeros(Int, length(times)+1)
    for i = 1:length(times)
        idx = configs[i+1]
        state = deepcopy(states[idx])
        idx = findfirst(pidxs .== transitions[i])
        state[idx] = 1 - state[idx]
        idx = findfirst([states[j] == state for j = 1:8])
        configs_ends[i] = idx
    end
    configs_ends[end] = configs[end]
    
    
    # Sample bridges
    new_times = Float64[]
    for i = 1:length(configs_ends)
        # Find the J value and times
        J = Zs[Int(ceil(configs[i] / 2))]
        tmin = i == 1 ? 0.0 : times[i-1]
        tmax = i == length(configs_ends) ? P.time : times[i]
        bridge_times = bridge(J, tmax-tmin, isodd(configs[i]), isodd(configs_ends[i]))
        append!(new_times, bridge_times .+ tmin)
    end
    
    return pidxs, states[configs[1]], new_times

end

"""
    reconstructSpin(P::Plaquettes, sidx::Int)

Determine the trajectory for a spin.
"""
function reconstructSpin(P::Plaquettes, sidx::Int)
    # Find the plaquettes with the spin 
    pidxs = spinPlaquetteIndexs(P, sidx)

    # Find the other spins assoicated with the plaquettes 
    sidxs = zeros(Int, 6) # Stores the spin indexs which affect the given spin index
    i = 1
    for pidx in pidxs 
        idxs = plaquetteSpinIndexs(P, pidx)
        for idx in idxs
            if idx != sidx 
                sidxs[i] = idx 
                i += 1
            end
        end
    end
    pidxs = [pidxs[1], pidxs[1], pidxs[2], pidxs[2], pidxs[3], pidxs[3]]

    # Find a list of times when spins flips
    next_times = [length(P.times[idx]) > 0 ? P.times[idx][1] : P.time for idx in sidxs]
    next_idxs = [1 for _ = 1:length(sidxs)]
    num_jumps = sum(length(P.times[idx]) for idx in sidxs)

    # Construct the trajectory 
    times = zeros(Float64, num_jumps)
    states = zeros(Int, num_jumps)
    for i = 1:num_jumps
        # Find the next time 
        idx = argmin(next_times)
        times[i] = next_times[idx]

        # Update the next times list 
        next_idxs[idx] += 1
        next_times[idx] = next_idxs[idx] > length(P.times[sidxs[idx]]) ? P.time : P.times[sidxs[idx]][next_idxs[idx]]

        # Update the state 
        states[i] = pidxs[idx]
    end

    return times, states
end


"""
    reconstructPlaquette(P::Plaquettes, idx::Int)

Determine the trajectory for a plaquette of spins.
"""
function reconstructPlaquette(P::Plaquettes, idx::Int)
    # Find the spins with the plaquette
    sidxs = plaquetteSpinIndexs(P, idx)

    # Find a list of times when spins flips
    next_times = [length(P.times[idx]) > 0 ? P.times[idx][1] : P.time for idx in sidxs]
    next_idxs = [1 for _ = 1:length(sidxs)]
    num_jumps = sum(length(P.times[idx]) for idx in sidxs)

    # Construct the trajectory 
    times = zeros(Float64, num_jumps+1)
    for i = 1:num_jumps
        # Find the next time 
        idx = argmin(next_times)
        times[i+1] = next_times[idx]

        # Update the next times list 
        next_idxs[idx] += 1
        next_times[idx] = next_idxs[idx] > length(P.times[sidxs[idx]]) ? P.time : P.times[sidxs[idx]][next_idxs[idx]]
    end

    return times
end

"""
    magnetization(P::Plaquettes, idx::Int)

Find the magnetization of a plaquette.
"""
function magnetization(P::Plaquettes, idx::Int)
    # Find the reconstruction of times 
    times = reconstructPlaquette(P, idx)

    # Do the time integration 
    Z = 0
    state = deepcopy(P.initial[idx])
    for i = 1:length(times)
        tmin = times[i]
        tmax = i == length(times) ? P.time : times[i+1]
        Z += (tmax - tmin) * (2*state - 1)
        state = !state
    end
    return Z
end