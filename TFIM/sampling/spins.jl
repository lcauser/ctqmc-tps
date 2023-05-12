#= 
    The Spins structure stores trajectories for multiple spins.
    Also contains functions to manipulate, and simulate spin collection trajectories.
=#

"""
    Spins(time::Real, spins::Array{Spin}})

Stores the trajectory for spins in a sytem.
"""
mutable struct Spins{T<:Real, Q<:Spin}
    time::T
    spins::Array{Q}
end

"""
    simulateSpins(N::Int, time::Real, [initials::Bool])

Simulate the trajectory for a spin system. Can pass through the initial states.
"""
function simulateSpins(N::Int, time::Real, initials::Array{Bool})
    spins = Array{Spin}(undef, N)
    for i = 1:N
        spins[i] = simulateSpin(time, initials[i, j])
    end
    return Spins(time, spins)
end
simulateSpins(N::Int, time::Real) = simulateSpins(N, time, collect(rand(Float64, N) .< S.c))


"""
    simulateSpinsPeriodic(N::Int, time::Real, s::Real=0.0)

Simulate the trajectory for a spin system with PBC in time.
"""
function simulateSpinsPeriodic(N::Int, time::Real, s::Real=0.0)
    spins = Array{Spin}(undef, N)
    for i = 1:N
        spins[i] = simulateSpinPeriodic(time, s)
    end
    return Spins(time, spins)
end


"""
    reconstruct(spins::Spins)
    reconstruct(spins::Spins, [idxs::Vector])

Reconstruct a trajectory to give all the system states at all times.
Specify the coordinates of particular spins to retrieve a partial trajectory.
"""
function reconstruct(spins::Spins, idxs::Vector{Int})
    # Find the initial states
    state = [spins.spins[idx].initial for idx in idxs]

    # Find the list of first jump times & number of jumps
    next_times = [length(spins.spins[idx].times) > 0 ? spins.spins[idx].times[1] : spins.time for idx in idxs]
    next_idxs = [1 for _ = 1:length(idxs)]
    num_jumps = sum(length(spins.spins[idx].times) for idx in idxs)
# 
    # Construct the trajectory 
    times = zeros(Float64, num_jumps+1)
    states = zeros(Bool, num_jumps+1, length(idxs))
    states[1, :] = copy(state)
    for i = 1:num_jumps
        #println("----")
        #println(next_idxs)
        #println(next_times)

        # Find the next time
        idx = argmin(next_times)
        times[i+1] = next_times[idx]

        # Update the next times list
        next_idxs[idx] += 1
        next_times[idx] = next_idxs[idx] > length(spins.spins[idxs[idx]].times) ? spins.time : spins.spins[idxs[idx]].times[next_idxs[idx]]

        # Update the state
        state[idx] = 1 - state[idx]
        states[i+1, :] = copy(state)    
    end

    return times, states
end
reconstruct(spins::Spins) = reconstruct(spins, collect(1:length(spins.spins)))


"""
    state(spins::Spins, time::Real)

Find the state of a system of spins at some time.
"""
function state(spins::Spins, time::Real)
    N = length(spins.spins)
    st = zeros(Bool, S.Ny, S.Nx)
    for i = 1:N
        st[i] = state(spins.spins[i], time)
    end
    return st
end


"""
    magnetization(spin::Spins)

Calculate the time-integrated magnetizations of a collection of spins.
"""
function magnetization(spins::Spins)
    N = length(spins.spins)
    Mzs = Array{Float64}(undef, N)
    for i = 1:N
        Mzs[i] = magnetization(spins.spins[i])
    end
    return Mzs
end


"""
    absolute_magnetization(spins::Spins)

Calculate the time-integrated absolute magnetization of a collection of spins.
"""
function absolute_magnetization(spins::Spins)
    # Reconstruct the trajectory
    times, states = reconstruct(spins)

    # Loop through calculating absolute integration
    M_int = 0.0
    for i = 1:length(times)
        # Calculate time interval
        t1 = times[i]
        t2 = i == length(times) ? spins.time : times[i+1]

        # Determine mag 
        M = abs(sum([s ? 1 : -1 for s in states[i, :]]))
        M_int += (t2 - t1) * M
    end

    return M_int
end


"""
    absolute_magnetization_binder(spins::Spins)

Calculate the quantities needed for the Binder cumulant of the magnetization.
"""
function absolute_magnetization_binder(spins::Spins)
    # Reconstruct the trajectory
    N = length(spins.spins)
    times, states = reconstruct(spins)

    # Loop through calculating absolute integration
    M_int = 0.0
    M4_int = 0.0
    M2_int = 0.0
    for i = 1:length(times)
        # Calculate time interval
        t1 = times[i]
        t2 = i == length(times) ? spins.time : times[i+1]

        # Determine mag 
        M = abs(sum([s ? 1 : -1 for s in states[i, :]]))
        M_int += (t2 - t1) * M
        M2_int += (t2 - t1) * (M/N)^2
        M4_int += (t2 - t1) * (M/N)^4
    end

    return M_int, M2_int, M4_int
end