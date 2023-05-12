#= 
    The Spin structure stores trajectories for individual spins.
    Also contains functions to manipulate, and simulate spin trajectories.
=#

"""
    Spin(initial::Bool, time::Real, times::Vector{Real})

Stores the trajectory for an individual spin in the system
"""
mutable struct Spin{Q<:Real,T<:Real}
    initial::Bool
    time::Q
    times::Vector{T}
end


"""
    simulateSpin(time::Real, [initial::Bool])

Simulate the trajectory for a spin. Can pass through the initial state.
"""
function simulateSpin(time, initial)
    t = 0.0 # Store the trajectory time 
    state = copy(initial)
    times = Float64[]
    while t < time
        # Find jump time 
        transitionTime = rate == 0.0 ? time : -log(rand(Float64))
        t += transitionTime 

        # If not past maximum time, add to trajectory
        if t < time
            push!(times, t) # Add to times
            state = !state # Change the spin
        end
    end

    return Spin(initial, time, times)
end

"""
    simulateSpinPeriodic(time, s=0.0)

Simulate the trajectory for a spin with PBC in time.
"""
function simulateSpinPeriodic(time, s=0.0)
    # Guess an initial state
    sbar = sqrt(1 + s^2)
    P1 = 1 / (2*(1 + s^2 + s*sbar))
    P1 += exp((-2*sbar)*time) / (2*(1 + s^2 - s*sbar))
    P2 = (s+sbar)^2 / (2*(1 + s^2 + s*sbar))
    P2 += exp((-2*sbar)*time) * (s-sbar)^2 / (2*(1 + s^2 - s*sbar))

    initial = rand(Float64) < (P1 / (P1 + P2))
    times = bridge(s, time, initial, initial)
    return Spin(initial, time, times)
end


"""
    state(spin::Spin, time::Real)

Calculate the state of a spin at some time.
"""
function state(spin::Spin, time)
    if length(spin.times) == 0
        return spin.initial
    elseif time < spin.times[1]
        return spin.initial
    else
        idx = findlast([t < time for t in spin.times])
        return idx % 2 == 1 ? !spin.initial : spin.initial
    end
end


"""
    reconstruct(spin::Spin, [tstart::Real, tend::Real])

Reconstruct a spin trajectory within the time.
"""
function reconstruct(spin::Spin, tstart, tend)
    # Find the initial state
    st = state(spin, tstart)

    # Find the number of flips
    if length(spin.times) == 0
        return [tstart], [st]
    elseif spin.times[end] < tstart
        return [tstart], [st]
    else
        idx1 = findfirst([t > tstart for t in spin.times])
        idx2 = findlast([t < tend for t in spin.times])
        times = zeros(Float64, 2+idx2-idx1)
        times[1] = tstart
        states = zeros(Bool, 2+idx2-idx1)
        states[1] = copy(st)
        for i = idx1:idx2
            st = !st 
            times[2 + i - idx1] = spin.times[i]
            states[2 + i - idx1] = st
        end
        return times, states
    end
end
reconstruct(spin::Spin) = reconstruct(spin::Spin, 0.0, spin.time)


"""
    update!(spin::Spin, s, times, Js, mag_bias=0.0)

Resample the trajectory for a spin. Pass throughing biasing dynamics s, and the 
cumulative magnetization of neighbouring spins, Js. Allows for an onsite potential,
mag_bias.
"""
function update!(spin::Spin, s, times, Js, mag_bias=0.0)
    # Keep track of probability
    #Prob = 1

    # Find the probability vector & time evol operators throughout the updates
    P = [0.5, 0.5]
    Ps = [deepcopy(P)]
    Us = []
    for i = 1:length(times)
        # Magnetization bias 
        s2 = s * (Js[i]) + mag_bias
        sbar = sqrt(1 + s2^2)

        # Find the time step
        t2 = i == length(times) ? spin.time : times[i+1]
        t1 = i == 1 ? 0 : times[i]
        dt = t2 - t1

        # Find evolution operator 
        U = zeros(Float64, 2, 2)
        e1 = exp((-1 + sbar)*dt)
        e2 = exp((-1 - sbar)*dt)
        s3 = (2*(1 + s2^2 + s2*sbar)) 
        s4 = (2*(1 + s2^2 - s2*sbar))
        U[1, 1] = e1 / s3
        U[1, 1] += e2 / s4
        U[2, 1] = e1* (s2+sbar) / s3
        U[2, 1] += e2 * (s2-sbar)  / s4
        U[1, 2] = U[2, 1]
        U[2, 2] = e1 * (s2+sbar)^2 / s3
        U[2, 2] += e2 * (s2-sbar)^2  / s4
        push!(Us, U)

        # Evolve in time
        P = U * P
        P = P / (P[1] + P[2])
        push!(Ps, deepcopy(P))
    end

    # Find the final state
    state = rand(Float64) < (Ps[end][1] / (Ps[end][1] + Ps[end][2]))
    states = [state]

    # Find the states after time steps in reverse order
    for i = 1:length(times)
        P1 = Us[end+1-i][2-states[end], 1] * Ps[end-i][1]
        P2 = Us[end+1-i][2-states[end], 2] * Ps[end-i][2]
        state = rand(Float64) < (P1 / (P1 + P2))
        push!(states, state)
    end
    reverse!(states)

    # Construct bridges between timesteps and make the new spin
    new_times = []
    for i = 1:length(states)-1
        # Time step
        t2 = i == length(times) ? spin.time : times[i+1]
        t1 = i == 1 ? 0 : times[i]
        
        # Create a bridge
        bridge_times = bridge(s*Js[i] + mag_bias, t2-t1, states[i], states[i+1])
        append!(new_times, bridge_times .+ t1)
    end

    # Update the spin
    spin.times = new_times
    spin.initial = states[1]
end


"""
    update_periodic!(spin::Spin, times, Js)

Resample the trajectory for a spin. Pass throughing biasing dynamics s, and the 
cumulative magnetization of neighbouring spins, Js. Trajectory will have PBC in time.
"""
function update_periodic!(spin::Spin, times, Js)
    # Find time evolution operators
    Us = []
    P = [1.0 0.0; 0.0 1.0]
    Ps = [deepcopy(P)]
    for i = 1:length(times)
        # Magnetization bias 
        s2 = Js[i]
        sbar = sqrt(1 + s2^2)

        # Find the time step
        t2 = i == length(times) ? spin.time : times[i+1]
        t1 = i == 1 ? 0 : times[i]
        dt = t2 - t1

        # Find evolution operator 
        U = zeros(Float64, 2, 2)
        e1 = 1
        e2 = exp((-2 * sbar)*dt)
        s3 = (2*(1 + s2^2 + s2*sbar)) 
        s4 = (2*(1 + s2^2 - s2*sbar))
        U[1, 1] = e1 / s3
        U[1, 1] += e2 / s4
        U[2, 1] = e1* (s2+sbar) / s3
        U[2, 1] += e2 * (s2-sbar)  / s4
        U[1, 2] = U[2, 1]
        U[2, 2] = e1 * (s2+sbar)^2 / s3
        U[2, 2] += e2 * (s2-sbar)^2  / s4
        push!(Us, U)

        # Evolve in time
        P = U * P
        P = P ./ (P[1, 1] + P[2, 2])
        push!(Ps, deepcopy(P))
    end

    # Find the boundary
    state = rand(Float64) < (Ps[end][1, 1] / (Ps[end][1, 1] + Ps[end][2, 2]))
    states = [state]

    # Find the states after time steps in reverse order
    for i = 1:length(times)
        P1 = Us[end+1-i][2-states[end], 1] * Ps[end-i][1, 2-states[1]]
        P2 = Us[end+1-i][2-states[end], 2] * Ps[end-i][2, 2-states[1]]
        state = rand(Float64) < (P1 / (P1 + P2))
        push!(states, state)
    end
    reverse!(states)

    # Construct bridges between timesteps and make the new spin
    new_times = []
    for i = 1:length(states)-1
        # Time step
        t2 = i == length(times) ? spin.time : times[i+1]
        t1 = i == 1 ? 0 : times[i]
        
        # Create a bridge
        bridge_times = bridge(Js[i], t2-t1, states[i], states[i+1])
        append!(new_times, bridge_times .+ t1)
    end

    # Update the spin
    spin.times = new_times
    spin.initial = states[1]
end


"""
    magnetization(spin::Spin)

Calculate the time-integrated magnetization of a spin.
"""
function magnetization(spin::Spin)
    # Find the initial magnetization 
    mz = spin.initial ? 1 : -1

    integral = 0
    # Time integrate 
    for i = 1:length(spin.times)+1
        # Find the timestep
        t1 = i == 1 ? 0.0 : spin.times[i-1]
        t2 = i == length(spin.times) + 1 ? spin.time : spin.times[i]

        # Integrate
        integral += (t2 - t1) * mz

        # Flip spin 
        mz = -1 * mz
    end
    return integral
end