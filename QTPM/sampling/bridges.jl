#=
    Deals with the calculation and generation of random bridges of a spin starting
    at state si, and finishing at state sf in some time t. It contains the option
    to have a bias in the magnetization given by some biasing strength s.
=#

"""
    partition(J::Real, t::Real, i::Bool, f::Bool)

Determine the log partition function for a bridge from i to f in time t, 
and z-magnetization J.
"""
function partition(J::Real, t::Real, i::Bool, f::Bool)
    if i == true && f == true
        Z = t * sqrt(1+J^2) - log(2)
        Z += log(1 + (J / sqrt(1 + J^2)) + exp(-2*t*sqrt(1 + J^2)) * (1 - (J / sqrt(1 + J^2))))
    elseif i == false && f == false
        Z = t * sqrt(1+J^2) - log(2)
        Z += log(1 - (J / sqrt(1 + J^2)) + exp(-2*t*sqrt(1 + J^2)) * (1 + (J / sqrt(1 + J^2))))
    else
        Z = t * sqrt(1+J^2) - log(2) - log(sqrt(1 + J^2))
        Z += log(1 -  exp(-2*t*sqrt(1 + J^2)))
    end

    return Z
end


"""
    survival(J::Real, t::Real, tau::Real, i::Bool, f::Bool)

Determine the survival probability at time tau from some initial state i and final state f,
with the z-magnetization J and total time t.
"""
function survival(J::Real, t::Real, tau::Real, i::Bool, f::Bool)
    return exp(partition(J, t-tau, i, f) - partition(J, t, i, f) + tau * (i ? J : -J))
end


"""
    survival_time(J::Real, t::Real, i::Bool, f::Bool)   

Calculate the survival time for system with magnetization J, time t,
initial state i and final state f.
"""
function survival_time(J::Real, t::Real, i::Bool, f::Bool)
    # Generate a random number which determines the time 
    r = rand(Float64)

    # Initiate the limits on time
    lower = 0
    upper = t

    # Calculate the partition sum for the complete time 
    Z = partition(J, t, i, f)

    # Check to see if the state survives 
    if i == f
        S = exp(partition(J, 0, i, f) - Z +  t * (i ? J : -J))
        S > r && return t + 1e-5
    end

    # Bisect until we find the target value 
    tau = 0.5*(upper+lower)
    val = exp(partition(J, t-tau, i, f) - Z + tau * (i ? J : -J))
    while abs(val - r) > 1e-8 || isnan(val)
        if val > r && !isnan(val)
            lower = 0.5*(upper+lower)
        else
            upper = 0.5*(upper+lower)
        end
        tau = 0.5*(upper+lower)
        val = exp(partition(J, t-tau, i, f) - Z + tau * (i ? J : -J))
    end

    return tau
end


"""
    bridge(J::Real, t::Real, i::Bool, f::Bool)

Sample a bridge from initial state i to final state f, with time t and coupling J.
"""
function bridge(J::Real, t::Real, i::Bool, f::Bool)
    # Initiate time
    time = 0.0
    times = Float64[]

    # Find all transition times 
    while time < t 
        # Determine survival time
        dt = survival_time(J, t-time, i, f)

        # Update system
        if time + dt < t
            # Store jump
            time += dt
            push!(times, time)
            i = !i
        else
            # Survive 
            time += dt
        end
    end
    return times
end