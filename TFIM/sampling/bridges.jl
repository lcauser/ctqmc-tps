#=
    Deals with the calculation and generation of random bridges of a spin starting
    at state si, and finishing at state sf in some time t. It contains the option
    to have a bias in the magnetization given by some biasing strength s.
=#

"""
    partition(s::Real, t::Real, si::Bool, sf::Bool)

Determine the partition function for a bridge from si to sf in time t, 
and magnetization bias s.
"""
function partition(s::Real, t::Real, si::Bool, sf::Bool)
    sbar = sqrt(1 + s^2)
    R1 = exp(sbar * t) * (s + sbar)^(2-si-sf) * (1 + s^2 - s*sbar)
    R2 = exp(-sbar * t) * (s - sbar)^(2-si-sf) * (1 + s^2 + s*sbar)
    R3 = 2*(1+s^2)
    R4 = exp(-t)
    return (R1 + R2) * R4 / R3
end


"""
    escape_rate(s::Real, t::Real, si::Bool, sf::Bool)    

Determine the escape rate of a spin in state si, which must finish in state sf
after time t with magnetization bias s.
"""
function escape_rate(s::Real, t::Real, si::Bool, sf::Bool)
    sbar = sqrt(1 + s^2)
    R1 = ((s+sbar)^(1+si-sf)) * (1+s^2-s*sbar)
    R2 = exp(-2*sbar*t) * ((s-sbar)^(1+si-sf))*(1+s^2+s*sbar)
    R3 = ((s+sbar)^(2-si-sf))*(1+s^2-s*sbar)
    R4 = exp(-2*sbar*t) * ((s-sbar)^(2-si-sf))*(1+s^2+s*sbar)
    return (R1 + R2) / (R3 + R4)
end


"""
    survival(s::Real, t::Real, tau::Real, si::Bool, sf::Bool)

Determine the survival probability after some time tau for a spin in state si
which must finish in sf after time t, with magnetization bias s.
"""
function survival(s::Real, t::Real, tau::Real, si::Bool, sf::Bool)
    # It cannot survive if it has to change state 
    if t == tau && si != sf
        return 0
    end

    # Calculate survival probability
    sbar = sqrt(1 + s^2)
    a = (s + sbar)^(2-si-sf) * (1 + s^2 -s*sbar)
    b = (s - sbar)^(2-si-sf) * (1 + s^2 +s*sbar)
    A = (2*sbar*t)
    B = (2*sbar*(t-tau))
    F = A - B + log((a + b*exp(-A))/(a + b*exp(-B)))
    S1 = -((s-sbar)^(2*si-1) - (s+sbar)^(2*si-1)) / (2*sbar)
    S2 = (s-sbar)^(2*si-1) * tau
    return exp(-(F*S1 + S2))
end


"""
    survival_time(s::Real, t::Real, si::Bool, sf::Bool)

Determine a random survival time for a spin in state si which must finish in sf
after time t, with magnetization bias s.
"""
function survival_time(s::Real, t::Real, si::Bool, sf::Bool)
    # Generate random number
    r = rand(Float64)

    # Initiate limits
    lower = 0
    upper = t

    # Check to see if the state survives: return time plus perturbation to ensure no flip happens
    survival(s, t, upper, si, sf) > r && return t + 1e-5

    # Bisect until we find the target value
    val = survival(s, t, 0.5*(upper+lower), si, sf)
    while abs(val - r) > 1e-8 || isnan(val)
         if val > r && !isnan(val)
            lower = 0.5*(upper+lower)
         else
            upper = 0.5*(upper+lower)
         end
         val = survival(s, t, 0.5*(upper+lower), si, sf)
    end
    return 0.5*(upper+lower)
end


"""
    bridge(s::Real, t::Real, si::Bool, sf::Bool)

Generate a random bridge from si to sf in time t, and magnetization bias s.
Returns the transition times and generation probability.
"""
function bridge(s::Real, t::Real, si::Bool, sf::Bool)
    # Initiate time
    time = 0.0
    times = Float64[]

    # Find all transition times 
    while time < t 
        # Determine survival time
        dt = survival_time(s, t-time, si, sf)

        # Update system
        if time + dt < t
            # Store jump
            time += dt
            push!(times, time)
            si = !si
        else
            # Survive 
            time += dt
        end
    end
    return times
end

"""
    bridge_prob(s::Real, t::Real, si::Bool, sf::Bool)

Calculate the probability for a bridge between si and sf in time t, under biased dynamics s.
"""
function bridge_prob(s::Real, t::Real, si::Bool, sf::Bool)
    # Initiate time
    time = 0.0
    times = Float64[]
    prob = 1

    # Find all transition times 
    while time < t 
        # Determine survival time
        dt = survival_time(s, t-time, si, sf)

        # Update system
        if time + dt < t
            # Store jump
            prob *= survival(s, t-time, dt, si, sf)
            prob *= escape_rate(s, t-time-dt, si, sf)
            time += dt
            push!(times, time)
            si = !si
        else
            # Survive 
            prob *= survival(s, t-time, t-time, si, sf)
            time += dt
        end
    end
    return times, prob
end

"""
    time_integrated_magnetization(s::Real, t::Real, si::Bool, times::Array{Real})

Calculate the exponential of the time integrated magnetization for a spin, biased by s.
si is the initial state of the spin, t is the total time, and times are the times the spin
changes state.
"""
function time_integrated_magnetization(s::Real, t::Real, si, times::Array{Real})
    integral = 0
    for i = 1:length(times)+1
        tl = i == 1 ? 0.0 : times[i-1]
        tu = i == length(times) + 1 ? t : times[i]
        integral += (tu - tl) * (2*si-1)
        si = 1 - si
    end
    return exp(-s*integral)
end
