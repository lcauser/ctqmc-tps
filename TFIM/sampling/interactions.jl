#= 
    The interaction structure contains information on how spins in a lattice interact with each other,
    in terms of Z matrices...
=#

struct Interactions{K<:AbstractArray, J<:Real,S<:AbstractArray{J}}
    dims::Vector{Int64}
    sites::K
    coeffs::S
end

"""
    interactionSpins(I::Interactions, spins::Spins, idx::Int)

Find all the indexs of spins which interact with some spin at a given index.
"""
function interactionSpins(I::Interactions, spins::Spins, idx::Int)
    idxs = [idx]
    for i in I.sites
        for site in i
            if length(I.dims) == 1
                s = idx + site
                s = s <= 0 ? length(spins.spins) + s : s
                s = s > length(spins.spins) ? s - length(spins.spins) : s
            elseif length(I.dims) == 2
                # Find the position of the first spin
                sy = Int(ceil((idx) / I.dims[2]))
                sx = idx - (sy - 1) * I.dims[2]

                # Find the position of the new spin
                sy2 = sy + site[1]
                sx2 = sx + site[2]
                sy2 = sy2 <= 0 ? I.dims[1] + sy2 : sy2
                sx2 = sx2 <= 0 ? I.dims[2] + sx2 : sx2
                sy2 = sy2 > I.dims[1] ? sy2 - I.dims[1] : sy2
                sx2 = sx2 > I.dims[2] ? sx2 - I.dims[2] : sx2

                # Find the index of the spin 
                s = (sy2 - 1) * I.dims[2] + sx2
            elseif length(I.dims) == 3
                # Find the area & volume
                A = I.dims[1]*I.dims[2]
                N = I.dims[3]*A

                # Find the position of the first spin
                sz = Int(ceil((idx) / A))
                sy = Int(ceil((idx - (sz - 1) * A) / I.dims[1]))
                sx = idx - (sz - 1) * A - (sy - 1)*I.dims[1]
                pos = [sx, sy, sz]

                # Find the position of the new spin
                pos_new = [pos[i] + site[i] for i = 1:3]
                pos_new = [pos_new[i] <= 0 ? I.dims[i] + pos_new[i] : pos_new[i] for i=1:3]
                pos_new = [pos_new[i] > I.dims[i] ? pos_new[i] - I.dims[i] : pos_new[i] for i=1:3]
                
                # Find the index of the spin 
                s = (pos_new[3] - 1) * A + ((pos_new[2] - 1) * I.dims[1]) + pos_new[1]
            end
            push!(idxs, s)
        end
    end
    return idxs
end

"""
    effectiveConstraint(I::Interactions, spins::Spins, idx::Int)

Determines the effective time-dependant dynamics for a spin using the trajectories 
of other spins on the lattice.
"""
function effectiveConstraint(I::Interactions, spins::Spins, idx::Int)
    # Reconstruct the trajectory
    idxs = interactionSpins(I, spins, idx)
    times, states = reconstruct(spins, idxs[2:end])

     # Find the J values
     mzs = 2 .* states .- 1
     Js = zeros(Float64, length(times))
     site = 1
     for i = 1:length(I.sites)
        J = I.coeffs[i] .* ones(Float64, length(times))
        for _ = 1:length(I.sites[i])
            J = J .* mzs[:, site]
            site += 1
        end
        Js += J
     end
     return times, Js
end


"""
    calculateInteraction(I::Interactions, spins::Spins, idx::Int)

Calculates all interactions for some spin with a given index.
"""
function calculateInteraction(I::Interactions, spins::Spins, idx::Int)
    # Reconstruct the trajectory
    idxs = interactionSpins(I, spins, idx)
    times, states = reconstruct(spins, idxs)

    # Find the interactions
    mzs = 2 .* states .- 1
    interactions = zeros(Float64, length(I.sites), length(times))
    site = 2
    for i = 1:length(I.sites)
       interaction = mzs[:, 1]
       for j = 1:length(I.sites[i])
           interaction = interaction .* mzs[:, site]
           site += 1
       end
       interactions[i, :] = interaction
    end

    # Time integrate 
    integral = zeros(Float64, length(I.sites))
    for i in eachindex(times)
        t0 = times[i]
        t1 = i == length(times) ? spins.time : times[i+1]
        integral += (t1-t0) .* interactions[:, i]
    end
    return integral
end

"""
    update_periodic!(spins::Spins, I::Interactions, idx::Int)
    update_periodic!(spins::Spins, I::Interactions)

Update a trajectory of spins with some interaction, with PBC in time.
"""
function update_periodic!(spins::Spins, I::Interactions, idx::Int)
    # Find the effective constraints and update the spin
    times, Js = effectiveConstraint(I, spins, idx)

    # Update the spin 
    update_periodic!(spins.spins[idx], times, Js)

    return idx
end
update_periodic!(spins::Spins, I::Interactions) = update_periodic!(spins, I, rand(1:length(spins.spins)))


"""
    Ising1D(N::Int, J::Real, delta::Real = 0.0)

The interactions for the Ising model in one dimension. J is the coupling strength,
g is the onsite potential.
"""
function Ising1D(N::Int, J::Real, g::Real = 0.0)
    sites = [[-1], [1], Int[]]
    coeffs = [-J, -J, -g]
    return Interactions([N], sites, coeffs)
end


"""
    Ising2D(Nx::Int, Ny::Int, J::Real, g::Real)

The interactions for the Ising model in two dimensions. J is the coupling strength,
g is the onsite potential.
"""
function Ising2D(Nx::Int, Ny::Int, J::Real, g::Real)
    sites = [[[-1, 0]], [[1, 0]], [[0, -1]], [[0, 1]], Int[]]
    coeffs = [-J, -J, -J, -J, -g]
    return Interactions([Ny, Nx], sites, coeffs)
end


"""
    Ising3D(Nx::Int, Ny::Int, Nz::Int, J::Real, g::Real)

The interactions for the Ising model in two dimensions. J is the coupling strength,
g is the onsite potential.
"""
function Ising3D(Nx::Int, Ny::Int, Nz::Int, J::Real, g::Real)
    sites = [[[-1, 0, 0]], [[1, 0, 0]], [[0, -1, 0]], [[0, 1, 0]], [[0, 0, -1]], [[0, 0, 1]], Int[]]
    coeffs = [-J, -J, -J, -J, -J, -J, -g]
    return Interactions([Nx, Ny, Nz], sites, coeffs)
end


"""
    Plaquette(Nx::Int, Ny::Int, J::Real, g::Real = 0.0) 

The interactions for the Triangular plaquette model. J is the coupling strength,
g is the onsite potential.
"""
function Plaquette(Nx::Int, Ny::Int, J::Real, g::Real = 0.0)
    sites = [[[1, 0], [1, 1]], [[-1, 0], [0, 1]], [[0, -1], [-1, -1]], Int[]]
    coeffs = [-J, -J, -J, -g]
    return Interactions([Ny, Nx], sites, coeffs)
end


"""
    Correl() 

An interaction to calculate correlations.
"""
function Correl()
    sites = [[1]]
    coeffs = [1]
    return Interactions([2], sites, coeffs)
end