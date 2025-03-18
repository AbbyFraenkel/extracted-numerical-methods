# Implementation Guide: Multi-Level Goal-Oriented Method

This implementation guide provides Julia code examples following SciML conventions for the Multi-Level Goal-Oriented Method as described in the paper "A painless multi-level automatic goal-oriented."

## Type Hierarchy

```julia
"""
    AbstractMultiLevelSolver{T}

Abstract supertype for all multi-level solver algorithms using goal-oriented adaptivity.
The type parameter T represents the floating-point type used for calculations.
"""
abstract type AbstractMultiLevelSolver{T} end

"""
    MultiLevelGoalOrientedSolver{T, S, A, M} <: AbstractMultiLevelSolver{T}

Implements the Multi-Level Automatic Goal-Oriented Method for solving PDEs with a focus on accurate
computation of specific quantities of interest.

# Type Parameters
- `T`: Floating-point type for numerical calculations
- `S`: Type of the PDE solver
- `A`: Type of the adjoint solver
- `M`: Type of the marking strategy

# Fields
- `problem::PDEProblem{T}`: The PDE problem definition
- `qoi::QuantityOfInterest{T}`: The quantity of interest functional
- `tolerance::T`: The desired error tolerance for the QoI
- `max_levels::Int`: Maximum number of refinement levels allowed
- `solver::S`: Solver for the forward PDE problem
- `adjoint_solver::A`: Solver for the adjoint PDE problem
- `marking_strategy::M`: Strategy for selecting elements to refine
- `marking_fraction::T`: Parameter for the marking strategy (e.g., θ for Dörfler marking)
- `verbose::Bool`: Whether to print detailed progress information

# Mathematical Foundation
The method is based on the goal-oriented error estimation approach that expresses the error in the 
quantity of interest using the adjoint solution:

``J(u) - J(u_h) = a(u - u_h, z) = l(z) - a(u_h, z)``

where ``u`` is the exact solution, ``u_h`` is the discrete solution, ``z`` is the adjoint solution,
``a(·,·)`` is the bilinear form, and ``l(·)`` is the linear functional from the weak form of the PDE.

# References
- [Error representation formula (eq8)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#error-representation)
- [Multi-level solution (eq11)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#multi-level-solution)
"""
struct MultiLevelGoalOrientedSolver{T<:AbstractFloat, S, A, M} <: AbstractMultiLevelSolver{T}
    problem::PDEProblem{T}
    qoi::QuantityOfInterest{T}
    tolerance::T
    max_levels::Int
    solver::S
    adjoint_solver::A
    marking_strategy::M
    marking_fraction::T
    verbose::Bool
    
    function MultiLevelGoalOrientedSolver(
        problem::PDEProblem{T},
        qoi::QuantityOfInterest{T},
        tolerance::T;
        max_levels::Int = 10,
        solver::S = default_solver(problem),
        adjoint_solver::A = default_adjoint_solver(problem, qoi),
        marking_strategy::M = DörflerMarking(),
        marking_fraction::T = convert(T, 0.5),
        verbose::Bool = false
    ) where {T<:AbstractFloat, S, A, M}
        @assert tolerance > zero(T) "Tolerance must be positive"
        @assert max_levels > 0 "Maximum levels must be positive"
        @assert 0 < marking_fraction ≤ 1 "Marking fraction must be in (0,1]"
        
        new{T, S, A, M}(problem, qoi, tolerance, max_levels, solver, 
                        adjoint_solver, marking_strategy, marking_fraction, verbose)
    end
end

"""
    MultiLevelSolution{T, M, V}

Store the results of a multi-level goal-oriented solve.

# Type Parameters
- `T`: Floating-point type for numerical calculations
- `M`: Type for the mesh representation
- `V`: Type for solution vectors

# Fields
- `base_solution::V`: The solution at the coarsest level
- `corrections::Vector{V}`: Solution corrections at each refined level
- `meshes::Vector{M}`: Mesh at each level
- `error_estimates::Vector{T}`: Estimated error in the QoI at each level
- `qoi_values::Vector{T}`: Value of the quantity of interest at each level
- `level::Int`: The final refinement level reached
- `converged::Bool`: Whether the solution converged to the requested tolerance
"""
struct MultiLevelSolution{T<:AbstractFloat, M, V}
    base_solution::V
    corrections::Vector{V}
    meshes::Vector{M}
    error_estimates::Vector{T}
    qoi_values::Vector{T}
    level::Int
    converged::Bool
end
```

## Core Functions

```julia
"""
    solve(solver::MultiLevelGoalOrientedSolver{T}, initial_mesh) where {T}

Solve a PDE problem using the Multi-Level Goal-Oriented Method, starting from 
the given initial mesh.

# Arguments
- `solver::MultiLevelGoalOrientedSolver{T}`: The solver configuration
- `initial_mesh`: The initial mesh for the coarsest level

# Returns
- `MultiLevelSolution`: The solution containing all levels and error estimates

# Mathematical Details
The algorithm follows these steps:
1. Solve the PDE on the initial mesh
2. Solve the adjoint problem on the same mesh
3. Estimate the error in the quantity of interest
4. While the error is above tolerance:
   a. Mark elements for refinement based on error contributions
   b. Create the next level mesh
   c. Compute a correction at this level
   d. Update the solution by adding the correction
   e. Solve the adjoint problem on the refined mesh
   f. Recompute the error estimate

# Example
```julia
problem = PoissonProblem(f=(x,y) -> sin(π*x)*sin(π*y), domain=Rectangle(0,1,0,1))
qoi = PointwiseQoI((0.5, 0.5))  # Value at center point
solver = MultiLevelGoalOrientedSolver(problem, qoi, 1e-6)
mesh = create_uniform_mesh(Triangle, 10, 10)  # 10×10 triangular mesh
solution = solve(solver, mesh)
```

# References
- [Multi-level solution procedure algorithm](../algorithms/MultiLevel_GoalOriented_Method.md#2-multi-level-solution-procedure)
"""
function solve(solver::MultiLevelGoalOrientedSolver{T}, initial_mesh) where {T}
    # Extract parameters from solver
    problem = solver.problem
    qoi = solver.qoi
    tol = solver.tolerance
    max_levels = solver.max_levels
    marking_strategy = solver.marking_strategy
    marking_fraction = solver.marking_fraction
    verbose = solver.verbose
    
    # Initialize storage
    level = 0
    meshes = [initial_mesh]
    solutions = []
    corrections = []
    adjoint_solutions = []
    error_estimates = T[]
    qoi_values = T[]
    
    # Level 0: Solve PDE on initial mesh
    verbose && println("Level 0: Solving PDE on initial mesh...")
    u0 = solve_pde(solver.solver, problem, initial_mesh)
    push!(solutions, u0)
    
    # Solve adjoint problem on initial mesh
    verbose && println("Level 0: Solving adjoint problem...")
    z0 = solve_adjoint(solver.adjoint_solver, problem, qoi, u0, initial_mesh)
    push!(adjoint_solutions, z0)
    
    # Compute initial error estimate
    verbose && println("Level 0: Computing error estimate...")
    η0, η0_K = estimate_error(problem, qoi, u0, z0, initial_mesh)
    push!(error_estimates, η0)
    
    # Compute QoI value
    q0 = evaluate_qoi(qoi, u0, initial_mesh)
    push!(qoi_values, q0)
    
    verbose && println("Level 0: QoI = $q0, Error estimate = $η0")
    
    # Main multi-level loop
    while abs(error_estimates[end]) > tol && level < max_levels
        level += 1
        verbose && println("\nLevel $level: Starting refinement...")
        
        # Mark elements for refinement based on error indicators
        marked_elements = mark_elements(
            marking_strategy, 
            meshes[level], 
            η0_K, 
            marking_fraction
        )
        
        verbose && println("Level $level: Marked $(length(marked_elements)) elements for refinement")
        
        # Create refined mesh for this level
        new_mesh = refine_mesh(meshes[level], marked_elements)
        push!(meshes, new_mesh)
        
        # Set up correction problem
        verbose && println("Level $level: Computing residual...")
        residual = compute_residual(problem, solutions[level], meshes[level])
        
        # Solve for correction on the refined mesh
        verbose && println("Level $level: Solving correction problem...")
        correction = solve_correction(
            solver.solver, 
            problem, 
            residual, 
            new_mesh
        )
        push!(corrections, correction)
        
        # Update the solution: u^l = u^{l-1} + e^l
        verbose && println("Level $level: Updating solution...")
        ul = add_correction(solutions[level], correction, meshes[level], new_mesh)
        push!(solutions, ul)
        
        # Solve adjoint problem on the refined mesh
        verbose && println("Level $level: Solving adjoint problem...")
        zl = solve_adjoint(solver.adjoint_solver, problem, qoi, ul, new_mesh)
        push!(adjoint_solutions, zl)
        
        # Compute error estimate for this level
        verbose && println("Level $level: Computing error estimate...")
        ηl, ηl_K = estimate_error(problem, qoi, ul, zl, new_mesh)
        push!(error_estimates, ηl)
        
        # Compute QoI value
        ql = evaluate_qoi(qoi, ul, new_mesh)
        push!(qoi_values, ql)
        
        verbose && println("Level $level: QoI = $ql, Error estimate = $ηl")
        
        # Check for diminishing returns
        if level > 1 && abs(error_estimates[end-1] / error_estimates[end]) < 1.1
            verbose && println("Warning: Diminishing returns in error reduction. Consider different refinement strategy.")
        end
    end
    
    # Create the final solution object
    converged = abs(error_estimates[end]) <= tol
    
    verbose && println("\nFinal result after $level levels:")
    verbose && println("  Converged: $converged")
    verbose && println("  Final QoI value: $(qoi_values[end])")
    verbose && println("  Final error estimate: $(error_estimates[end])")
    
    return MultiLevelSolution{T, typeof(initial_mesh), typeof(u0)}(
        solutions[1],            # base_solution
        corrections,             # corrections
        meshes,                  # meshes
        error_estimates,         # error_estimates
        qoi_values,              # qoi_values
        level,                   # level
        converged                # converged
    )
end

"""
    evaluate_qoi(solution::MultiLevelSolution, qoi::QuantityOfInterest)

Evaluate the quantity of interest on the multi-level solution.

# Arguments
- `solution::MultiLevelSolution`: The multi-level solution
- `qoi::QuantityOfInterest`: The quantity of interest functional

# Returns
- `value`: The value of the quantity of interest
- `error_estimate`: Estimated error in the quantity of interest

# Mathematical Details
The QoI is evaluated on the highest level solution, and the error estimate
from the solution object is returned.

# Example
```julia
value, error = evaluate_qoi(solution, qoi)
println("QoI = $value ± $error")
```
"""
function evaluate_qoi(solution::MultiLevelSolution, qoi::QuantityOfInterest)
    # The QoI value is already computed and stored during the solution process
    value = solution.qoi_values[end]
    error_estimate = solution.error_estimates[end]
    
    return value, error_estimate
end
```

## Error Estimation Functions

```julia
"""
    estimate_error(problem, qoi, u_h, z_h, mesh)

Estimate the error in the quantity of interest using the dual-weighted residual method.

# Arguments
- `problem`: The PDE problem
- `qoi`: The quantity of interest functional
- `u_h`: The primal solution
- `z_h`: The adjoint solution
- `mesh`: The mesh on which the solutions are defined

# Returns
- `η`: Global error estimate
- `η_K`: Element-wise error indicators

# Mathematical Details
The error in the QoI is estimated using the dual-weighted residual formula:
```
J(u) - J(u_h) ≈ l(z) - a(u_h, z) = ∑_K η_K
```
where `η_K` represents the contribution from element K.

# References
- [Error representation formula (eq8)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#error-representation)
- [Error estimation (eq9-10)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#dual-weighted-residual)
"""
function estimate_error(problem, qoi, u_h, z_h, mesh)
    # Compute element residuals and assemble error indicators
    η_K = zeros(num_elements(mesh))
    
    for (i, element) in enumerate(elements(mesh))
        # Compute interior residual contribution
        r_K = element_residual(problem, u_h, element, mesh)
        
        # Compute boundary residual contribution (for non-Dirichlet boundaries)
        r_∂K = boundary_residual(problem, u_h, element, mesh)
        
        # Apply dual weighting: use z_h as an approximation to z - z_h
        # In practice, better approximations can be used, like higher-order recovery
        z_h_K = restrict_to_element(z_h, element, mesh)
        
        # Compute the error indicator for this element
        η_K[i] = abs(integrate_element(r_K, z_h_K, element, mesh) + 
                     integrate_boundary(r_∂K, z_h_K, element, mesh))
    end
    
    # Global error estimate is the sum of element-wise indicators
    η = sum(η_K)
    
    return η, η_K
end

"""
    mark_elements(marking_strategy, mesh, error_indicators, marking_fraction)

Mark elements for refinement based on error indicators and a marking strategy.

# Arguments
- `marking_strategy`: The strategy to use for element marking
- `mesh`: The current mesh
- `error_indicators`: Element-wise error indicators
- `marking_fraction`: Strategy-specific parameter (e.g., θ for Dörfler marking)

# Returns
- `marked_elements`: Indices of elements selected for refinement

# Mathematical Details
For Dörfler (bulk) marking with parameter θ, elements are marked such that:
```
∑_{K ∈ M} |η_K| ≥ θ ∑_{K ∈ T_h} |η_K|
```
where M is the set of marked elements.

# References
- [Dörfler marking (eq14)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#dörfler-marking-strategy)
"""
function mark_elements(::DörflerMarking, mesh, error_indicators, θ)
    # Sort elements by error indicator in descending order
    sorted_indices = sortperm(error_indicators, rev=true)
    
    # Compute total error
    total_error = sum(error_indicators)
    
    # Initialize marked elements
    marked_indices = Int[]
    marked_error = 0.0
    
    # Mark elements until reaching the threshold fraction θ
    for i in sorted_indices
        push!(marked_indices, i)
        marked_error += error_indicators[i]
        
        # Check if we've reached the threshold
        if marked_error >= θ * total_error
            break
        end
    end
    
    return marked_indices
end
```

## Helper Functions for Multi-Level Solutions

```julia
"""
    compute_residual(problem, u_h, mesh)

Compute the residual of the PDE given a solution u_h on mesh.

# Arguments
- `problem`: The PDE problem
- `u_h`: The approximate solution
- `mesh`: The mesh on which u_h is defined

# Returns
- `residual`: The residual function for use in correction problems

# Mathematical Details
For a PDE in weak form a(u,v) = l(v), the residual is:
```
r(v) = l(v) - a(u_h,v)
```
This represents how well u_h satisfies the PDE.
"""
function compute_residual(problem, u_h, mesh)
    # Create a functor that computes the residual for any test function v
    function residual(v)
        # Evaluate the right-hand side linear form
        rhs = assemble_linear_form(problem, v, mesh)
        
        # Evaluate the bilinear form with u_h
        lhs = assemble_bilinear_form(problem, u_h, v, mesh)
        
        # Return the residual
        return rhs - lhs
    end
    
    return residual
end

"""
    solve_correction(solver, problem, residual, mesh)

Solve for the correction on a refined mesh using the residual.

# Arguments
- `solver`: The PDE solver
- `problem`: The PDE problem
- `residual`: The residual function
- `mesh`: The refined mesh

# Returns
- `correction`: The solution to the correction problem

# Mathematical Details
The correction e at level l satisfies:
```
a(e^l, v_h^l) = l(v_h^l) - a(u^{l-1}, v_h^l) = residual(v_h^l)
```
This is equivalent to solving the original PDE but with the residual as the right-hand side.

# References
- [Correction equation (eq12)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#multi-level-solution)
"""
function solve_correction(solver, problem, residual, mesh)
    # Create a modified problem with the residual as the right-hand side
    correction_problem = create_correction_problem(problem, residual)
    
    # Solve this problem on the refined mesh
    correction = solve_pde(solver, correction_problem, mesh)
    
    return correction
end

"""
    add_correction(u_prev, correction, mesh_prev, mesh_curr)

Add a correction to the previous level's solution to get the current level solution.

# Arguments
- `u_prev`: Solution at the previous level
- `correction`: Correction at the current level
- `mesh_prev`: Mesh at the previous level
- `mesh_curr`: Mesh at the current level

# Returns
- `u_curr`: Solution at the current level

# Mathematical Details
The multi-level solution is updated as:
```
u^l = u^{l-1} + e^l
```
This requires appropriate transfer operations between different meshes.

# References
- [Multi-level solution (eq11)](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#multi-level-solution)
"""
function add_correction(u_prev, correction, mesh_prev, mesh_curr)
    # Transfer the previous solution to the current mesh
    u_prev_on_curr = transfer_solution(u_prev, mesh_prev, mesh_curr)
    
    # Add the correction
    u_curr = u_prev_on_curr + correction
    
    return u_curr
end
```

## Integration with KitchenSink

This section outlines how to integrate the Multi-Level Goal-Oriented Method with the KitchenSink solver framework.

```julia
"""
    register_with_kitchensink()

Register the Multi-Level Goal-Oriented Method with KitchenSink.
"""
function register_with_kitchensink()
    # Register the main solver
    KitchenSink.register_solver(
        "MultiLevelGoalOriented", 
        MultiLevelGoalOrientedSolver,
        "Multi-Level Automatic Goal-Oriented Method for adaptive PDE solving focused on quantities of interest"
    )
    
    # Register related utility functions
    KitchenSink.register_utility(
        "EstimateError",
        estimate_error,
        "Estimate error in quantity of interest using dual-weighted residual method"
    )
    
    KitchenSink.register_utility(
        "MarkElements",
        mark_elements,
        "Mark elements for refinement based on error indicators"
    )
    
    KitchenSink.register_utility(
        "EvaluateQoI",
        evaluate_qoi,
        "Evaluate quantity of interest on multi-level solution"
    )
    
    # Register example problem types
    KitchenSink.register_example(
        "GoalOrientedPoisson",
        create_goal_oriented_poisson,
        "Poisson problem with point value quantity of interest"
    )
end
```

This implementation guide provides a detailed framework for implementing the Multi-Level Goal-Oriented Method in Julia following SciML conventions. The code structures and function signatures maintain exact fidelity to the mathematical descriptions in the original paper while providing a clear path to integration with the KitchenSink solver framework.
