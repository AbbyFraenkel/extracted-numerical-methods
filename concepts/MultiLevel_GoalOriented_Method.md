# Multi-Level Automatic Goal-Oriented Method

## L1: Core Concept (Essential Definition)

The Multi-Level Automatic Goal-Oriented Method is an adaptive refinement approach for solving partial differential equations with a focus on accurately computing specific quantities of interest rather than the entire solution. It combines hierarchical basis functions, goal-oriented error estimation using adjoint solutions, and multi-level correction techniques to efficiently achieve high accuracy in the quantities of interest with minimal computational cost. The method automatically identifies and refines regions of the domain that contribute most significantly to the error in the quantity of interest.

## L2: Functional Details (Mathematical Formulation)

### Mathematical Foundation

The method is built on several key mathematical principles:

1. **Partial Differential Equation (PDE) Problem**: 
   Consider a PDE problem of finding $u \in V$ such that:
   ```
   a(u, v) = l(v)  ∀v ∈ V
   ```
   where $a(\cdot, \cdot)$ is a bilinear form, $l(\cdot)$ is a linear functional, and $V$ is an appropriate function space.

2. **Quantity of Interest (QoI)**: 
   A linear or nonlinear functional $J(u)$ representing a specific output of interest:
   ```
   J(u) = ∫_Ω j(u) dΩ + ∫_∂Ω g(u) dΓ
   ```

3. **Adjoint Problem**:
   Find $z \in V$ such that:
   ```
   a(v, z) = J'(u)(v)  ∀v ∈ V
   ```
   where $J'(u)(v)$ is the Gâteaux derivative of $J$ at $u$ in the direction $v$.

4. **Error Representation**:
   For the discretized problem (finding $u_h \in V_h$ where $V_h \subset V$), the error in the QoI can be represented as:
   ```
   J(u) - J(u_h) = a(u - u_h, z) = l(z) - a(u_h, z)
   ```

5. **Error Estimation**:
   The error can be localized to elements $K$ in the mesh $\mathcal{T}_h$:
   ```
   J(u) - J(u_h) ≈ ∑_{K ∈ \mathcal{T}_h} η_K
   ```
   where $\eta_K$ are element-wise error indicators.

### Multi-Level Approach

The multi-level approach builds solutions hierarchically:

1. **Level 0 (Base Level)**: 
   Solve the problem on an initial coarse mesh to get $u^0 \in V_h^0$.

2. **Correction at Level l**:
   - Compute the residual using the solution from the previous level
   - Solve for a correction $e^l \in V_h^l$ such that:
     ```
     a(e^l, v_h^l) = l(v_h^l) - a(u^{l-1}, v_h^l)  ∀v_h^l ∈ V_h^l
     ```
   - Update the solution: $u^l = u^{l-1} + e^l$

3. **Multi-Level Solution**:
   The final solution after $L$ levels is given by:
   ```
   u^L = u^0 + ∑_{l=1}^L e^l
   ```

### Adaptive Refinement Strategy

The goal-oriented refinement strategy involves:

1. **Error Indicator Computation**:
   Compute element-wise error indicators $\eta_K$ using the adjoint solution and residuals.

2. **Element Marking**:
   Use the Dörfler (bulk) marking strategy to select elements for refinement:
   ```
   ∑_{K ∈ \mathcal{M}} |η_K| ≥ θ ∑_{K ∈ \mathcal{T}_h} |η_K|
   ```
   where $\mathcal{M}$ is the set of marked elements and $\theta \in (0,1)$ is a parameter.

3. **Mesh Refinement**:
   Refine the marked elements to create the next level mesh.

## L3: Complete Knowledge (Theoretical Foundations)

### Theoretical Error Analysis

The goal-oriented error estimation approach provides rigorous control of the error in the quantity of interest. For problems with sufficient regularity, the error converges as:

```
|J(u) - J(u_h)| ≤ C h^{2p} |u|_{p+1} |z|_{p+1}
```

where $p$ is the polynomial order, $h$ is the mesh size, and $|\cdot|_{p+1}$ denotes an appropriate Sobolev norm. This indicates that the error in the QoI converges twice as fast as the error in the solution itself.

The multi-level approach offers several advantages:

1. **Efficient Error Reduction**: Error in the QoI typically reduces by a factor of $10^{-2}$ to $10^{-3}$ per refinement level for well-behaved problems.

2. **Optimal Refinement**: The goal-oriented approach targets refinement precisely where it matters for the QoI, avoiding unnecessary refinement elsewhere.

3. **Computational Efficiency**: The multi-level correction approach solves smaller problems at each level, rather than solving the entire problem on increasingly refined meshes.

### Adjoint-Based Error Estimation

The adjoint-based error estimation approach considers the dual-weighted residual method, where the error is represented as:

```
J(u) - J(u_h) = ∑_K r_K(u_h)(z - z_h) + r_∂K(u_h)(z - z_h)
```

where $r_K$ and $r_∂K$ are the element interior and boundary residuals, respectively. The accuracy of this estimate depends on how well we can approximate $z - z_h$.

Strategies for improving this approximation include:
- Using higher polynomial order for the adjoint solution
- Solving the adjoint problem on a finer mesh
- Using patch recovery techniques to enhance the adjoint approximation

### Implementation Considerations

1. **Data Structures**:
   - Hierarchical mesh representation
   - Efficient storage of corrections at each level
   - Solution transfer operators between levels

2. **Solvers**:
   - Direct solvers for smaller problems
   - Iterative solvers with appropriate preconditioners for larger problems
   - Multigrid or domain decomposition methods for parallel implementations

3. **Adjoint Problem**:
   - The adjoint problem has the transposed operator of the primal problem
   - Careful handling of boundary conditions is required
   - Storage requirements can be reduced using checkpointing strategies for time-dependent problems

4. **Convergence Criteria**:
   - Error estimate below specified tolerance
   - Maximum number of refinement levels reached
   - Diminishing returns in error reduction (efficiency index analysis)

### Advanced Extensions

1. **hp-Adaptivity**:
   The method can be extended to include p-refinement (increasing polynomial order) in addition to h-refinement (mesh refinement), leading to exponential convergence rates for smooth solutions.

2. **Space-Time Adaptivity**:
   For time-dependent problems, the approach can be extended to space-time adaptivity, where both spatial and temporal discretizations are adaptively refined.

3. **Nonlinear Problems**:
   For nonlinear problems, the method requires linearization of both the primal and adjoint problems, typically using Newton's method or fixed-point iterations.

4. **Multi-Goal Adaptation**:
   When multiple quantities of interest are considered, weighted combinations of goal functionals can be used to drive the adaptation process.

### References to Key Equations

- Error representation formula: [eq8](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#error-representation)
- Multi-level solution: [eq11](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#multi-level-solution)
- Error estimation: [eq13](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#multi-level-error-estimation)
- Dörfler marking: [eq14](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#dörfler-marking-strategy)
- Convergence rate: [eq16](../mathematics/equations/MultiLevel_GoalOriented_Equations.md#convergence-rate)
