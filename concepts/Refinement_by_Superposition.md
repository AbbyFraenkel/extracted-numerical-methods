# Refinement-by-Superposition Method for curl- and div-Conforming Discretizations

## L1: Core Concept (Essential Definition)

The Refinement-by-Superposition Method is an adaptive refinement technique for finite element discretizations that preserves essential mathematical properties in electromagnetics and fluid dynamics applications. Unlike standard mesh refinement that can break conformity requirements, this method maintains crucial continuity properties (tangential for curl-conforming or normal for div-conforming elements) while enabling local refinement. It works by superimposing finer-scale basis functions on existing elements and enforcing appropriate constraints, creating a hierarchical representation that preserves exactness properties of differential operators and maintains the mathematical structure of the underlying physical problem.

## L2: Functional Details (Mathematical Formulation)

### Mathematical Foundation

The method is anchored in several key mathematical concepts:

1. **De Rham Complex**: The method preserves the exact sequence of function spaces, known as the de Rham complex:
   ```
   H¹(Ω) --∇--> H(curl; Ω) --∇×--> H(div; Ω) --∇·--> L²(Ω)
   ```
   This sequence encodes fundamental physical properties in electromagnetics and fluid dynamics.

2. **Curl-Conforming Elements**: These ensure tangential continuity across element boundaries, critical for electromagnetic fields:
   ```
   n × u|_{∂K₁} = n × u|_{∂K₂}
   ```
   where n is the normal vector at the interface between elements K₁ and K₂.

3. **Div-Conforming Elements**: These ensure normal continuity across element boundaries, essential for conservation laws:
   ```
   n · u|_{∂K₁} = n · u|_{∂K₂}
   ```

4. **Hierarchical Basis Functions**: The superposition approach uses hierarchical basis functions:
   ```
   φ^S_i(x) = ∑_{j=1}^{n_b} w_{ij} φ_j(x)
   ```
   where the weights w_{ij} define how basis functions are combined.

5. **Commuting Diagram Property**: The method preserves commuting diagrams that ensure consistency between continuous and discrete differential operators:
   ```
   ∇ × (∇φ_h) = 0,    ∇ · (∇ × v_h) = 0
   ```

### Refinement-by-Superposition Procedure

The core procedure involves these steps:

1. **Element Marking**: Identify elements for refinement based on error indicators or other criteria.

2. **Superposition Element Creation**:
   - Geometrically refine the marked elements
   - Set up hierarchical basis functions on the refined elements
   - For curl-conforming elements, use edge-based functions
   - For div-conforming elements, use face-based functions

3. **Constraint Enforcement**:
   - For curl-conforming elements, enforce tangential continuity at interfaces
   - For div-conforming elements, enforce normal continuity at interfaces
   - Set up constraint equations relating degrees of freedom (DOFs):
     ```
     u_i = ∑_{j ∈ J} c_{ij} u_j
     ```
     where c_{ij} are constraint coefficients

4. **DOF Management**:
   - Identify constrained and independent DOFs
   - Number the DOFs to optimize solver performance
   - Transform system matrices using constraint equations:
     ```
     Ã = C^T A C
     ```
     where C is the constraint matrix

5. **Verification of Conformity**:
   - Check appropriate continuity conditions
   - Verify exactness properties
   - Confirm commuting diagram properties

### Error Analysis and Convergence Properties

For curl-conforming approximations, the error estimate is:
```
‖u - u_h‖_{H(curl)} ≤ C h^p |u|_{H^{p+1}(curl)}
```

For div-conforming approximations:
```
‖u - u_h‖_{H(div)} ≤ C h^p |u|_{H^{p+1}(div)}
```

where h is the mesh size and p is the polynomial order.

## L3: Complete Knowledge (Theoretical Foundations)

### Detailed Mathematical Theory

#### Function Spaces and Norms

The method operates within these function spaces:

1. **H(curl)**: Functions with square-integrable curl:
   ```
   H(curl; Ω) = {v ∈ L²(Ω)³ : ∇ × v ∈ L²(Ω)³}
   ```
   with norm:
   ```
   ‖v‖²_{H(curl)} = ‖v‖²_{L²(Ω)} + ‖∇ × v‖²_{L²(Ω)}
   ```

2. **H(div)**: Functions with square-integrable divergence:
   ```
   H(div; Ω) = {v ∈ L²(Ω)³ : ∇ · v ∈ L²(Ω)}
   ```
   with norm:
   ```
   ‖v‖²_{H(div)} = ‖v‖²_{L²(Ω)} + ‖∇ · v‖²_{L²(Ω)}
   ```

#### Discrete De Rham Complex

The discrete version of the de Rham complex is:
```
V_h¹ --∇--> V_h^{curl} --∇×--> V_h^{div} --∇·--> V_h⁰
```

The method ensures that the following diagram commutes:
```
H¹(Ω) --∇--> H(curl; Ω) --∇×--> H(div; Ω) --∇·--> L²(Ω)
   ↓ Π_h¹       ↓ Π_h^{curl}     ↓ Π_h^{div}      ↓ Π_h⁰  
V_h¹ --∇--> V_h^{curl} --∇×--> V_h^{div} --∇·--> V_h⁰
```
where Π_h are projection operators.

#### Element Types and Basis Functions

1. **Curl-Conforming Elements**:
   - Nédélec elements (edge elements)
   - Edge basis functions with tangential continuity
   - Degrees of freedom associated with edges

2. **Div-Conforming Elements**:
   - Raviart-Thomas elements (face elements)
   - Face basis functions with normal continuity
   - Degrees of freedom associated with faces

#### Constraint Formulation Details

When introducing superposition elements, constraints must be enforced to maintain conformity. For an edge shared between a parent element and refined elements, the constraints ensure:

1. **Curl-Conforming Case**:
   ```
   u_t^{parent} = ∑_i w_i u_t^{child,i}
   ```
   where u_t represents the tangential component and w_i are weights.

2. **Div-Conforming Case**:
   ```
   u_n^{parent} = ∑_i w_i u_n^{child,i}
   ```
   where u_n represents the normal component.

### Implementation Considerations

#### Finite Element Assembly

The superposition method modifies the standard finite element assembly process:

1. **Matrix Transformation**:
   The system matrix A is transformed to Ã = C^T A C, where C is the constraint matrix.

2. **Right-hand Side Transformation**:
   Similarly, the right-hand side b is transformed to b̃ = C^T b.

3. **Solution Recovery**:
   After solving Ã x̃ = b̃, the full solution is recovered using x = C x̃.

#### Basis Function Implementation

1. **Edge Elements (Curl-Conforming)**:
   - First-order Nédélec: u = a + b × x
   - Higher-order: add appropriate polynomial terms while preserving tangential continuity

2. **Face Elements (Div-Conforming)**:
   - First-order Raviart-Thomas: u = a + b x
   - Higher-order: add appropriate polynomial terms while preserving normal continuity

#### Solver Technology

1. **Direct Solvers**:
   - Suitable for moderate-sized problems
   - Handle constraints through matrix transformations
   - May suffer from increased fill-in due to constraints

2. **Iterative Solvers**:
   - Preferable for large-scale problems
   - Require specialized preconditioners for constrained systems
   - Domain decomposition methods can be effective

3. **Multigrid Methods**:
   - Can leverage the hierarchical structure
   - Require careful handling of constrained DOFs
   - Can achieve optimal complexity O(N) for N DOFs

### Applications and Extensions

#### Electromagnetic Applications

1. **Maxwell's Equations**: The curl-conforming case is essential for electric and magnetic field modeling.
2. **Eddy Current Problems**: Used for low-frequency electromagnetics in conducting materials.
3. **Wave Propagation**: Applied to high-frequency electromagnetic wave modeling.

#### Fluid Dynamics Applications

1. **Incompressible Flows**: The div-conforming case ensures mass conservation.
2. **Porous Media Flows**: Used for modeling flows in heterogeneous porous media.
3. **Mixed Methods**: Combined with other elements for saddle-point problems.

#### Advanced Extensions

1. **hp-Adaptivity**: Combined with polynomial enrichment for exponential convergence.
2. **Time-Dependent Problems**: Extended to space-time formulations.
3. **Nonlinear Problems**: Applied within iterative schemes for nonlinear PDEs.
4. **Multi-Physics Coupling**: Used to maintain physical constraints at interfaces between different physics domains.

### References to Key Equations

- De Rham complex: [eq1](../mathematics/equations/Refinement_by_Superposition_Equations.md#function-spaces-and-de-rham-complex)
- Curl-conforming space: [eq2](../mathematics/equations/Refinement_by_Superposition_Equations.md#curl-conforming-space)
- Div-conforming space: [eq3](../mathematics/equations/Refinement_by_Superposition_Equations.md#div-conforming-space)
- Tangential continuity: [eq8](../mathematics/equations/Refinement_by_Superposition_Equations.md#tangential-continuity-constraint)
- Normal continuity: [eq9](../mathematics/equations/Refinement_by_Superposition_Equations.md#normal-continuity-constraint)
- Superposition basis: [eq10](../mathematics/equations/Refinement_by_Superposition_Equations.md#superposition-basis-construction)
- Constraint equation: [eq12](../mathematics/equations/Refinement_by_Superposition_Equations.md#constraint-equation-for-dofs)
- System transformation: [eq13](../mathematics/equations/Refinement_by_Superposition_Equations.md#system-matrix-transformation)
- Error estimate (curl): [eq14](../mathematics/equations/Refinement_by_Superposition_Equations.md#error-estimate-for-curl-conforming-superposition)
- Error estimate (div): [eq15](../mathematics/equations/Refinement_by_Superposition_Equations.md#error-estimate-for-div-conforming-superposition)
