# Equation Database: Quasi-Sequential Algorithm for PDE-Constrained Optimization

## Source Paper
"A quasi-sequential algorithm for PDE-constrained optimization based on space-time OCFE"

## Equations

### PDE-Constrained Optimization Problem

{
  "equation_id": "eq1",
  "latex": "$$\\min_{y, u} J(y, u) = \\frac{1}{2}\\|y - y_d\\|^2_{L^2(\\Omega_T)} + \\frac{\\alpha}{2}\\|u\\|^2_{L^2(\\Omega_T)}$$",
  "description": "Objective function for PDE-constrained optimization with tracking term and regularization",
  "section": "2.1",
  "page": "3"
}

{
  "equation_id": "eq2",
  "latex": "$$\\text{subject to} \\quad \\frac{\\partial y}{\\partial t} + \\mathcal{L}y = f + Bu \\quad \\text{in } \\Omega \\times (0,T)$$",
  "description": "PDE constraint - time-dependent partial differential equation",
  "section": "2.1",
  "page": "3"
}

{
  "equation_id": "eq3",
  "latex": "$$y(\\mathbf{x}, 0) = y_0(\\mathbf{x}) \\quad \\text{in } \\Omega$$",
  "description": "Initial condition for the PDE",
  "section": "2.1",
  "page": "3"
}

{
  "equation_id": "eq4",
  "latex": "$$\\mathcal{B}y = g \\quad \\text{on } \\partial\\Omega \\times (0,T)$$",
  "description": "Boundary conditions for the PDE",
  "section": "2.1",
  "page": "3"
}

### Optimality System

{
  "equation_id": "eq5",
  "latex": "$$\\frac{\\partial y}{\\partial t} + \\mathcal{L}y = f + Bu \\quad \\text{in } \\Omega \\times (0,T)$$",
  "description": "State equation in the optimality system",
  "section": "2.2",
  "page": "4"
}

{
  "equation_id": "eq6",
  "latex": "$$-\\frac{\\partial p}{\\partial t} + \\mathcal{L}^*p = y - y_d \\quad \\text{in } \\Omega \\times (0,T)$$",
  "description": "Adjoint equation in the optimality system",
  "section": "2.2",
  "page": "4"
}

{
  "equation_id": "eq7",
  "latex": "$$\\alpha u - B^*p = 0 \\quad \\text{in } \\Omega \\times (0,T)$$",
  "description": "Gradient equation relating control and adjoint variables",
  "section": "2.2",
  "page": "4"
}

### Space-Time Discretization

{
  "equation_id": "eq8",
  "latex": "$$\\Omega_T = \\Omega \\times (0,T) = \\bigcup_{k=1}^{N_e} \\Omega_k$$",
  "description": "Decomposition of space-time domain into elements",
  "section": "3.1",
  "page": "5"
}

{
  "equation_id": "eq9",
  "latex": "$$y_h(\\mathbf{x}, t) = \\sum_{i=1}^{N_y} Y_i \\phi_i(\\mathbf{x}, t)$$",
  "description": "Approximation of state variable using basis functions",
  "section": "3.1",
  "page": "5"
}

{
  "equation_id": "eq10",
  "latex": "$$u_h(\\mathbf{x}, t) = \\sum_{i=1}^{N_u} U_i \\psi_i(\\mathbf{x}, t)$$",
  "description": "Approximation of control variable using basis functions",
  "section": "3.1",
  "page": "5"
}

{
  "equation_id": "eq11",
  "latex": "$$p_h(\\mathbf{x}, t) = \\sum_{i=1}^{N_p} P_i \\chi_i(\\mathbf{x}, t)$$",
  "description": "Approximation of adjoint variable using basis functions",
  "section": "3.1",
  "page": "5"
}

### Discretized Optimality System

{
  "equation_id": "eq12",
  "latex": "$$\\mathbf{K_y Y} = \\mathbf{F} + \\mathbf{B U}$$",
  "description": "Discretized state equation",
  "section": "3.2",
  "page": "6"
}

{
  "equation_id": "eq13",
  "latex": "$$\\mathbf{K_p P} = \\mathbf{M}(\\mathbf{Y} - \\mathbf{Y_d})$$",
  "description": "Discretized adjoint equation",
  "section": "3.2",
  "page": "6"
}

{
  "equation_id": "eq14",
  "latex": "$$\\alpha \\mathbf{M_u U} - \\mathbf{B}^T \\mathbf{P} = \\mathbf{0}$$",
  "description": "Discretized gradient equation",
  "section": "3.2",
  "page": "6"
}

### Quasi-Sequential Algorithm

{
  "equation_id": "eq15",
  "latex": "$$\\mathbf{K_y Y}^{(k+1)} = \\mathbf{F} + \\mathbf{B U}^{(k)}$$",
  "description": "State update in quasi-sequential algorithm",
  "section": "4.1",
  "page": "7"
}

{
  "equation_id": "eq16",
  "latex": "$$\\mathbf{K_p P}^{(k+1)} = \\mathbf{M}(\\mathbf{Y}^{(k+1)} - \\mathbf{Y_d})$$",
  "description": "Adjoint update in quasi-sequential algorithm",
  "section": "4.1",
  "page": "7"
}

{
  "equation_id": "eq17",
  "latex": "$$\\mathbf{U}^{(k+1)} = \\mathbf{U}^{(k)} + \\beta_k \\mathbf{d}^{(k)}$$",
  "description": "Control update with step size in quasi-sequential algorithm",
  "section": "4.1",
  "page": "7"
}

{
  "equation_id": "eq18",
  "latex": "$$\\mathbf{d}^{(k)} = -\\nabla J(\\mathbf{U}^{(k)}) = -\\alpha \\mathbf{M_u U}^{(k)} + \\mathbf{B}^T \\mathbf{P}^{(k+1)}$$",
  "description": "Search direction using negative gradient",
  "section": "4.1",
  "page": "7"
}

### Line Search Criteria

{
  "equation_id": "eq19",
  "latex": "$$J(\\mathbf{U}^{(k)} + \\beta_k \\mathbf{d}^{(k)}) \\leq J(\\mathbf{U}^{(k)}) + c_1 \\beta_k (\\nabla J(\\mathbf{U}^{(k)}))^T \\mathbf{d}^{(k)}$$",
  "description": "Armijo condition for sufficient decrease in line search",
  "section": "4.2",
  "page": "8"
}

### Convergence Criterion

{
  "equation_id": "eq20",
  "latex": "$$\\|\\nabla J(\\mathbf{U}^{(k)})\\| < \\varepsilon$$",
  "description": "Termination criterion based on gradient norm",
  "section": "4.3",
  "page": "8"
}

### Reduced Space Form

{
  "equation_id": "eq21",
  "latex": "$$\\hat{J}(\\mathbf{U}) = J(\\mathbf{Y}(\\mathbf{U}), \\mathbf{U})$$",
  "description": "Reduced objective function depending only on control",
  "section": "4.3",
  "page": "9"
}

{
  "equation_id": "eq22",
  "latex": "$$\\nabla \\hat{J}(\\mathbf{U}) = \\alpha \\mathbf{M_u U} - \\mathbf{B}^T \\mathbf{P}(\\mathbf{Y}(\\mathbf{U}))$$",
  "description": "Gradient of reduced objective function",
  "section": "4.3",
  "page": "9"
}

### OCFE Basis Functions

{
  "equation_id": "eq23",
  "latex": "$$\\phi_i(\\mathbf{x}, t) = \\sum_{j=1}^{n_s} \\sum_{k=1}^{n_t} c_{ijk} N_j(\\mathbf{x}) T_k(t)$$",
  "description": "Space-time basis functions as tensor products of spatial and temporal basis functions",
  "section": "3.1",
  "page": "5"
}

{
  "equation_id": "eq24",
  "latex": "$$N_j(\\mathbf{x}) = \\prod_{d=1}^D L_j^d(\\xi_d)$$",
  "description": "Spatial basis functions as products of 1D basis functions",
  "section": "3.1",
  "page": "5"
}

### Error Estimates

{
  "equation_id": "eq25",
  "latex": "$$\\|y - y_h\\|_{L^2(\\Omega_T)} \\leq C h^{p+1} |y|_{H^{p+1}(\\Omega_T)}$$",
  "description": "Error estimate for state variable approximation",
  "section": "5.1",
  "page": "10"
}

{
  "equation_id": "eq26",
  "latex": "$$\\|u - u_h\\|_{L^2(\\Omega_T)} \\leq C h^{p+1} |u|_{H^{p+1}(\\Omega_T)}$$",
  "description": "Error estimate for control variable approximation",
  "section": "5.1",
  "page": "10"
}

{
  "equation_id": "eq27",
  "latex": "$$|J(y,u) - J(y_h,u_h)| \\leq C h^{2(p+1)} (|y|_{H^{p+1}(\\Omega_T)}^2 + |u|_{H^{p+1}(\\Omega_T)}^2)$$",
  "description": "Error estimate for objective function approximation",
  "section": "5.1",
  "page": "10"
}

### Convergence Rate

{
  "equation_id": "eq28",
  "latex": "$$\\|\\mathbf{U}^* - \\mathbf{U}^{(k)}\\| \\leq C q^k$$",
  "description": "Linear convergence rate of quasi-sequential algorithm with rate q < 1",
  "section": "4.4",
  "page": "9"
}
