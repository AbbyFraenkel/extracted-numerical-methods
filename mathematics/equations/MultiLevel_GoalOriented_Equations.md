# Equation Database: Multi-Level Automatic Goal-Oriented Method

## Source Paper
"A painless multi-level automatic goal-oriented"

## Equations

### PDE Problem Definition

{
  "equation_id": "eq1",
  "latex": "$$\\mathcal{L}u = f \\quad \\text{in } \\Omega$$",
  "description": "Abstract form of the partial differential equation",
  "section": "2.1",
  "page": "3"
}

{
  "equation_id": "eq2",
  "latex": "$$\\mathcal{B}u = g \\quad \\text{on } \\partial\\Omega$$",
  "description": "Boundary conditions for the PDE",
  "section": "2.1",
  "page": "3"
}

### Quantity of Interest

{
  "equation_id": "eq3",
  "latex": "$$J(u) = \\int_{\\Omega} j(u) \\, d\\Omega + \\int_{\\partial\\Omega} g(u) \\, d\\Gamma$$",
  "description": "General form of a quantity of interest functional",
  "section": "2.2",
  "page": "4"
}

### Weak Form

{
  "equation_id": "eq4",
  "latex": "$$a(u, v) = l(v) \\quad \\forall v \\in V$$",
  "description": "Weak form of the PDE problem",
  "section": "2.3",
  "page": "5"
}

### Discrete Problem

{
  "equation_id": "eq5",
  "latex": "$$a(u_h, v_h) = l(v_h) \\quad \\forall v_h \\in V_h$$",
  "description": "Discretized weak form on finite-dimensional space V_h",
  "section": "2.3",
  "page": "5"
}

### Adjoint Problem

{
  "equation_id": "eq6",
  "latex": "$$a(v, z) = J'(u)(v) \\quad \\forall v \\in V$$",
  "description": "Weak form of the adjoint problem",
  "section": "3.1",
  "page": "6"
}

### Discrete Adjoint Problem

{
  "equation_id": "eq7",
  "latex": "$$a(v_h, z_h) = J'(u_h)(v_h) \\quad \\forall v_h \\in V_h$$",
  "description": "Discretized adjoint problem",
  "section": "3.1",
  "page": "6"
}

### Error Representation

{
  "equation_id": "eq8",
  "latex": "$$J(u) - J(u_h) = a(u - u_h, z) = l(z) - a(u_h, z)$$",
  "description": "Error representation formula using adjoint solution",
  "section": "3.2",
  "page": "7"
}

### Dual-Weighted Residual

{
  "equation_id": "eq9",
  "latex": "$$J(u) - J(u_h) \\approx \\sum_{K \\in \\mathcal{T}_h} \\eta_K$$",
  "description": "Error estimation as sum of element-wise indicators",
  "section": "3.2",
  "page": "7"
}

{
  "equation_id": "eq10",
  "latex": "$$\\eta_K = r_K(u_h)(z - z_h) + r_{\\partial K}(u_h)(z - z_h)$$",
  "description": "Error indicator for element K using residuals and adjoint",
  "section": "3.3",
  "page": "8"
}

### Multi-Level Solution

{
  "equation_id": "eq11",
  "latex": "$$u^L = u^0 + \\sum_{l=1}^{L} e^l$$",
  "description": "Multi-level solution with corrections",
  "section": "4.1",
  "page": "9"
}

{
  "equation_id": "eq12",
  "latex": "$$a(e^l, v_h^l) = l(v_h^l) - a(u^{l-1}, v_h^l) \\quad \\forall v_h^l \\in V_h^l$$",
  "description": "Equation for computing the correction at level l",
  "section": "4.1",
  "page": "9"
}

### Multi-Level Error Estimation

{
  "equation_id": "eq13",
  "latex": "$$J(u) - J(u^L) \\approx l(z^L) - a(u^L, z^L)$$",
  "description": "Error estimation for multi-level solution",
  "section": "4.2",
  "page": "10"
}

### Dörfler Marking Strategy

{
  "equation_id": "eq14",
  "latex": "$$\\sum_{K \\in \\mathcal{M}} |\\eta_K| \\geq \\theta \\sum_{K \\in \\mathcal{T}_h} |\\eta_K|$$",
  "description": "Dörfler (bulk) marking criterion for selecting elements",
  "section": "3.3",
  "page": "8"
}

### Effectivity Index

{
  "equation_id": "eq15",
  "latex": "$$I_{\\text{eff}} = \\frac{\\text{estimated error}}{\\text{true error}} = \\frac{\\sum_{K \\in \\mathcal{T}_h} \\eta_K}{J(u) - J(u_h)}$$",
  "description": "Effectivity index for evaluating error estimator quality",
  "section": "5.1",
  "page": "11"
}

### Convergence Rate

{
  "equation_id": "eq16",
  "latex": "$$|J(u) - J(u_h)| \\leq C h^{2p} |u|_{p+1} |z|_{p+1}$$",
  "description": "Theoretical error convergence rate for smooth solutions",
  "section": "5.2",
  "page": "12"
}
