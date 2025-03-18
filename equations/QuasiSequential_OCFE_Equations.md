# Equation Database: Quasi-Sequential Algorithm for PDE-Constrained Optimization

## Source Paper
"A quasi-sequential algorithm for PDE-constrained optimization based on space-time OCFE"

## Optimization Problem Formulation

{
  "equation_id": "eq1",
  "latex": "$$\\min_{y, u} \\quad J(y, u)$$",
  "description": "Objective function minimization",
  "section": "Problem Formulation",
  "page": "N/A"
}

{
  "equation_id": "eq2",
  "latex": "$$F(y, u) = 0$$",
  "description": "PDE constraints",
  "section": "Problem Formulation",
  "page": "N/A"
}

{
  "equation_id": "eq3",
  "latex": "$$y(x, 0) = y_0(x)$$",
  "description": "Initial conditions",
  "section": "Problem Formulation",
  "page": "N/A"
}

{
  "equation_id": "eq4",
  "latex": "$$B(y, u) = 0$$",
  "description": "Boundary conditions",
  "section": "Problem Formulation",
  "page": "N/A"
}

{
  "equation_id": "eq5",
  "latex": "$$u_{\\min} \\leq u \\leq u_{\\max}$$",
  "description": "Control constraints",
  "section": "Problem Formulation",
  "page": "N/A"
}

{
  "equation_id": "eq6",
  "latex": "$$g(y, u) \\leq 0$$",
  "description": "Additional constraints",
  "section": "Problem Formulation",
  "page": "N/A"
}

## Space-Time Discretization

{
  "equation_id": "eq7",
  "latex": "$$\\frac{\\partial y}{\\partial t} = \\mathcal{L}(y, u, t, x)$$",
  "description": "General form of time-dependent PDE",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

{
  "equation_id": "eq8",
  "latex": "$$\\Omega \\times [0, T] \\approx \\bigcup_{i=1}^{N_s} \\bigcup_{j=1}^{N_t} \\Omega_i \\times [t_{j-1}, t_j]$$",
  "description": "Space-time domain discretization",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

{
  "equation_id": "eq9",
  "latex": "$$y(x, t) \\approx \\sum_{k=1}^{N_x} \\sum_{l=1}^{N_t} c_{kl} \\phi_k(x) \\psi_l(t)$$",
  "description": "Basis function representation within space-time element",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

{
  "equation_id": "eq10",
  "latex": "$$\\frac{\\partial y(x_p, t_q)}{\\partial t} - \\mathcal{L}(y(x_p, t_q), u(x_p, t_q), t_q, x_p) = 0$$",
  "description": "Residual equations at collocation points",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

{
  "equation_id": "eq11",
  "latex": "$$y(x_i^+, t) = y(x_i^-, t)$$",
  "description": "Spatial continuity constraint",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

{
  "equation_id": "eq12",
  "latex": "$$y(x, t_j^+) = y(x, t_j^-)$$",
  "description": "Temporal continuity constraint",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

{
  "equation_id": "eq13",
  "latex": "$$F(\\mathbf{c}, \\mathbf{u}) = \\mathbf{0}$$",
  "description": "Combined residual and constraint system",
  "section": "Space-Time OCFE",
  "page": "N/A"
}

## Quasi-Sequential Optimization

{
  "equation_id": "eq14",
  "latex": "$$u_{k+1} = u_k + \\alpha_k \\Delta u_k$$",
  "description": "Control update formula",
  "section": "Quasi-Sequential Algorithm",
  "page": "N/A"
}

{
  "equation_id": "eq15",
  "latex": "$$\\nabla_u J(y_k, u_k) = \\frac{\\partial J}{\\partial u}(y_k, u_k) - \\lambda_k^T \\frac{\\partial F}{\\partial u}(y_k, u_k)$$",
  "description": "Gradient computation using adjoint approach",
  "section": "Gradient Computation",
  "page": "N/A"
}

{
  "equation_id": "eq16",
  "latex": "$$\\left(\\frac{\\partial F}{\\partial y}(y_k, u_k)\\right)^T \\lambda_k = -\\frac{\\partial J}{\\partial y}(y_k, u_k)$$",
  "description": "Adjoint equation",
  "section": "Adjoint Problem",
  "page": "N/A"
}

## Error Analysis

{
  "equation_id": "eq17",
  "latex": "$$\\|y - y_h\\|_{\\infty} \\sim O(e^{-\\alpha N})$$",
  "description": "Spectral convergence for smooth solutions",
  "section": "Convergence Properties",
  "page": "N/A"
}

{
  "equation_id": "eq18",
  "latex": "$$\\|y - y_h\\|_{H^1(\\Omega \\times [0,T])} \\leq C(h_s^p + h_t^q)$$",
  "description": "Error bound for space-time OCFE",
  "section": "Error Estimates",
  "page": "N/A"
}

{
  "equation_id": "eq19",
  "latex": "$$J(y_k, u_k) - J(y^*, u^*) \\leq C \\|u_k - u^*\\|^2$$",
  "description": "Convergence rate of objective function",
  "section": "Convergence Properties",
  "page": "N/A"
}

{
  "equation_id": "eq20",
  "latex": "$$\\int_{\\Omega} y(x, T) dx - \\int_{\\Omega} y(x, 0) dx = \\int_0^T \\int_{\\Omega} S(x, t) dx dt$$",
  "description": "Conservation property preservation",
  "section": "Mathematical Properties",
  "page": "N/A"
}
