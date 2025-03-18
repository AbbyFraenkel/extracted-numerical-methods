# Equation Database: Refinement-by-Superposition Method

## Source Paper
"A Refinement-by-Superposition Method for curl- and div-Conforming Discretizations"

## Equations

### Function Spaces and De Rham Complex

{
  "equation_id": "eq1",
  "latex": "$$H^1(\\Omega) \\xrightarrow{\\nabla} H(\\text{curl}; \\Omega) \\xrightarrow{\\nabla \\times} H(\\text{div}; \\Omega) \\xrightarrow{\\nabla \\cdot} L^2(\\Omega)$$",
  "description": "De Rham complex showing relationship between function spaces",
  "section": "2.1",
  "page": "3"
}

### Curl-Conforming Space

{
  "equation_id": "eq2",
  "latex": "$$H(\\text{curl}; \\Omega) = \\{\\mathbf{v} \\in L^2(\\Omega)^3 : \\nabla \\times \\mathbf{v} \\in L^2(\\Omega)^3\\}$$",
  "description": "Definition of the H(curl) function space",
  "section": "2.1",
  "page": "3"
}

### Div-Conforming Space

{
  "equation_id": "eq3",
  "latex": "$$H(\\text{div}; \\Omega) = \\{\\mathbf{v} \\in L^2(\\Omega)^3 : \\nabla \\cdot \\mathbf{v} \\in L^2(\\Omega)\\}$$",
  "description": "Definition of the H(div) function space",
  "section": "2.1",
  "page": "3"
}

### Discrete De Rham Complex

{
  "equation_id": "eq4",
  "latex": "$$V_h^1 \\xrightarrow{\\nabla} V_h^{\\text{curl}} \\xrightarrow{\\nabla \\times} V_h^{\\text{div}} \\xrightarrow{\\nabla \\cdot} V_h^0$$",
  "description": "Discrete De Rham complex with finite-dimensional subspaces",
  "section": "2.2",
  "page": "4"
}

### Commuting Diagram Property

{
  "equation_id": "eq5",
  "latex": "$$\\begin{array}{ccc} 
H^1(\\Omega) & \\xrightarrow{\\nabla} & H(\\text{curl}; \\Omega) \\\\
\\downarrow \\Pi_h^1 & & \\downarrow \\Pi_h^{\\text{curl}} \\\\
V_h^1 & \\xrightarrow{\\nabla} & V_h^{\\text{curl}} 
\\end{array}$$",
  "description": "Commuting diagram showing consistency between continuous and discrete operators",
  "section": "2.2",
  "page": "4"
}

### Curl-Conforming Element Basis Functions

{
  "equation_id": "eq6",
  "latex": "$$\\mathbf{w}_i(\\mathbf{x}) = \\sum_{j=1}^{n_e} c_{ij} \\mathbf{N}_j(\\mathbf{x})$$",
  "description": "Edge basis functions for curl-conforming elements",
  "section": "3.1",
  "page": "5"
}

### Div-Conforming Element Basis Functions

{
  "equation_id": "eq7",
  "latex": "$$\\mathbf{w}_i(\\mathbf{x}) = \\sum_{j=1}^{n_f} c_{ij} \\mathbf{F}_j(\\mathbf{x})$$",
  "description": "Face basis functions for div-conforming elements",
  "section": "3.1",
  "page": "5"
}

### Tangential Continuity Constraint

{
  "equation_id": "eq8",
  "latex": "$$\\mathbf{n} \\times \\mathbf{u}|_{\\partial K_1} = \\mathbf{n} \\times \\mathbf{u}|_{\\partial K_2}$$",
  "description": "Tangential continuity constraint for curl-conforming elements across element boundaries",
  "section": "3.2",
  "page": "6"
}

### Normal Continuity Constraint

{
  "equation_id": "eq9",
  "latex": "$$\\mathbf{n} \\cdot \\mathbf{u}|_{\\partial K_1} = \\mathbf{n} \\cdot \\mathbf{u}|_{\\partial K_2}$$",
  "description": "Normal continuity constraint for div-conforming elements across element boundaries",
  "section": "3.2",
  "page": "6"
}

### Superposition Basis Construction

{
  "equation_id": "eq10",
  "latex": "$$\\varphi^S_i(\\mathbf{x}) = \\sum_{j=1}^{n_b} w_{ij} \\varphi_j(\\mathbf{x})$$",
  "description": "Construction of superposition basis functions as linear combinations of standard basis",
  "section": "4.1",
  "page": "7"
}

### Hierarchical Basis Decomposition

{
  "equation_id": "eq11",
  "latex": "$$V_h = V_0 \\oplus V_1 \\oplus \\ldots \\oplus V_L$$",
  "description": "Hierarchical decomposition of function space into levels",
  "section": "4.2",
  "page": "8"
}

### Constraint Equation for DOFs

{
  "equation_id": "eq12",
  "latex": "$$u_i = \\sum_{j \\in \\mathcal{J}} c_{ij} u_j$$",
  "description": "Constraint equation relating dependent and independent degrees of freedom",
  "section": "4.3",
  "page": "9"
}

### System Matrix Transformation

{
  "equation_id": "eq13",
  "latex": "$$\\tilde{A} = C^T A C$$",
  "description": "Transformation of system matrix using constraint matrix",
  "section": "4.3",
  "page": "9"
}

### Error Estimate for Curl-Conforming Superposition

{
  "equation_id": "eq14",
  "latex": "$$\\|\\mathbf{u} - \\mathbf{u}_h\\|_{H(\\text{curl})} \\leq C h^p |\\mathbf{u}|_{H^{p+1}(\\text{curl})}$$",
  "description": "Error estimate for curl-conforming approximation with superposition refinement",
  "section": "5.1",
  "page": "10"
}

### Error Estimate for Div-Conforming Superposition

{
  "equation_id": "eq15",
  "latex": "$$\\|\\mathbf{u} - \\mathbf{u}_h\\|_{H(\\text{div})} \\leq C h^p |\\mathbf{u}|_{H^{p+1}(\\text{div})}$$",
  "description": "Error estimate for div-conforming approximation with superposition refinement",
  "section": "5.1",
  "page": "10"
}

### Exactness Property Verification

{
  "equation_id": "eq16",
  "latex": "$$\\nabla \\times (\\nabla \\phi_h) = \\mathbf{0}, \\quad \\nabla \\cdot (\\nabla \\times \\mathbf{v}_h) = 0$$",
  "description": "Exactness properties that must be maintained by discrete operators",
  "section": "5.2",
  "page": "11"
}
