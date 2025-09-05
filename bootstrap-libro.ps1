# bootstrap-libro.ps1
# Crea chapters/*.tex con secciones y el indice.tex que los incluye.

# Asegura carpeta
$chapDir = "chapters"
if (!(Test-Path $chapDir)) { New-Item -ItemType Directory -Path $chapDir | Out-Null }

# Helper para escribir archivo con contenido
function Write-Tex($path, $text) {
  $dir = Split-Path $path
  if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  Set-Content -Path $path -Value $text -Encoding UTF8
}

# ============ Definición de capítulos ============

$chapters = @(
  # PART I – Mathematical Preliminaries
  @{file="ch01_introduction.tex"; title="Introduction"; sections=@(
    "Historical context of AI and neural networks",
    "AI winters and the deep learning revolution",
    "Why mathematics matters in deep learning"
  )},
  @{file="ch02_linear_algebra.tex"; title="Linear Algebra Essentials"; sections=@(
    "Vector spaces and inner products",
    "Eigenvalues, eigenvectors, diagonalization",
    "Singular value decomposition (SVD)",
    "Tensor notation and operations"
  )},
  @{file="ch03_calculus.tex"; title="Multivariable Calculus and Analysis"; sections=@(
    "Gradients, Jacobians, Hessians",
    "Taylor expansions in multiple variables",
    "Divergence, curl, and Laplacians",
    "Variational principles"
  )},
  @{file="ch04_prob_info.tex"; title="Probability, Statistics, and Information Theory"; sections=@(
    "Random variables and distributions",
    "Expectation, variance, covariance",
    "Gaussian and exponential families",
    "Entropy, KL divergence, mutual information"
  )},
  @{file="ch05_optimization.tex"; title="Optimization Theory"; sections=@(
    "Convexity, duality, and Lagrangians",
    "Gradient descent and its variants",
    "Stochastic optimization and convergence",
    "Newton and quasi-Newton methods"
  )},
  @{file="ch06_functional_analysis.tex"; title="Functional Analysis Foundations"; sections=@(
    "Normed spaces, Banach and Hilbert spaces",
    "Orthogonal polynomials (Hermite, Laguerre, Legendre)",
    "Special functions (Gamma, Beta, Bessel)",
    "Operator theory foundations for PINNs \\& QINNs"
  )},

  # PART II – Classical Neural Networks
  @{file="ch07_perceptron.tex"; title="The Perceptron and Linear Models"; sections=@(
    "McCulloch–Pitts neurons",
    "Rosenblatt’s perceptron and linear separability",
    "Logistic regression as probabilistic perceptron"
  )},
  @{file="ch08_mlps.tex"; title="Feedforward Networks (MLPs)"; sections=@(
    "Activation functions (ReLU, sigmoid, tanh, GELU, softmax)",
    "Universal Approximation Theorem",
    "Forward and backward propagation",
    "Vanishing and exploding gradients"
  )},
  @{file="ch09_cnns.tex"; title="Convolutional Neural Networks (CNNs)"; sections=@(
    "Mathematical basis of convolution",
    "Feature maps, receptive fields, pooling",
    "Modern CNN architectures (AlexNet, VGG, ResNet)",
    "Applications in vision, audio, and physics"
  )},
  @{file="ch10_rnns.tex"; title="Recurrent Neural Networks (RNNs)"; sections=@(
    "Sequences as dynamical systems",
    "Gradient vanishing and exploding",
    "LSTMs and GRUs",
    "Applications in NLP, speech, time-series"
  )},
  @{file="ch11_autoencoders.tex"; title="Autoencoders and Representation Learning"; sections=@(
    "Linear autoencoders and PCA",
    "Nonlinear autoencoders",
    "Variational Autoencoders (VAEs)",
    "Latent space geometry"
  )},
  @{file="ch12_gnns.tex"; title="Graph Neural Networks (GNNs)"; sections=@(
    "Graph Laplacians and spectral methods",
    "Message passing frameworks",
    "Applications in chemistry, materials, biology"
  )},

  # PART III – Neural Networks for Differential Equations
  @{file="ch13_math_methods_pdes.tex"; title="Mathematical Methods for Differential Equations"; sections=@(
    "Classification of ODEs and PDEs",
    "Boundary and initial conditions",
    "Separation of variables",
    "Sturm–Liouville problems and orthogonal expansions",
    "Fourier and Laplace transforms",
    "Spectral methods (Chebyshev, Legendre)",
    "Galerkin and Finite Element Methods (FEM)",
    "Method of Frobenius and special functions"
  )},
  @{file="ch14_pinns.tex"; title="Physics-Informed Neural Networks (PINNs)"; sections=@(
    "Embedding PDEs into loss functions",
    "Collocation and weak formulations",
    "Elliptic, parabolic, and hyperbolic PDEs",
    "Applications: fluids, electromagnetism, quantum mechanics",
    "Extensions: XPINNs, VPINNs, Bayesian PINNs",
    "Inverse problems (intro to iPINNs)"
  )},
  @{file="ch15_ipinns.tex"; title="Inverse Physics-Informed Neural Networks (iPINNs)"; sections=@(
    "Motivation: the role of inverse problems",
    "Formulation with unknown coefficients",
    "Loss functions for parameter estimation",
    "Applications: heat, waves, Schrödinger, materials",
    "Ill-posedness and regularization",
    "Sensitivity to noise and data quality"
  )},
  @{file="ch16_qinns.tex"; title="Quantum Neural Networks (QINNs)"; sections=@(
    "Hilbert spaces and Dirac notation",
    "Quantum perceptron and gates as layers",
    "Variational quantum circuits (VQE, QAOA)",
    "Quantum Boltzmann Machines, QCNNs, Quantum Reservoirs",
    "Parameter-shift rule for gradients",
    "Challenges: barren plateaus, NISQ hardware",
    "Applications in optimization, chemistry, cryptography"
  )},
  @{file="ch17_neural_operators.tex"; title="Neural Operators and DeepONets"; sections=@(
    "Learning operators between function spaces",
    "Comparison with PINNs, iPINNs, FEM",
    "Applications in PDEs and scientific computing"
  )},

  # PART IV – Reinforcement Learning
  @{file="ch18_classical_rl.tex"; title="Classical Reinforcement Learning"; sections=@(
    "Agents, environments, states, actions, rewards",
    "Markov Decision Processes (MDPs)",
    "Value functions and Bellman equations",
    "Tabular methods: SARSA, Q-learning"
  )},
  @{file="ch19_deep_rl.tex"; title="Deep Reinforcement Learning"; sections=@(
    "Deep Q-Networks (DQN)",
    "Policy gradient methods (REINFORCE, PPO)",
    "Actor–Critic architectures (A2C, A3C)",
    "Landmark systems: AlphaGo, AlphaZero, MuZero"
  )},

  # PART V – Modern Architectures
  @{file="ch20_generative_models.tex"; title="Generative Models"; sections=@(
    "GANs and minimax optimization",
    "Wasserstein GANs, StyleGAN",
    "Diffusion models and stochastic processes",
    "Applications in synthesis and design"
  )},
  @{file="ch21_transformers.tex"; title="Transformers and Attention Mechanisms"; sections=@(
    "Self-attention: queries, keys, values",
    "Multi-head attention",
    "Positional encodings",
    "Transformer architectures: BERT, GPT, multimodal",
    "Applications in PDEs and symbolic regression"
  )},

  # PART VI – Advanced Topics
  @{file="ch22_beyond_gd.tex"; title="Optimization Beyond Gradient Descent"; sections=@(
    "Variational inference",
    "Expectation-Maximization (EM)",
    "Federated optimization challenges"
  )},
  @{file="ch23_math_frontiers.tex"; title="Mathematical Frontiers of Neural Networks"; sections=@(
    "Neural Tangent Kernels (NTK)",
    "Infinite-width limits and mean-field theory",
    "Geometry of loss landscapes",
    "Generalization bounds and capacity"
  )},
  @{file="ch24_meta_transfer.tex"; title="Meta-Learning and Transfer Learning"; sections=@(
    "Few-shot learning",
    "Pretraining and fine-tuning",
    "Continual learning"
  )},
  @{file="ch25_explainability.tex"; title="Explainability and Interpretability"; sections=@(
    "Saliency maps and Grad-CAM",
    "SHAP and LIME",
    "Interpretable PINNs and DRL policies"
  )},
  @{file="ch26_ethics.tex"; title="Ethical and Societal Aspects"; sections=@(
    "Bias and fairness in AI",
    "Privacy and security",
    "AI regulation and governance"
  )},

  # PART VII – Practical Implementation
  @{file="ch27_frameworks.tex"; title="Computational Frameworks"; sections=@(
    "PyTorch fundamentals",
    "TensorFlow and Keras",
    "JAX and differentiable programming"
  )},
  @{file="ch28_scaling.tex"; title="Efficient Training and Scaling"; sections=@(
    "Hardware acceleration: GPUs, TPUs",
    "Parallelization and distributed training",
    "Memory-efficient backpropagation"
  )},
  @{file="ch29_case_studies.tex"; title="Case Studies in Scientific Machine Learning"; sections=@(
    "Navier–Stokes with PINNs",
    "QINNs for quantum chemistry",
    "DRL for robotics and control",
    "CNNs/GNNs for materials science"
  )}
)

# ============ Genera cada capítulo .tex ============
foreach ($c in $chapters) {
  $path = Join-Path $chapDir $c.file
  $body = "\\chapter{$($c.title)}`r`n"
  foreach ($s in $c.sections) { $body += "\\section{$s}`r`n% TODO`r`n`r`n" }
  Write-Tex $path $body
}

# ============ Genera indice.tex con \part + \include ============
$indice = @"
% indice.tex (generado)
\part{Mathematical Preliminaries}
\include{chapters/ch01_introduction}
\include{chapters/ch02_linear_algebra}
\include{chapters/ch03_calculus}
\include{chapters/ch04_prob_info}
\include{chapters/ch05_optimization}
\include{chapters/ch06_functional_analysis}

\part{Classical Neural Networks}
\include{chapters/ch07_perceptron}
\include{chapters/ch08_mlps}
\include{chapters/ch09_cnns}
\include{chapters/ch10_rnns}
\include{chapters/ch11_autoencoders}
\include{chapters/ch12_gnns}

\part{Neural Networks for Differential Equations}
\include{chapters/ch13_math_methods_pdes}
\include{chapters/ch14_pinns}
\include{chapters/ch15_ipinns}
\include{chapters/ch16_qinns}
\include{chapters/ch17_neural_operators}

\part{Reinforcement Learning}
\include{chapters/ch18_classical_rl}
\include{chapters/ch19_deep_rl}

\part{Modern Architectures}
\include{chapters/ch20_generative_models}
\include{chapters/ch21_transformers}

\part{Advanced Topics}
\include{chapters/ch22_beyond_gd}
\include{chapters/ch23_math_frontiers}
\include{chapters/ch24_meta_transfer}
\include{chapters/ch25_explainability}
\include{chapters/ch26_ethics}

\part{Practical Implementation}
\include{chapters/ch27_frameworks}
\include{chapters/ch28_scaling}
\include{chapters/ch29_case_studies}
"@
Write-Tex "indice.tex" $indice

Write-Host "Listo. Se crearon 'indice.tex' y $(($chapters | Measure-Object).Count) archivos en 'chapters/'. Compila 'main.tex'."
