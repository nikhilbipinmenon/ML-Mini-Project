# Presentation Slide Deck Structure
## UE24CS352A Machine Learning Mini-Project
### Problem Statement #39: Structural Damage Image Classification Using Classical Machine Learning

---

### Slide 1: Title Slide
- **Title**: Structural Damage Image Classification Using Classical Machine Learning
- **Course**: UE24CS352A Machine Learning Mini-Project
- **Problem Statement #**: 39
- **Reference Paper**: Stanford University CS229 (*Structural Damage Image Classification*, Stephanie Ho & Christopher Troncoso, 2017)
- **Team Members**: Nikhil B Menon (PES2UG24CS317) & Niharika SM (PES2UG24CS316)
- **Key Focus**: Strict adherence to classical syllabus models on raw continuous pixel representations without heuristic computer vision descriptors.

---

### Slide 2: Problem Statement & Motivation
- **Real-World Context**: Following catastrophic seismic events, emergency first responders and civil structural engineers must rapidly evaluate infrastructure safety before buildings collapse.
- **The Challenge**: Manual visual inspection is dangerously slow, subjective, and labor-intensive during crisis hours.
- **Project Objective**: Formulate a binary supervised image classification system to accurately predict whether a concrete or masonry structural element is **Damaged (Class 1)** or **Undamaged (Class 0)**.
- **Academic Benchmark Constraint**: Strict evaluation of canonical supervised algorithms on continuous raw pixel matrices ($d = 2,304$), replicating the foundational baseline of Stanford CS229.

---

### Slide 3: Dataset Architecture & Preprocessing
- **Source**: Official Pacific Earthquake Engineering Research (PEER) Center Hub ImageNet ($\phi$-Net Task 2: Damage State), curated by UC Berkeley.
- **Volume**: 13,271 high-resolution images across diverse building structures (beams, columns, masonry walls, foundation slabs).
- **Class Balancing**: Extracted a balanced 50/50 subset of 3,000 images (1,500 Damaged & 1,500 Undamaged) to ensure unbiased sensitivity and specificity.
- **Feature Pipeline**:
  - Downscaled to uniform $48 \times 48$ spatial resolution.
  - Converted to single-channel continuous grayscale.
  - Zero-centered continuous normalization to $[-1.0, 1.0]$ via $x = (x / 127.5) - 1.0$.
  - Flattened into a 2,304-dimensional 1D input vector.
- **Dimensionality Analysis**: 2D PCA demonstrates significant spectral overlap, confirming high intrinsic non-linearity.

---

### Slide 4: Model 1 — Logistic Regression ($L_2$ Regularized)
- **Mathematical Formulation**:
  $$P(Y = 1 \mid X) = \sigma(\theta^T X) = \frac{1}{1 + e^{-\theta^T X}}$$
- **Loss Function**: Convex Cross-Entropy Log-Loss with $L_2$ Ridge Penalty:
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right] + \frac{1}{2C} \|\theta\|_2^2$$
- **Inductive Bias**: Assumes that the log-odds of structural damage vary as a linear combination of raw pixel intensities across a global separating hyperplane.
- **Speaker Note**: *"Logistic Regression examines all 2,304 pixels globally. It penalizes extreme weights via L2 shrinkage, making it robust against high noise but limited when dealing with localized crack geometry."*

---

### Slide 5: Model 2 — Decision Tree (ID3 Algorithm with Shannon Entropy)
- **Mathematical Formulation**:
  - Shannon Entropy:
    $$H(S) = - \sum_{c \in \{0, 1\}} p_c \log_2 p_c$$
  - Information Gain:
    $$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
- **Continuous Thresholding**: Sorts candidate continuous pixel values along feature dimensions and evaluates midpoint cutoffs to maximize mutual information.
- **Regularization**: Pre-pruning via `max_depth = 10` and `min_samples_split = 5` to stop recursive tree expansion and prevent memorization.
- **Inductive Bias**: Assumes decision boundaries are orthogonal, axis-aligned hyperplanes in feature space, focusing on isolated localized pixel thresholds.
- **Speaker Note**: *"Unlike Logistic Regression which sums all pixels, ID3 isolates individual critical pixel coordinates. If a pixel at coordinate (24, 12) is darker than -0.4, it routes down one subtree, isolating local edge discontinuities."*

---

### Slide 6: Model 3 — Support Vector Machine (Soft-Margin RBF Kernel)
- **Mathematical Formulation (Lagrangian Dual)**:
  $$\max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)})$$
  $$\text{Subject to: } 0 \le \alpha_i \le C, \quad \sum_{i=1}^m \alpha_i y^{(i)} = 0$$
- **Kernel Function**: Radial Basis Function (RBF / Gaussian Kernel):
  $$K(x, z) = \exp(-\gamma \|x - z\|^2)$$
- **Inductive Bias**: Projects the 2,304-dimensional space into an infinite-dimensional Reproducing Kernel Hilbert Space (RKHS). Predictions are governed entirely by Euclidean distance to critical boundary support vectors.
- **Speaker Note**: *"SVM with an RBF kernel measures similarity between an unseen structural image and the hardest historical boundary examples. It achieved our highest accuracy (60.0%) and ROC-AUC (0.639)."*

---

### Slide 7: Quantitative Benchmark Results
- Evaluated on holdout test partition:

| Model | Accuracy (%) | Sensitivity / Recall (%) | Specificity (%) | Precision (%) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **SVM (RBF Kernel)** | **60.00%** | **62.33%** | **57.67%** | **59.55%** | **0.6091** | **0.6390** |
| 🥈 **Decision Tree (ID3)** | **55.67%** | **57.67%** | **53.67%** | **55.45%** | **0.5654** | **0.5510** |
| 🥉 **Logistic Regression ($L_2$)** | **54.50%** | **49.00%** | **60.00%** | **55.06%** | **0.5185** | **0.5540** |

---

### Slide 8: Stanford CS229 Paper Replication Analysis
- **Stanford Paper Findings (Paper #39: Ho & Troncoso)**:
  - In Table 1 of CS229, classical models trained directly on raw pixel matrices achieved **50.7% to 54.7% accuracy**.
  - Linear models struggled because raw pixel representations lack spatial translation invariance.
- **Our Experimental Replication**:
  - Our balanced benchmark achieves **54.5% to 60.0% accuracy**, directly validating the Stanford research hypothesis.
  - A fracture shifted by 5 pixels changes all 2,304 numerical coordinates, demonstrating the fundamental theoretical limit of classical models without convolutional weight sharing.

---

### Slide 9: Inductive Biases & Why Models Disagree (Hard Voting)
- **Why do models predict 2-out-of-3 on some images?**
  1. **Logistic Regression**: Linear global sum. Fails when a crack is narrow but background is dark.
  2. **Decision Tree (ID3)**: Axis-aligned threshold. Triggered easily by local shadows or gravel textures.
  3. **SVM (RBF)**: Non-linear Euclidean clustering. Sensitive to overall pixel intensity distributions.
- **Ensemble Consensus**:
  - We implemented a **Hard Voting Ensemble Rule**: $\text{Consensus} = \mathbb{I}\left(\sum_{k=1}^3 \hat{y}_k \ge 2\right)$.
  - Majority consensus resolves individual model failure modes and reduces prediction variance.

---

### Slide 10: Interactive Demonstration (Streamlit Web Dashboard)
- **Features**:
  - Drag-and-drop custom image uploader.
  - Dropdown library of 10 curated PEER Hub benchmark images (5 damaged, 5 undamaged).
  - Side-by-side visualization: Original RGB vs $48 \times 48$ continuous grayscale representation.
  - Live prediction table with confidence percentages for all 3 models.
  - Real-time Matplotlib ROC-AUC curves comparison.

---


