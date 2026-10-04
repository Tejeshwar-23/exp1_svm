# Support Vector Machines — Comprehensive Viva Voce Defense Bank

**Project:** SVM with Different Kernels (Linear, Polynomial, RBF)  
**Assigned Topic:** Topic 1 (Candidate Presentation)  
**Seminar Series:** Team 15 ML Technical Seminar  
**Source of Truth:** [`01_svm_kernel_comparison.py`](file:///c:/Users/Jashwanth/Documents/Projects/ML-Seminar/01_svm_kernel_comparison.py) and [`outputs/svm/svm_results.txt`](file:///c:/Users/Jashwanth/Documents/Projects/ML-Seminar/outputs/svm/svm_results.txt)  

---

## Category 1: Fundamental Concepts & Definitions

### Q1. What is a Support Vector Machine, and what makes it distinct from a standard Perceptron?
**Model Answer:**  
A Support Vector Machine (SVM) is a supervised learning algorithm used primarily for classification and regression.  
While a Rosenblatt Perceptron iteratively searches for *any* hyperplane that separates linearly separable data—often halting at a boundary that passes dangerously close to training samples—SVM searches for the *optimal* hyperplane that maximizes the geometric margin (the distance between the boundary and the closest training points on either side). This principle, known as Structural Risk Minimization (SRM), provides theoretical guarantees against overfitting and maximizes noise tolerance on unseen test data.

### Q2. What are Support Vectors, and what is their mathematical definition?
**Model Answer:**  
Support vectors are the critical subset of training points that lie exactly on the margin boundaries or violate them.  
In the dual formulation of SVM, the optimal weight vector is expressed as:
$$w^* = \sum_{i=1}^n \alpha_i y_i x_i$$
According to the Karush-Kuhn-Tucker (KKT) complementary slackness conditions:
$$\alpha_i [y_i(w^T x_i + b) - 1] = 0$$
For any point strictly outside the margin ($y_i(w^T x_i + b) > 1$), its Lagrange multiplier $\alpha_i$ must equal zero. Only points on the margin boundaries or violating the margin have $\alpha_i > 0$. These points are called support vectors because they physically determine the orientation and position of the hyperplane.

### Q3. What happens to the decision boundary if we add 5,000 extra training samples far away from the margin?
**Model Answer:**  
The decision boundary remains **100% unchanged**.  
Because all 5,000 new points lie strictly outside the margin boundary ($y_i(w^T x_i + b) > 1$), their corresponding dual multipliers are $\alpha_i = 0$. Since $w^*$ is a linear combination of only points with $\alpha_i > 0$, non-support vector points contribute zero weight to the model.

### Q4. What is the difference between Hard-Margin SVM and Soft-Margin SVM?
**Model Answer:**  
- **Hard-Margin SVM**: Assumes the data is strictly linearly separable. It enforces the constraint $y_i(w^T x_i + b) \ge 1$ for every training sample with zero tolerance for errors or margin violations. If the data has even one overlapping sample or outlier, the hard-margin solver fails to find a feasible solution.
- **Soft-Margin SVM** (Cortes & Vapnik, 1995): Introduces non-negative slack variables $\xi_i \ge 0$ to tolerate misclassifications and margin intrusions:
  $$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \ge 1 - \xi_i$$
  Parameter $C > 0$ governs the trade-off between maximizing margin width and penalizing slack violations.

### Q5. What is the geometric interpretation of the slack variable $\xi_i$?
**Model Answer:**  
The slack variable $\xi_i \ge 0$ measures the distance by which point $x_i$ violates the margin boundary:
1. $\xi_i = 0$: The point lies on or outside the correct margin boundary (correctly classified).
2. $0 < \xi_i \le 1$: The point is inside the margin boundary but on the correct side of the decision plane (correctly classified, but violates the margin).
3. $\xi_i > 1$: The point falls on the wrong side of the decision plane (misclassified).

### Q6. How does SVM handle multi-class classification?
**Model Answer:**  
SVM is natively a binary classifier. Multi-class classification is handled through decomposition strategies:
1. **One-vs-Rest (OvR)**: Trains $K$ binary classifiers, where classifier $k$ separates class $k$ from all remaining $K-1$ classes. The query point is assigned to the class with the highest decision value: $\arg\max_k (w_k^T x + b_k)$.
2. **One-vs-One (OvO)**: Trains $\frac{K(K-1)}{2}$ pairwise binary classifiers. The final class is chosen by majority voting. Scikit-learn's `SVC` uses the OvO strategy internally.

---

## Category 2: Mathematical Derivations & Theoretical Foundations

### Q7. Derive why maximizing the margin is mathematically equivalent to minimizing $\frac{1}{2}\|w\|^2$.
**Model Answer:**  
Let the decision plane be $w^T x + b = 0$. The perpendicular Euclidean distance from any sample $x_i$ to this plane is:
$$\text{dist}(x_i) = \frac{y_i(w^T x_i + b)}{\|w\|_2}$$
By scaling $w$ and $b$, we define the canonical hyperplane such that the closest points satisfy:
$$\min_i |w^T x_i + b| = 1$$
The positive bounding plane is $w^T x + b = +1$ and the negative bounding plane is $w^T x + b = -1$.  
Subtracting these two equations and projecting the vector connecting them onto the unit normal vector $\frac{w}{\|w\|}$:
$$\text{Margin Width } M = \frac{(x_+ - x_-)^T w}{\|w\|} = \frac{1 - (-1)}{\|w\|} = \frac{2}{\|w\|}$$
To maximize the margin $M = \frac{2}{\|w\|}$, we invert it to minimize $\frac{\|w\|}{2}$. Squaring and multiplying by $\frac{1}{2}$ produces the equivalent convex quadratic objective:
$$\min_{w, b} \frac{1}{2}\|w\|^2$$
Squaring ensures differentiability at $w=0$ and yields a strictly convex quadratic program with a unique global minimum.

### Q8. Why do we solve SVM in the dual formulation rather than the primal formulation?
**Model Answer:**  
The Lagrangian dual formulation provides three major mathematical advantages:
1. **Dimensional Invariance**: The primal formulation solves for $w \in \mathbb{R}^d$, which depends on the dimensionality of the feature space $d$. In the dual formulation, we solve for $n$ Lagrange multipliers $\alpha \in \mathbb{R}^n$, which depends only on the number of samples $n$.
2. **Inner Product Dependency**: In the dual problem:
   $$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$
   Data vectors appear **exclusively as dot products** $x_i^T x_j$.
3. **Enables the Kernel Trick**: Because data only enters through dot products, we can substitute $x_i^T x_j$ with a non-linear kernel $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$, operating in high- or infinite-dimensional Hilbert spaces without ever computing $\phi(x)$ explicitly.

### Q9. What are the KKT conditions for Soft-Margin SVM?
**Model Answer:**  
For the soft-margin primal problem with Lagrangian multipliers $\alpha_i \ge 0$ (for margin constraints) and $\mu_i \ge 0$ (for slack constraints $\xi_i \ge 0$):
1. **Stationarity**:
   - $\nabla_w \mathcal{L} = 0 \implies w = \sum_{i=1}^n \alpha_i y_i x_i$
   - $\nabla_b \mathcal{L} = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0$
   - $\nabla_{\xi_i} \mathcal{L} = 0 \implies C - \alpha_i - \mu_i = 0 \implies \alpha_i \le C$
2. **Primal Feasibility**:
   - $y_i(w^T x_i + b) \ge 1 - \xi_i$
   - $\xi_i \ge 0$
3. **Dual Feasibility**:
   - $0 \le \alpha_i \le C$
   - $\mu_i \ge 0$
4. **Complementary Slackness**:
   - $\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$
   - $\mu_i \xi_i = (C - \alpha_i)\xi_i = 0$

### Q10. What is the difference between an unbounded support vector and a bounded support vector?
**Model Answer:**  
- **Unbounded (Free) Support Vector ($0 < \alpha_i < C$):**  
  From $(C - \alpha_i)\xi_i = 0$, since $\alpha_i < C$, $\xi_i$ must equal $0$.  
  From $\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$, we have $y_i(w^T x_i + b) = 1$.  
  These points lie **exactly on the margin boundary**. They are free to move along the boundary and are used to compute the bias term $b^*$.
- **Bounded Support Vector ($\alpha_i = C$):**  
  Here, the dual weight hits the upper ceiling $C$. From $(C - \alpha_i)\xi_i = 0$, $\xi_i$ can be strictly positive ($\xi_i > 0$).  
  These points **violate the margin**: they either lie inside the margin slab ($0 < \xi_i \le 1$) or on the wrong side of the decision plane ($\xi_i > 1$, misclassified).

### Q11. How is the intercept term $b^*$ calculated in practice?
**Model Answer:**  
The bias $b^*$ is computed using an unbounded support vector $x_k$ where $0 < \alpha_k < C$ (and thus $\xi_k = 0$).  
Since $y_k(w^T x_k + b) = 1$, and multiplying both sides by $y_k$ (since $y_k \in \{-1, +1\} \implies y_k^2 = 1$):
$$b^* = y_k - w^T x_k = y_k - \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x_k)$$
To ensure numerical stability against floating-point noise, implementations average $b^*$ across all $N_{\text{free}}$ unbounded support vectors:
$$b^* = \frac{1}{|S_{\text{free}}|} \sum_{k \in S_{\text{free}}} \left( y_k - \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x_k) \right)$$

### Q12. What is Cover's Theorem on separability, and why is it fundamental to SVM?
**Model Answer:**  
Formulated by Thomas Cover in 1965, Cover's Theorem states:
> *"A complex pattern-classification problem cast in a high-dimensional space non-linearly is more likely to be linearly separable than in a low-dimensional space, provided the space is not densely populated."*

Mathematically, if $N$ random points in $\mathbb{R}^d$ are partitioned into two classes, the probability of linear separability $P(N, d)$ increases monotonically as the dimensionality $d$ increases relative to $N$. When $d \to \infty$, $P(N, d) \to 1$.  
Cover's theorem provides the theoretical foundation for kernel SVM: by projecting low-dimensional non-separable data into higher-dimensional feature spaces, linear separation becomes achievable.

---

## Category 3: The Kernel Trick & Mercer's Theorem

### Q13. Explain the "Kernel Trick" in simple and rigorous terms.
**Model Answer:**  
- **Simple explanation**: If data cannot be separated in 2D, we project it into 3D or 100D where a flat plane can cut between classes. But calculating coordinates in 100D is slow. The kernel trick is a mathematical shortcut: a function that gives you the exact dot product in 100D space by doing a simple calculation using only the original 2D coordinates.
- **Rigorous mathematical explanation**: The dual SVM optimization and inference functions depend solely on inner products $\langle \phi(x_i), \phi(x_j) \rangle$ in feature space $\mathcal{H}$. A kernel function $K(x_i, x_j)$ directly computes this inner product:
  $$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_{\mathcal{H}}$$
  We completely bypass the explicit calculation of the high-dimensional mapping $\phi(x)$, eliminating the combinatorial computational and memory explosion of operating in $\mathcal{H}$.

### Q14. State Mercer's Theorem and its practical necessity in SVM.
**Model Answer:**  
**Theorem**: A continuous, symmetric function $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ admits an expansion $K(x, z) = \sum_{i=1}^\infty \lambda_i \phi_i(x) \phi_i(z)$ with positive coefficients $\lambda_i > 0$ if and only if for all square-integrable functions $g(x)$:
$$\iint K(x, z) g(x) g(z) \, dx \, dz \ge 0$$
In discrete terms, for any dataset $\{x_1, \dots, x_n\}$, the Gram matrix $K_{ij} = K(x_i, x_j)$ must be **symmetric positive semi-definite (PSD)**.  
**Practical Significance**:  
If $K$ is positive semi-definite, the objective function of the SVM dual quadratic program:
$$f(\alpha) = \frac{1}{2} \alpha^T Q \alpha - \mathbf{1}^T \alpha, \quad \text{where } Q_{ij} = y_i y_j K(x_i, x_j)$$
has a positive semi-definite matrix $Q$. This guarantees that the optimization landscape is **strictly convex**, possessing a single global minimum with zero local minima traps.

### Q15. Prove that the RBF (Gaussian) kernel corresponds to an infinite-dimensional feature space.
**Model Answer:**  
Let the RBF kernel with $\gamma = \frac{1}{2}$ in 1D be:
$$K(x, z) = \exp\left(-\frac{(x - z)^2}{2}\right) = \exp\left(-\frac{x^2}{2}\right) \exp\left(-\frac{z^2}{2}\right) \exp(xz)$$
Using the Taylor series expansion for the exponential function $\exp(u) = \sum_{k=0}^\infty \frac{u^k}{k!}$:
$$\exp(xz) = \sum_{k=0}^\infty \frac{(xz)^k}{k!} = \sum_{k=0}^\infty \left(\frac{x^k}{\sqrt{k!}}\right) \left(\frac{z^k}{\sqrt{k!}}\right)$$
Multiplying by the leading scalar exponentials:
$$K(x, z) = \sum_{k=0}^\infty \left( e^{-x^2/2} \frac{x^k}{\sqrt{k!}} \right) \left( e^{-z^2/2} \frac{z^k}{\sqrt{k!}} \right) = \langle \phi(x), \phi(z) \rangle$$
where the feature mapping is:
$$\phi(x) = e^{-x^2/2} \left[ 1, \frac{x}{\sqrt{1!}}, \frac{x^2}{\sqrt{2!}}, \frac{x^3}{\sqrt{3!}}, \dots, \frac{x^k}{\sqrt{k!}}, \dots \right]^T$$
Because the summation index $k$ runs to infinity, $\phi(x)$ has an **infinite number of dimensions**. The RBF kernel maps data into an infinite-dimensional Hilbert space.

### Q16. Can a Sigmoid kernel be used in SVM? Is it always a Mercer kernel?
**Model Answer:**  
The Sigmoid (Hyperbolic Tangent) kernel is defined as:
$$K(x, z) = \tanh(\alpha x^T z + c)$$
It was historically proposed to make SVM emulate a two-layer Perceptron neural network.  
**However, the Sigmoid kernel is NOT generally a Mercer kernel**. It is only conditionally positive semi-definite for specific values of $\alpha$ and $c$. If parameters violate positive semi-definiteness, the Gram matrix can have negative eigenvalues, rendering the QP non-convex and causing standard solvers to fail to converge.

### Q17. How do you verify if a custom kernel function is valid?
**Model Answer:**  
To verify that a custom function $K(x, z)$ is a valid Mercer kernel:
1. **Symmetry**: Check that $K(x, z) = K(z, x)$ for all $x, z$.
2. **Gram Matrix Positive Semi-Definiteness**: For any arbitrary set of vectors $\{x_1, \dots, x_n\}$, compute the Gram matrix $K_{ij} = K(x_i, x_j)$. Verify that all eigenvalues are non-negative ($\lambda_i \ge 0$), or equivalently that $v^T K v \ge 0$ for all non-zero vectors $v \in \mathbb{R}^n$.
3. **Closure Properties**: Build the kernel using known kernel composition rules:
   - Sum of kernels: $K_1 + K_2$ is valid.
   - Product of kernels: $K_1 \cdot K_2$ is valid.
   - Scaling: $c \cdot K$ ($c > 0$) is valid.
   - Function projection: $f(x) K(x, z) f(z)$ is valid.

### Q18. What is the computational complexity of evaluating a new test point with Linear vs RBF SVM?
**Model Answer:**  
- **Linear SVM**: Prediction requires evaluating $f(x) = \text{sign}(w^T x + b)$. Because $\phi(x) = x$, we pre-calculate the $d$-dimensional vector $w = \sum_{i \in \text{SV}} \alpha_i y_i x_i$ once after training. Evaluating a test point requires a single vector dot product of length $d$: **$O(d)$ time**, completely independent of the number of support vectors $N_{\text{SV}}$.
- **RBF SVM**: In RBF, $w$ lives in infinite dimensions and cannot be explicitly stored as a vector. Predicting a test point requires evaluating:
  $$f(x_{\text{test}}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i \exp(-\gamma \|x_i - x_{\text{test}}\|^2) + b \right)$$
  This requires computing Euclidean distance against all $N_{\text{SV}}$ support vectors: **$O(N_{\text{SV}} \cdot d)$ time**.

---

## Category 4: Hyperparameters & Optimization

### Q19. What is the effect of tuning hyperparameter $C$?
**Model Answer:**  
Parameter $C$ controls the soft-margin trade-off between maximizing margin width and penalizing misclassifications:
- **Small $C$ (e.g. $0.01$)**: The penalty on slack variables $\sum \xi_i$ is weak. The optimizer prioritizes a large margin $M = 2/\|w\|$, tolerating training errors. This introduces **high bias and low variance** (underfitting risk), but provides resilience against heavy noise.
- **Large $C$ (e.g. $1000$)**: The penalty on misclassifications is severe. The optimizer forces $\xi_i \to 0$, creating a narrow margin and bending the decision surface to classify every outlier correctly. This introduces **low bias and high variance** (overfitting risk).

### Q20. What is the effect of tuning hyperparameter $\gamma$ (gamma) in RBF SVM?
**Model Answer:**  
In RBF, $\gamma = \frac{1}{2\sigma^2}$, governing the radius of influence of each support vector:
- **Small $\gamma$ (e.g. $0.001$)**: Represents a large Gaussian variance $\sigma^2$. The Gaussian bell curves are broad and flat. Support vectors exert influence over long distances. The decision boundary becomes smooth and nearly linear, risking **underfitting**.
- **Large $\gamma$ (e.g. $100$)**: Represents a tiny variance $\sigma^2$. The Gaussian peaks are steep and narrow. A support vector only influences points in its immediate vicinity. The decision boundary fragments into isolated circular "islands" around training points, risking **extreme overfitting**.

### Q21. How does scikit-learn calculate `gamma='scale'` and `gamma='auto'`?
**Model Answer:**  
- **`gamma='scale'`** (default in scikit-learn):
  $$\gamma = \frac{1}{n_{\text{features}} \cdot \text{Var}(X)}$$
  Standardizes kernel scale by accounting for both feature dimensionality and sample variance.
- **`gamma='auto'`**:
  $$\gamma = \frac{1}{n_{\text{features}}}$$
  Accounts only for dimensionality, ignoring feature scale differences.

### Q22. What algorithm is used to solve the SVM quadratic program in scikit-learn?
**Model Answer:**  
Scikit-learn uses **Sequential Minimal Optimization (SMO)**, developed by John Platt in 1998, as implemented in the underlying C++ library `libsvm`.  
**How SMO works**:  
Standard QP solvers require storing and inverting an $n \times n$ matrix, requiring $O(n^2)$ memory and $O(n^3)$ time.  
SMO breaks the massive QP problem down into the smallest possible sub-problems: optimizing **exactly two Lagrange multipliers ($\alpha_A, \alpha_B$)** at each iteration while holding all other $\alpha_i$ fixed.  
Because the linear equality constraint $\sum \alpha_i y_i = 0$ must hold, choosing two multipliers reduces the optimization to an analytic 1D line search, which can be solved closed-form without numerical matrix inversion.

### Q23. Why is feature scaling mandatory before training an SVM?
**Model Answer:**  
SVM is fundamentally a **distance-based geometric classifier**:
1. For Linear SVM, margin width is $2/\|w\|_2 = 2 / \sqrt{\sum w_j^2}$. If feature 1 has range $[0, 1000]$ and feature 2 has range $[0, 1]$, the optimizer will set $w_1 \approx 0$ to minimize $\|w\|^2$, effectively discarding feature 2.
2. For RBF SVM, similarity is computed via Euclidean distance $\|x_i - x_j\|^2 = \sum (x_{ik} - x_{jk})^2$. Unscaled features with large numerical magnitudes dominate the Euclidean norm, reducing the other features to numerical noise.  
Standardizing with `StandardScaler` ($\mu=0, \sigma=1$) ensures all dimensions contribute equally to geometric distances.

### Q24. What is the difference between `SVC(kernel='linear')` and `LinearSVC` in scikit-learn?
**Model Answer:**  
- **`SVC(kernel='linear')`**: Backed by `libsvm`. Solves the **dual problem** via SMO. It returns support vector indices, dual coefficients, and scales between $O(n^2)$ and $O(n^3)$ with sample count.
- **`LinearSVC`**: Backed by `liblinear`. Solves the **primal problem** directly using coordinate descent or trust-region Newton methods. It does not compute a kernel matrix and does not retain support vectors. It scales in **$O(n \cdot d)$ linear time**, allowing it to handle millions of samples and hundreds of thousands of features in seconds.

---

## Category 5: Project-Specific Implementation & Empirical Defense

### Q25. State your exact project results across all three kernels.
**Model Answer:**  
From our verified output file `outputs/svm/svm_results.txt` on the held-out test set ($N_{\text{test}} = 150$):

| Kernel | Accuracy | Precision | Recall | F1 Score | Support Vectors ($N_{\text{SV}}$) |
|---|---|---|---|---|---|
| **Linear** | 0.8533 (85.33%) | 0.8630 | 0.8400 | 0.8514 | 117 (33.4% of train) |
| **Polynomial ($d=3$)** | 0.8733 (87.33%) | 0.9242 | 0.8133 | 0.8652 | 118 (33.7% of train) |
| **RBF (Gaussian)** | **0.9467 (94.67%)** | **0.9855** | **0.9067** | **0.9444** | **95 (27.1% of train)** |

### Q26. Explain the "Support Vector Sparsity Paradox": Why did the most complex model (RBF) have the FEWEST support vectors?
**Model Answer:**  
This is the most critical theoretical finding in our project:
- **Intuition**: People assume a non-linear boundary requires more support vectors to memorize complex curves.
- **Reality**: In soft-margin SVM, any sample with slack $\xi_i > 0$ (points inside the margin or misclassified) is a **bounded support vector with $\alpha_i = C$**.
- **The Cause**: The Linear model underfits the two-moons geometry, severing both crescents. Dozens of points near the cut are trapped on the wrong side or inside the margin, artificially inflating the support vector count to 117.
- **The RBF Advantage**: Because RBF naturally aligns with the underlying crescent manifold, points in the interior of each moon are far outside the margin where $\xi_i = 0$ and $\alpha_i = 0$. RBF only requires support vectors along the clean narrow channel between the crescents, producing a genuinely sparser model (95 SVs).

### Q27. Why did the Polynomial kernel ($d=3$) achieve high precision (92.42%) but low recall (81.33%)?
**Model Answer:**  
Polynomial non-linearity is **global**. A cubic curve in 2D space has at most two inflection points. In fitting the upper moon, the cubic polynomial bent downward, but was constrained from bending sharply around the lower crescent.  
To minimize the global penalty, the boundary positioned itself conservatively away from the positive cluster. Consequently:
- It made fewer positive predictions, generating only 5 false positives (boosting precision to 92.42%).
- But it misclassified 14 actual positive points as negative (false negatives), depressing recall to 81.33%.

### Q28. How did you ensure zero data leakage during feature preprocessing?
**Model Answer:**  
In `01_svm_kernel_comparison.py`, we executed:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
The scaler was fitted **strictly on `X_train`**, calculating the empirical mean $\mu$ and standard deviation $\sigma$ from only the 350 training samples. That fitted transformation was then applied to `X_test`.  
If we had called `scaler.fit(X)` on the complete 500 samples prior to `train_test_split`, the mean and variance would include information from the held-out test set, leaking future distribution properties into training and invalidating our experimental evaluation.

### Q29. What was the exact decision boundary plotting resolution and technique?
**Model Answer:**  
In our `plot_decision_boundary` function:
1. We determined the spatial extrema of the scaled features: `[x_min - 0.5, x_max + 0.5]` and `[y_min - 0.5, y_max + 0.5]`.
2. We generated a dense coordinate mesh using `np.meshgrid` with step size `h = 0.02`.
3. We flattened the mesh via `np.c_[xx.ravel(), yy.ravel()]`, creating ~70,000 spatial query points.
4. We passed all 70,000 points into `model.predict()`, reshaped the output to `xx.shape`, and rendered the filled decision regions with `ax.contourf` and the boundary with `ax.contour(..., linestyles='--')`.

### Q30. If you had to deploy this model in an embedded microcontroller with 64 KB RAM, which model would you deploy?
**Model Answer:**  
I would deploy the **Linear SVM** if 85.3% accuracy is acceptable, because its inference requires storing only a single 2D weight vector $w = [w_1, w_2]$ and scalar $b$ (12 bytes of RAM), evaluating in 2 multiply-adds.  
If 94.7% accuracy is mandatory, I would deploy the **RBF SVM**, which requires storing 95 support vector coordinate pairs and 95 dual multipliers:
$$95 \times (2 \text{ features} + 1 \text{ weight}) \times 4 \text{ bytes (float32)} \approx 1.14 \text{ KB of memory}$$
1.14 KB fits easily within a 64 KB microcontroller limit, making our 95-SV RBF model highly viable for edge deployment.

---

## Category 6: Tough Faculty "Curveball" & Advanced Viva Scenarios

### Q31. "Isn't SVM completely obsolete now that we have Deep Learning?"
**Model Answer:**  
No, that is a common misconception. SVMs maintain decisive advantages over deep neural networks in several domains:
1. **Mathematical Convexity**: SVM training has zero local minima traps or saddle points. Deep neural networks solve non-convex objectives where weight initialization and gradient stochasticity produce variable results.
2. **Small to Medium Datasets**: When sample sizes range from $N = 50$ to $N = 5,000$ (e.g. rare disease diagnosis, clinical trials), deep neural networks suffer catastrophic overfitting. SVM maximizes the geometric margin, providing robust generalization bounds independent of sample size.
3. **High-Dimensional Text & Genomics**: In sparse problems where features far exceed samples ($d \gg N$), linear SVMs (`LinearSVC`) train in seconds and achieve state-of-the-art accuracy without requiring hyperparameter tuning of hundreds of thousands of neural weights.

### Q32. "What is the VC Dimension of an RBF SVM, and does an infinite-dimensional feature space imply immediate overfitting?"
**Model Answer:**  
The Vapnik-Chervonenkis (VC) dimension of an RBF kernel SVM with $\gamma > 0$ is **infinite** ($\infty$) if unrestricted. In theory, an RBF kernel can shatter an arbitrarily large number of points by choosing sufficiently large $\gamma$.  
**Why it doesn't immediately overfit**:  
Vapnik's Structural Risk Minimization bounds prove that generalization error is bounded not by the raw dimension of the mapped space, but by the **geometric margin**:
$$h \le \min\left(d, \left\lceil \frac{R^2}{M^2} \right\rceil\right) + 1$$
By regularizing via parameter $C$ and maximizing margin $M = 2/\|w\|$, SVM constrains its effective capacity, achieving strong out-of-sample generalization despite operating in an infinite-dimensional Hilbert space.

### Q33. "Can SVM provide calibrated posterior class probabilities like Logistic Regression?"
**Model Answer:**  
Natively, SVM does **not** provide calibrated probabilities; its decision function $f(x) = w^T \phi(x) + b$ outputs an uncalibrated signed geometric distance.  
To extract probabilities, we use **Platt Scaling** (1999) (enabled in scikit-learn via `SVC(probability=True)`).  
Platt scaling fits a 1D sigmoid function over the SVM decision values using an internal 5-fold cross-validation:
$$P(y=1 \mid x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
where parameters $A$ and $B$ are estimated via maximum likelihood. Note that Platt scaling increases training time by a factor of 5 due to cross-validation.

### Q34. "What happens if two features in your dataset are perfectly collinear ($x_2 = 2 x_1$)?"
**Model Answer:**  
Unlike unregularized Linear Regression where collinearity causes the Gram matrix $X^T X$ to become singular and non-invertible:
SVM remains **completely stable**.  
The $\frac{1}{2}\|w\|^2$ regularization term acts like an $L_2$ Ridge penalty. When $x_2 = 2x_1$, the optimizer simply splits the weight across both features to minimize $w_1^2 + w_2^2$ (specifically setting $w_1 = \frac{1}{5} w_{\text{total}}$ and $w_2 = \frac{2}{5} w_{\text{total}}$). The decision boundary and margin width remain identical.

### Q35. "How would you handle an extreme class imbalance where Class 0 has 9,900 samples and Class 1 has 100 samples?"
**Model Answer:**  
In standard SVM, the objective treats errors equally, meaning the model will favor Class 0 to minimize $\sum \xi_i$.  
To address this:
1. **Weighted Soft-Margin SVM (`class_weight='balanced'`)**: Assign separate penalty parameters $C_0$ and $C_1$:
   $$\min \frac{1}{2}\|w\|^2 + C_0 \sum_{i \in \text{Class 0}} \xi_i + C_1 \sum_{j \in \text{Class 1}} \xi_j$$
   Setting $C_1 = \frac{N_0}{N_1} C_0 = 99 \cdot C_0$ forces the solver to penalize errors on minority Class 1 ninety-nine times more heavily.
2. **Evaluation Metric Shift**: Discard Accuracy (which gives 99% for a trivial majority classifier) and evaluate using PR-AUC (Precision-Recall AUC) or Balanced F1.

### Q36. "If the Gram matrix $K$ has rank $r < n$, does the dual optimization problem still have a unique solution?"
**Model Answer:**  
- If $K$ has rank $r < n$, the quadratic term $\frac{1}{2} \alpha^T Q \alpha$ is positive semi-definite but not strictly positive definite.
- **Weight Vector $w^*$**: The optimal hyperplane normal $w^* = \sum \alpha_i y_i \phi(x_i)$ is **always unique** because the primal problem is strictly convex in $w$.
- **Dual Multipliers $\alpha^*$**: The optimal dual vector $\alpha^*$ may not be strictly unique; multiple combinations of $\alpha_i$ may yield the identical $w^*$ vector. However, all optimal dual solutions produce the exact same geometric decision boundary and identical margin width.
