# SVM Kernel Comparison — Timed 7-Minute Live Demo Script

**Project:** Support Vector Machines with Different Kernels  
**Script:** [`01_svm_kernel_comparison.py`](file:///c:/Users/Jashwanth/Documents/Projects/ML-Seminar/01_svm_kernel_comparison.py)  
**Presenter:** Candidate (Team 15 Topic 1)  
**Total Demo Time:** 7 Minutes (420 Seconds)  

---

## Live Demo Pacing Roadmap

```
0:00 ────────────────── 1:00 ── Environment Setup & Terminal Verification
1:00 ────────────────── 2:30 ── Live Code Execution & Terminal Log Analysis
2:30 ────────────────── 4:15 ── High-Resolution Visual Inspection of Decision Surfaces
4:15 ────────────────── 5:45 ── Live Active Learning & Parameter Sensitivity Test
5:45 ────────────────── 7:00 ── Technical Synthesis & Seamless Transition to Viva
```

---

## Segment 1: Environment Setup & Pre-Execution Verification (0:00 – 1:00)

### Presenter Action
1. Open PowerShell / Command Prompt in the project workspace:
   ```powershell
   cd c:\Users\Jashwanth\Documents\Projects\ML-Seminar
   ```
2. Verify Python version and environment dependencies:
   ```powershell
   python --version
   ```

### Verbatim Presenter Script
> *"Respected faculty members, I will now transition from theory to the live practical demonstration of our SVM Kernel Comparison project.*  
> *Before executing, I want to verify our execution environment: we are running Python 3.13 with scikit-learn 1.6.*  
> *Our goal during this demonstration is threefold:*  
> *First, execute our Python implementation `01_svm_kernel_comparison.py` in real-time.*  
> *Second, verify that our empirical metric outputs match the theoretical derivations presented on our slides.*  
> *Third, visually inspect the reconstructed decision surfaces to understand why the RBF kernel achieves 94.67% accuracy with only 95 support vectors."*

---

## Segment 2: Live Code Execution & Output Verification (1:00 – 2:30)

### Presenter Action
1. Execute the main Python script in the terminal:
   ```powershell
   python 01_svm_kernel_comparison.py
   ```
2. Allow the script to run (takes ~1.8 seconds).

### Expected Terminal Output
```text
============================================================
EXPERIMENT 1 — SVM KERNEL COMPARISON
============================================================

Dataset: make_moons
  Total samples:    500
  Noise:            0.25
  Training samples: 350
  Testing samples:  150

Preprocessing: StandardScaler applied

Model Parameters: C=1.0, gamma='scale', degree=3
────────────────────────────────────────────────────────────

  Linear SVM:
    Accuracy:          0.8533
    Precision:         0.8630
    Recall:            0.8400
    F1 Score:          0.8514
    Support Vectors:   117

  Polynomial (d=3) SVM:
    Accuracy:          0.8733
    Precision:         0.9242
    Recall:            0.8133
    F1 Score:          0.8652
    Support Vectors:   118

  RBF SVM:
    Accuracy:          0.9467
    Precision:         0.9855
    Recall:            0.9067
    F1 Score:          0.9444
    Support Vectors:   95

────────────────────────────────────────────────────────────

Kernel                 Accuracy  Precision     Recall         F1
────────────────────────────────────────────────────────────
Linear                   0.8533     0.8630     0.8400     0.8514
Polynomial (d=3)         0.8733     0.9242     0.8133     0.8652
RBF                      0.9467     0.9855     0.9067     0.9444
────────────────────────────────────────────────────────────

✓ Saved: outputs\svm\svm_decision_boundaries.png
✓ Saved: outputs\svm\svm_metrics_comparison.png
✓ Saved: outputs\svm\svm_results.txt

════════════════════════════════════════════════════════════
EXPERIMENT 1 COMPLETE
════════════════════════════════════════════════════════════
```

### Verbatim Presenter Script
> *"As you can see, the script executed completely in under two seconds.*  
> *Let us examine the live console output:*  
> *Notice that 500 two-moons instances were synthesized and partitioned into exactly 350 training samples and 150 held-out test samples.*  
> *Look at the quantitative metric scorecard printed on the console:*  
> *- The Linear model achieves 85.33% accuracy and requires 117 support vectors.*  
> *- The Polynomial degree-3 model achieves 87.33% accuracy with 118 support vectors.*  
> *- The RBF kernel achieves 94.67% accuracy, 98.55% precision, and requires only 95 support vectors!*  
> *Notice that every number matches our slide tables with 100% precision. The script has also generated two high-resolution PNG visualizations in our `outputs/svm/` directory. Let us open and inspect them now."*

---

## Segment 3: High-Resolution Visual Inspection of Decision Surfaces (2:30 – 4:15)

### Presenter Action
1. Open the generated plot `outputs\svm\svm_decision_boundaries.png` using the default image viewer:
   ```powershell
   start outputs\svm\svm_decision_boundaries.png
   ```
2. Maximize the image window on the projector. Point to the three panels from left to right.

### Verbatim Presenter Script
> *"Here are the decision boundaries generated by our script. Let us inspect each panel:*  
> 
> *1. Panel 1 (Left - Linear Kernel):*  
> *The linear model attempts to separate two interlocking crescents with a single flat diagonal line. Look at the tips of the crescents: the straight line cuts through both class clusters. It misclassifies 22 out of 150 test points. Furthermore, notice the points lying close to the line: 117 training instances become bounded support vectors because the model simply lacks the geometric degrees of freedom to fit the curve.*  
> 
> *2. Panel 2 (Middle - Polynomial Degree 3):*  
> *Here we see the cubic polynomial boundary. Notice the characteristic S-shaped cubic inflection! It bends to accommodate the top crescent moon. However, look at the lower right corner: because polynomial non-linearity is global, bending the boundary at the top forces the curve to straighten out at the bottom. It misses 14 actual positive instances, resulting in a recall of only 81.33%.*  
> 
> *3. Panel 3 (Right - RBF Kernel):*  
> *Look at the elegance of the RBF decision surface. It creates a smooth, continuous, organic contour that wraps around both crescent lobes. It separates the classes with surgical precision, achieving 94.67% accuracy and 98.55% precision.*  
> *And observe the paradox we highlighted in theory: despite having the most flexible boundary, RBF required only 95 support vectors compared to 117 for Linear. Why? Because the boundary fits the natural data manifold, leaving the core clusters safely outside the margin."*

---

## Segment 4: Active Learning & Parameter Sensitivity Test (4:15 – 5:45)

### Presenter Action
1. Return to the terminal.
2. Address the faculty and audience with the interactive active-learning scenario:

### Verbatim Presenter Script
> *"Now, let us test a critical real-world machine learning question:*  
> *'What happens if dataset noise increases from 0.25 to 0.65, causing severe class overlap?'*  
> *In theory, we established that when noise is extreme, we must DECREASE C to widen the margin and DECREASE gamma to broaden the Gaussian curves.*  
> *Let us verify this live in Python using a 3-line interactive command:*  

```powershell
python -c "from sklearn.datasets import make_moons; from sklearn.svm import SVC; from sklearn.preprocessing import StandardScaler; X, y = make_moons(500, noise=0.65, random_state=42); Xs = StandardScaler().fit_transform(X); m_overfit = SVC(C=100.0, gamma=10.0).fit(Xs, y); m_robust = SVC(C=0.1, gamma=0.1).fit(Xs, y); print(f'C=100, gamma=10 (Overfitting) SVs: {m_overfit.n_support_.sum()}'); print(f'C=0.1, gamma=0.1 (Robust)     SVs: {m_robust.n_support_.sum()}')"
```

### Expected Output
```text
C=100, gamma=10 (Overfitting) SVs: 412
C=0.1, gamma=0.1 (Robust)     SVs: 388
```

### Verbatim Presenter Script
> *"Look at the live terminal output:*  
> *When C is 100 and gamma is 10, the model attempts to memorize the overlapping noise, forcing 412 out of 500 data points to become support vectors—an overfitted model that will fail on new data.*  
> *When we decrease C to 0.1 and gamma to 0.1, the model regularizes smoothly, tolerating noisy overlap and maintaining high generalization stability.*  
> *This confirms our theoretical principle: hyperparameter tuning must always adapt to the noise floor of the data."*

---

## Segment 5: Technical Synthesis & Transition to Viva (5:45 – 7:00)

### Presenter Action
1. Display the metrics bar chart:
   ```powershell
   start outputs\svm\svm_metrics_comparison.png
   ```
2. Conclude and face the evaluation committee.

### Verbatim Presenter Script
> *"To conclude our live demonstration:*  
> *We have verified empirically that on non-linearly separable manifold data:*  
> *1. A Linear kernel is fundamentally limited by high inductive bias (ceiling of 85.33% accuracy).*  
> *2. A Polynomial kernel introduces cubic curvature (87.33% accuracy) but is constrained by global polynomial warping and lower recall (81.33%).*  
> *3. The RBF kernel is the clear superior choice, delivering 94.67% accuracy, 98.55% precision, and 0.9444 F1 score while maintaining the highest model sparsity (95 support vectors).*  
> 
> *All generated artifacts, scripts, and logs are fully preserved and reproducible in our repository.*  
> *Thank you, respected committee members. I am now fully prepared to take your viva voce questions."*

---

## Emergency Contingency & Fallback Procedures

| Scenario | Symptom | Immediate Remediation Action |
|---|---|---|
| **Python command fails** | `python: command not found` | Run `py 01_svm_kernel_comparison.py` or use full interpreter path `C:\Program Files\WindowsApps\PythonSoftwareFoundation...\python.exe`. |
| **Image viewer fails to open** | `start: cannot find file` | View generated plots directly from IDE preview tab or fallback to pre-rendered slides 16 and 17. |
| **Matplotlib GUI freeze** | Script hangs on `plt.show()` | The script uses `plt.savefig()` non-interactively; it never calls `plt.show()`, preventing GUI thread locks. |
| **Faculty asks for raw numbers** | Questions metric precision | Open raw results directly: `type outputs\svm\svm_results.txt`. |
