# 🧠 Pruning Spiking Neural Networks: A Constrained Optimization Approach

**Research Internship Project — University of Oulu** **Author:** Muhammad Hamza (IIT Kharagpur)  
**Supervisors:** Shuai Lee & Ameer Hamza Khan  

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![snnTorch](https://img.shields.io/badge/snnTorch-111111?style=for-the-badge&logo=python&logoColor=white)](https://snntorch.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
Spiking Neural Networks (SNNs) offer brain-like, asynchronous computation, promising orders of magnitude less power consumption than standard ANNs. However, deploying SNNs onto severely energy-constrained neuromorphic hardware requires rigorous network sparsity.

This repository contains the complete research pipeline for **Criticality-Constrained Quadratic Pruning (CQP)**. By combining synaptic weight magnitudes with surrogate gradient "criticality" scores, this iterative pruning algorithm safely compresses SNNs to extreme sparsities (90%+) while mitigating the catastrophic accuracy collapse seen in traditional Magnitude (L1) pruning.

### ✨ Key Scientific Contributions
1. **Combined Importance Scoring:** Overcame the NP-hard "Continuous Relaxation Trap" of standard convex solvers (CVXPY) by developing a native PyTorch metric that mathematically binds physical weight norms with surrogate gradient criticality.
2. **Temporal Redundancy via KL-Divergence:** Proved that up to 90% of SNN inference timesteps on complex datasets can be truncated with < 5% information loss, yielding massive theoretical energy savings for neuromorphic chips.
3. **The Criticality Cliff:** Empirically mapped the "critical brain hypothesis" in artificial spiking networks, isolating the exact sub-graph of indispensable neurons required to prevent topological shattering.
4. **Zombie-Weight Masking:** Engineered a dynamic gradient-masking loop to prevent Adam momentum states from resurrecting pruned weights during the iterative healing process.

---

## 📈 Key Results (Fashion-MNIST)
Evaluated on a 3-layer fully-connected Leaky Integrate-and-Fire (LIF) network (784 → 256 → 128 → 10) using Direct Current Injection encoding. 

* **Baseline Performance:** The dense SNN achieved **89.6%** accuracy, outperforming the equivalent ANN baseline (89.0%).
* **Extreme Sparsity (90%):** Standard Magnitude pruning collapsed to `61.0%`. Our CQP algorithm maintained **`78.5%`**, yielding a definitive **+17.5% performance advantage**.
* **Temporal Savings:** KL-Divergence mapping identified the network's stabilization threshold at $t=1/10$, allowing for a theoretical **90% reduction** in inference energy.

---

## 🧮 The Mathematics of CQP
Traditional pruning relies purely on the L1 norm of weights. In SNNs, this ignores the temporal sensitivity of spiking thresholds. CQP calculates a surrogate gradient over multiple batches to score a neuron's "criticality" ($C_i$). 

To avoid the mathematical collapse of rigid solver constraints, CQP dynamically calculates a combined score:
$$Importance_i = ||W_i||_1 \times (C_i)^\alpha$$

Where $\alpha$ dictates the balance:
* $\alpha = 0$: Pure Magnitude Pruning
* $\alpha = 0.5$: The "Goldilocks Zone" (Peak CQP performance)
* $\alpha > 1$: Over-prioritizes gradients (protects useless small weights)

---

## 📂 Repository Structure & Pipeline

This project is structured as a sequential pipeline. Running the scripts in order reproduces the entire research lifecycle:

| Phase | Script | Description |
| :---: | :--- | :--- |
| **1** | `01_generate_data.py` | Downloads and formats the Fashion-MNIST/MNIST dataset into PyTorch tensors. |
| **2** | `02_train_snn.py` | Trains the baseline LIF SNN and an equivalent ReLU ANN. |
| **3** | `03_criticality_scores.py` | Computes the surrogate gradient criticality scores for the input layer. |
| **4** | `04_kl_temporal.py` | Measures KL-divergence between timesteps to calculate temporal redundancy. |
| **5** | `05_pruning_sweep.py` | **[Core Algorithm]** Runs Iterative CQP, Magnitude, and Random pruning sweeps up to 90% sparsity. |
| **6** | `06_layer_sensitivity.py` | Prunes individual layers to map topological bottlenecks. |
| **7** | `07_alpha_sweep.py` | Sweeps the $\alpha$ parameter to prove the mathematical superiority of Combined Scoring. |
| **8** | `08_plots.py` | Generates all publication-ready Matplotlib figures from the JSON logs. |
| **-** | `models.py` | Shared PyTorch/snnTorch architectures. |
| **-** | `utils.py` | Data loaders, direct current injection encoding, and evaluation logic. |

---

## 🚀 Quick Start
**Prerequisites:** Python 3.8+, PyTorch, snnTorch, Matplotlib.

```bash
# Clone the repository
git clone [https://github.com/Hamza-Faarooq/SNN-Criticality-Pruning.git](https://github.com/Hamza-Faarooq/SNN-Criticality-Pruning.git)
cd SNN-Criticality-Pruning

# Install dependencies
pip install torch snntorch torchvision matplotlib numpy

# Run the complete pipeline
python 01_generate_data.py
python 02_train_snn.py
python 03_criticality_scores.py
python 04_kl_temporal.py
python 05_pruning_sweep.py
python 06_layer_sensitivity.py
python 07_alpha_sweep.py
python 08_plots.py
