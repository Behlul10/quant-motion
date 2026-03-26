

Research Question: Can Variational Quantum Eigensolver (VQE) optimization produce more natural motion smoothing than classical methods by treating "motion energy" as a quantum Hamiltonian minimization problem?

# Objectives

1\. Primary: Implement VQE-based motion smoothing and compare against classical scipy optimization  
2\. Secondary: Benchmark on IBM Quantum hardware (127-qubit Heron) vs. simulator  
3\. Deliverable: Reproducible Jupyter notebook pipeline \+ technical report

# Technical Approach (Notebook-Based)

## Phase 1: Data Pipeline (Weeks 1-3)

- Export MediaPipe hand tracking data from Vision Mapper (JSON format)  
- Load into pandas, visualize raw noise  
- Implement classical baseline: scipy.optimize.minimize with jerk cost function  
- Store datasets in GCP Cloud Storage (utilize $300 credit)

## Phase 2: Quantum Implementation (Weeks 4-8)

- Encode joint trajectories as quantum states (amplitude encoding)  
- Define Hamiltonian:H \= (acclerationi)2 (minimize jerk)  
- Implement VQE using Qiskit with SPSA optimizer  
- Run on:  
  - Local simulator (Aer) for development  
  - IBM Quantum **ibm\_brisbane** or **ibm\_sherbrooke** (real hardware) for final validation

## Phase 3: Analysis (Weeks 9-12)

- Metrics: Total jerk, smoothness index, computation time  
- Visualization: Overlay classical vs. quantum trajectories  
- Statistical comparison on 10+ motion samples

# Deliverables

## 1\. GitHub Repository:

- notebooks/01\_data\_loading.ipynb  
- notebooks/02\_classical\_baseline.ipynb  
- notebooks/03\_vqe\_smoothing.ipynb  
- notebooks/04\_hardware\_comparison.ipynb  
- src/ module with reusable functions

## 2\. Dataset: 10 sample hand motions (raw \+ classical \+ quantum processed)

## 3\. Technical Report (8-10 pages): Comparison methodology, results, limitations

## 4\. Final Demo: Video showing raw noisy input → quantum-smoothed output

# Resource Requirements

- IBM Quantum: Free tier (10 min/month) \+ potential academic access via advisor  
- Local: Laptop with Python, Qiskit, Jupyter

# Success Criteria

\- VQE produces visually smoother motion than classical method on at least 70% of samples

\- Successfully executes on real IBM quantum hardware (even if simulator performs better due to noise)

\- Complete documentation allowing reproduction by other CHI researchers  
