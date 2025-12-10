# BVSF Model

Simulation code for studying evolutionary dynamics of fungal symbionts in ambrosia beetles.  
The model integrates Lotka–Volterra competition, genetic drift under bottlenecks, and selection during gallery growth to evaluate fixation, coexistence, and community outcomes.

## Main Files
- **BVSF_v5.py** — Generates all simulation figures (LV dynamics, drift–selection, multi-species, ESS).
- **reliability_tests_v5.py** — Numerical and reproducibility validation.

## Run
```bash
python BVSF_v5.py
python reliability_tests_v5.py

##Requirements
numpy
scipy
matplotlib
scikit-learn

#Citation
Jiang, Z.-R. (2025). BVSF evolutionary simulation framework for fungal symbiosis.
GitHub: https://github.com/sugkp112/bvsf-model

#Contact
Email: ziru.jiang@gmail.com
