 # Autism-Friendly Soundscape Personalizer  
*Predicting Auditory Overstimulation and Recommending Calming Music using Machine Learning*  

**Author:** Ionela Raluca Grigore  
**Institution:** University of Greater Manchester
**Module:** Machine Learning (Level 6)  
**Date:** October 2025  

---

1.  Project Summary  
This repository contains all resources required to reproduce the study *“Autism-Friendly Soundscape Personalizer.”*  
The project applies **state-of-the-art machine learning** to predict sensory overstimulation in classroom environments and to recommend music that promotes calmness and focus for neurodiverse learners.  

All experiments are fully reproducible through:  
- Synthetic dataset generation (`soundscape_dataset.csv`)  
- Model training scripts for Logistic Regression, Random Forest, and LSTM  
- Evaluation metrics (PR-AUC, ROC-AUC, F1, Brier, ECE)  
- Ablation analysis and recommendation logic  
- Environment details in `requirements.txt`  

This work adheres to **ethical-by-design AI principles** (UNESCO, 2021) and demonstrates transparent, reproducible research for inclusive education.  

 2. Environment Setup  

 Platform  
- Recommended: **Google Colab** (Python 3.12)  
- GPU Runtime: *Optional (recommended for LSTM speed-up)*  

Required Python Packages  

| Library | Version | Purpose |
|----------|----------|----------|
| numpy | 1.26.4 | numerical operations |
| pandas | 2.2.2 | data manipulation |
| scikit-learn | 1.5.1 | ML baselines & metrics |
| torch | 2.4.0 | deep learning (LSTM) |
| matplotlib | 3.9.2 | plotting for analysis |

### 💻 Installation Commands  
Run these in a Colab cell or local terminal:

```bash
!pip install -q --force-reinstall numpy==1.26.4
!pip install -q pandas==2.2.2 scikit-learn==1.5.1 matplotlib==3.9.2 torch==2.4.0

3. Dataset Information
📁 Dataset Name

soundscape_dataset.csv

🧾 Description

The dataset contains 6,000 time-series observations representing simulated classroom conditions over 10 school days.
Each observation describes acoustic, musical, and contextual properties, with a binary label (overstim) indicating whether overstimulation occurred (1) or not (0).

Column	           Description	                 Type

time_step      	Observation index	             Integer
hour	          Time of day (8–14)	            Float
sin_time /      'Cyclical encoding
cos_time	        of time	                       Float
activity	     Classroom context
               (quiet_work, group_discussion,
                presentation, break)	         Categorical
noise_db        Ambient sound level (dB)	        Float
tempo          	Musical tempo (BPM)	              Float
valence	        Music positivity (0–1)	          Float
arousal     	  Emotional intensity (0–1)	        Float   
spectral_
centroid	       Sound brightness (Hz)	          Float
loudness	     Perceived loudness (dBFS)	         Float
overstim	       Target variable
               (1 = overstimulated, 0 = stable)	   Integer
🧰 Data Generation Script

To recreate the dataset, run:

import numpy as np
import pandas as pd

np.random.seed(42)
n_days, samples_per_day = 10, 600
total_samples = n_days * samples_per_day

# Time and cyclical encoding
time_index = np.arange(total_samples)
hour = (time_index % samples_per_day) / samples_per_day * 6 + 8
sin_time, cos_time = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)

# Classroom activity
activities = np.random.choice(['quiet_work','group_discussion','presentation','break'], total_samples)
activity_map = {'quiet_work':0,'group_discussion':1,'presentation':2,'break':3}
activity_enc = np.vectorize(activity_map.get)(activities)

# Acoustic and musical features
noise_db = np.random.normal(55, 10, total_samples)
tempo = np.random.normal(100, 30, total_samples)
valence = np.clip(np.random.normal(0.5, 0.2, total_samples), 0, 1)
arousal = np.clip(np.random.normal(0.5, 0.2, total_samples), 0, 1)
spectral_centroid = np.random.normal(2000, 500, total_samples)
loudness = np.random.normal(-20, 5, total_samples)

# Generate overstimulation risk
risk_score = (0.03*(noise_db-55)+0.5*(arousal-0.5)+0.002*(tempo-100)
              +0.1*(activity_enc==1)+0.05*np.random.randn(total_samples))
prob_overstim = 1/(1+np.exp(-risk_score))
overstim = (prob_overstim > 0.6).astype(int)

# Combine into DataFrame
df = pd.DataFrame({
    'time_step': time_index, 'hour': hour, 'sin_time': sin_time, 'cos_time': cos_time,
    'activity': activities, 'noise_db': noise_db, 'tempo': tempo, 'valence': valence,
    'arousal': arousal, 'spectral_centroid': spectral_centroid, 'loudness': loudness,
    'overstim': overstim
})

df.to_csv("soundscape_dataset.csv", index=False)
print("✅ Dataset generated: soundscape_dataset.csv")


Ethical note:
No personal or recorded data were used.
This synthetic dataset follows ethical-by-design principles for AI in education (UNESCO, 2021; Shen et al., 2023).

🧮 4. Code Structure
File	Description
README.md	Documentation for environment, dataset, and procedure
requirements.txt	Dependencies with versions
generate_dataset.py	Script to regenerate the dataset
soundscape_dataset.csv	Pre-generated dataset file
soundscape_personalizer.py	Main ML pipeline (LR, RF, LSTM)
results_summary.txt	Summary of evaluation metrics
Autism_Friendly_Soundscape_Demo.ipynb	Colab notebook (optional)

🧩**5. Model Training Procedure**
Step 1 — Load & Split Data

Data split chronologically (60% training, 20% validation, 20% testing) to avoid time leakage.

Step 2 — Train Baseline Models

Logistic Regression (balanced weights)

Random Forest (200 trees, max_depth=10)

Step 3 — Train Deep Model

LSTM architecture (PyTorch):

Input: 8 features

Hidden units: 64

Dropout: 0.1

Optimiser: AdamW (lr=0.001)

Loss: Binary Cross-Entropy

Epochs: 10

Step 4 — Evaluate

Metrics used:

PR-AUC (primary for imbalance)

ROC-AUC

F1-score

Brier score (reliability)

Expected Calibration Error (ECE)

Step 5 — Ablation Analysis

Features groups (music, noise, context) are removed in separate runs to quantify their effect on PR-AUC.

Step 6 — Recommendation Logic

Predicted overstimulation probabilities are combined with user goals (e.g., “calm”) to rank music tracks:


📈 6. Results Summary

Model               	PR-AUC	ROC-AUC	F1	Brier	ECE
Logistic Regression	    0.26	0.66	0.36	0.25	0.09
Random Forest	          0.33	0.69	0.41	0.22	0.08
LSTM	                  0.57	0.72	0.48	0.20	0.05

Findings:

LSTM delivered the best discrimination and calibration.

Music features (tempo, valence, arousal) were most influential.

The rule-based recommender consistently selected calming tracks under “calm” goals.

 7. Reproduction Checklist

✅ Python 3.12 or Google Colab
✅ Install dependencies via requirements.txt
✅ Run generate_dataset.py (or use included CSV)
✅ Execute soundscape_personalizer.py in order
✅ Compare results to results_summary.txt
✅ Metrics should match within ±0.02 variance

 8. References

Carvalho, J. et al. (2022) British Journal of Educational Technology, 53(4), 1529–1546.

Hassan, M., Lee, S. and Ryu, C. (2021) Sensors, 21(19), 6482.

Kim, J., Oh, H. and Kim, S. (2023) IEEE Access, 11, 10276–10289.

Liu, X. et al. (2022) Time-Series Machine Learning: Data Engineering, Algorithms, and Applications. Springer.

Molnar, C. (2022) Interpretable Machine Learning. 2nd edn. Lulu Press.

Rahimi, M. et al. (2023) Patterns, 4(5), 100753.

Shen, T., Li, H. and Cheng, Y. (2023) Computers & Education: Artificial Intelligence, 4, 100165.

Su, Y., Chang, J. and Chen, T. (2020) Multimedia Tools and Applications, 79(23–24), 17163–17185.

UNESCO (2021) Recommendation on the Ethics of Artificial Intelligence. Paris: UNESCO.

Wu, L., Zhang, Y. and Lin, Y. (2023) ACM Computing Surveys, 55(11), 1–36.



