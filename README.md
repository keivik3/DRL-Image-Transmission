# RL-ImageTx: Reinforcement Learning for Timeliness-Aware Image Transmission

## 🧠 Overview
This project explores **reinforcement learning (RL)** techniques for optimizing **image transmission over wireless networks**.  
We propose a **Monte Carlo ε-greedy agent** that dynamically adjusts image compression and coding parameters to balance **quality (PSNR)** and **timeliness (Value of Information, VoI)** under variable channel conditions.

The project includes a reproducible Python simulator and a LaTeX report (IEEE format).

---

## 🧩 Key Features
- **Adaptive transmission control** using RL  
- **Timeliness-aware reward** combining PSNR, VoI, and delay  
- **Reproducible experiments** with simulated wireless channel  
- **Lightweight Monte Carlo ε-greedy algorithm**  
- Extendable to DQN or PPO agents  

---

## 📁 Repository Structure
```
RL-ImageTx/
│
├── src/ # Source code
├── experiments/ # Results and analysis
├── report/ # LaTeX paper
└── data/ # Sample images
```


---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/RL-ImageTx.git
cd RL-ImageTx
pip install -r requirements.txt
```

---
# 📊 Expected Results

* RL agent learns to adjust transmission settings under varying SNR;
* Improvement in average VoI and PSNR over fixed baselines;
* Learning curve showing stable convergence.

# 🧾 Report

The full IEEE-format project report is available in:
```
report/RL_Image_Transmission_Project.pdf
```
# 📄 License

This project is released under the licence
