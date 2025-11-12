# RL-ImageTx: Reinforcement Learning for Timeliness-Aware Image Transmission

## 🧠 Overview
This project explores **reinforcement learning (RL)** techniques for optimizing **image transmission over wireless networks**.  
We propose a **Monte Carlo ε-greedy agent** that dynamically adjusts image compression and coding parameters to balance **quality (PSNR)** and **timeliness (Value of Information, VoI)** under variable channel conditions.

The project includes a reproducible Python simulator and a LaTeX report (IEEE format).

---

## State for 12.11.2025:
#!/bin/bash
cat > README.md << 'EOF'
# Autoencoder + DQN for Adaptive Video Compression

This project builds a sequential **autoencoder** for video (CartPole-v1) using PyTorch.  
Frames are encoded, decoded, and used with previously reconstructed frames to improve temporal consistency.  
A **DQN agent** learns to select the latent code length dynamically to balance quality and compression speed.

## Overview
- **Encoder/Decoder:** Separate CNN modules with fully connected layers.
- **Temporal context:** Each new frame is decoded together with several previously decoded ones.
- **Optimization:** Encoder and decoder parameters concatenated and trained jointly with Adam.
- **RL controller:** DQN chooses the latent length (not percentage) based on reconstruction quality (PSNR) and speed.

## Gym Rendering Issue
Some Gym versions return `env.render()` as a list instead of an array.  
Fix:
```python
if isinstance(frame, list): frame = frame[0]
frame = np.asarray(frame)
```
This ensures frame shape is (H, W, 3) for grayscale conversion.
## DQN Summary:
The DQN (ε-greedy) selects latent code length K using:
* State: reconstruction error, compression ratio
* Action: choose K ∈ {32, 64, 96, 128}
* Reward: PSNR - λ*K
It learns to optimize trade-off between fidelity and compression.

# Notes
* Fixed reshape() issue in encoder for non-contiguous tensors.
* Gym API updated (obs, info = env.reset()).
* Added compatibility handling for Gym/Gymnasium versions.

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
