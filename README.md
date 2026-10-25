# Intelligent Bearing & Gearbox Fault Diagnosis for Industrial & EV Drivetrains

End-to-end **Condition-Based Maintenance (CBM)** and **Edge AI Fault Diagnosis** pipeline built in **MATLAB & Simulink**, validated on experimental accelerometer recordings from the **Case Western Reserve University (CWRU) Bearing Dataset**.

## 🚀 Key Performance Highlights
* **Route A Accuracy (Order Features + RBF-SVM):** **99.23%** (Macro-F1: **0.9933**)
* **Route B Accuracy (CWT Scalogram Model):** **100.00%** (Macro-F1: **1.0000**)
* **Edge Inference Latency:** **6.16 ms** per 1s window (Beats $<8\text{ ms}$ Strong Result target)
* **Embedded Deployment:** ANSI C library compiled via **MATLAB Coder (`codegen`)**
* **System Simulation:** Self-contained **Simulink Digital Twin** with real-time scopes

## 📁 Repository Structure
* `run_Project5_Kaggle_Master.m` — Master training, evaluation, and figure export script.
* `classifyWindow.m` — Lightweight C-compatible classification entry point.
* `run_RealTime_Edge_Monitor.m` — Real-time streaming HMI using `dsp.AsyncBuffer` and exponential smoothing.
* `Project5_Drivetrain_Edge_Simulink.slx` — Self-contained Model-Based Design Simulink Digital Twin.
* `Fig1` to `Fig5` — Publication-grade diagnostic plots.

## 🛠️ How to Run
1. Place your CWRU `.mat` files inside `Kaggle_Data/raw/`.
2. Open MATLAB and run `run_Project5_Kaggle_Master` to train models and export figures.
3. Generate standalone C code: `codegen classifyWindow -args {zeros(12000,1)} -config:lib -report`
4. Launch the live monitoring HMI: `run_RealTime_Edge_Monitor` or run `Project5_Drivetrain_Edge_Simulink.slx`.
