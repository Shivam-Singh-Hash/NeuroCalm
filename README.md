# 🧠 NeuroCalm: Multi-Modal Deep Learning Stress Diagnostic Suite

**NeuroCalm** is a production-grade, AI-powered physiological monitoring platform designed for modern clinical psychology and human performance analysis. By ingesting high-frequency wearable sensor telemetry—including Electrodermal Activity (EDA), Skin Temperature, and Heart Rate—NeuroCalm leverages advanced deep learning architectures to provide real-time, highly granular diagnostics of the human nervous system.

### ✨ Key Features
*   **Multi-Modal Neural Inference**: Powered by a robust PyTorch backend, utilizing a global model hub with specialized architectures including CNNs (Spatial Feature Extraction), LSTMs (Temporal Dependency Mapping), and a best-in-class Hybrid CNN-LSTM model.
*   **High-Fidelity Telemetry Dashboard**: A stunning, glassmorphic React.js interface featuring synchronized, real-time biosignal charting (via Recharts) and an AI-driven temporal stress ribbon.
*   **Premium Clinical Reporting**: One-click generation of beautifully structured, native PDF diagnostic reports utilizing jsPDF, completely formatted for clinical hand-offs with dynamic confidence scores and AI recommendations. 
*   **NeuroChat AI Assistant**: Deep integration with Groq (LLaMA-3) to provide instant, human-readable insights based directly on the subject's localized biometric shifts and internal machine-learning state.
*   **Acoustic Intervention Engine**: Built-in 4Hz Theta-wave binaural audio synthesizer mapped with deep Brown noise to actively assist in down-regulating sympathetic nervous system spikes during high-stress detections.

### 🛠 Tech Stack
*   **Frontend**: React.js, Recharts, jsPDF, html2canvas, Lucide Icons, Vanilla CSS (Glassmorphism UI).
*   **Backend Engine**: Python, Flask, PyTorch, Scikit-Learn, Pandas.
*   **AI Models**: Hybrid CNN-LSTM Architecture, Groq API (LLaMA-3).

---
*Created as an advanced biometric diagnostic platform.*
