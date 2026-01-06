# AS-MAS  
**AI-Driven Autonomous Orbital Multi-Agent System**

AS-MAS is a physics-first, AI-driven platform designed for satellite orbit monitoring and anomaly detection.  
The project is engineered as a **distributed system** rather than a single ML model, combining orbital mechanics, time-series analysis, and deep learning within a production-ready pipeline.

---

## 🎯 Project Vision

The goal of AS-MAS is to detect **orbital anomalies** by:
1. Modeling expected satellite motion using physics-based propagation
2. Measuring deviations (residuals) between expected and observed behavior
3. Detecting abnormal patterns using both statistical and deep learning approaches

The system prioritizes:
- Deterministic physics before AI
- Observability and reproducibility
- Incremental validation through controlled anomaly scenarios

---

## 🧱 System Architecture (Current State)

**Pipeline Overview:**
TLE Data
↓
Orbit Propagation (Physics Core)
↓
Residual Computation
↓
Time-Series Persistence
↓
Baseline Anomaly Detection (Z-score)
↓
LSTM Autoencoder (Reconstruction Error)


---

## ⚙️ Implemented Components

### 1️⃣ Physics Core
- TLE parsing and satellite loading
- Orbit propagation using Skyfield
- Position and velocity computation
- Real-time UTC timestamp support

### 2️⃣ Residual Layer
- Computation of deviation between expected and observed orbital states
- Residual magnitude as a domain-aware anomaly signal

### 3️⃣ Data Pipeline
- Residual time-series persisted to CSV
- Append-mode storage for continuous data accumulation
- Designed to be database-ready (PostgreSQL in next sprint)

### 4️⃣ Baseline Anomaly Detection
- Rolling statistics (mean, standard deviation)
- Z-score–based anomaly detection
- Cold-start handling (no false alarms during initial window)
- Used as a reference baseline for ML models

### 5️⃣ LSTM Autoencoder (Sprint 3)
- Sliding window dataset built from residual time-series
- LSTM-based autoencoder trained on normal orbital behavior
- Reconstruction error used as anomaly score
- Validated using controlled anomaly scenarios

---

## 🧪 Controlled Anomaly Validation

To validate system behavior, **synthetic but realistic anomalies** were injected:

- **Spike anomaly:** Single-timestep extreme deviation  
  (e.g., sensor glitch, sudden maneuver)

Results:
- Baseline method detected strong point anomalies depending on threshold
- LSTM autoencoder showed a **significant increase in reconstruction error**
- Demonstrated clear separation between normal and anomalous behavior

This confirms that the system detects **pattern deviations**, not just large values.

---

## ✅ Current Project Status

- ✔ Physics-based orbital propagation
- ✔ Residual signal generation
- ✔ Time-series persistence
- ✔ Baseline anomaly detector
- ✔ LSTM Autoencoder (train + inference)
- ✔ Controlled anomaly validation

**Sprint 3 successfully completed.**

---

## 🚀 Next Steps (Sprint 4)

- Migrate data layer from CSV to PostgreSQL
- Introduce a scheduler for periodic execution
- Prepare infrastructure for monitoring and drift detection
- Move toward a fully autonomous, continuously running system

---

## 🧠 Key Engineering Principles

- Physics-first, AI-second design
- Explicit separation between data, logic, and models
- Validation before optimization
- ML models treated as system components, not standalone solutions

---

## 📌 Disclaimer

This project is a research and engineering prototype.  
It focuses on architectural clarity, correctness, and learning value rather than operational deployment.
