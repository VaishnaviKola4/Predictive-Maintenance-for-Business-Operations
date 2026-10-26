# Technical Deep Dive: Mathematical Formulations & Architecture

This document details the mathematical, mechanical, and architectural foundations of the **Predictive Maintenance for Business Operations System**.

---

## 1. Machine Degradation Dynamics & Physics-Informed Synthesis

Industrial machine degradation follows an empirical multi-stage wear curve, famously known as the **Bathtub Curve** and the **P-F Curve (Potential Failure to Functional Failure)**.

```
 Condition / Performance
    ^
 1.0|------- Nominal Baseline Operation -------+
    |                                          \  (Point P: Potential Failure Detected)
    |                                           \
 0.7|--------------------------------------------\-- Warning Threshold (failure_imminent_48h=1)
    |                                             \
    |                                              \  Degradation Acceleration
 0.0|-----------------------------------------------\---+ (Point F: Breakdown Seizure)
    +----------------------------------------------------> Operating Hours / Cumulative Stress
```

### 1.1 Cumulative Wear Model
Machine wear state \(W_m(t)\) evolves over time \(t\) as a function of cumulative runtime and operational stress load factor \(L_m(t)\):

$$\frac{dW_m}{dt} = \kappa_m \cdot L_m(t) \cdot \mathbb{I}(\text{operating})$$

Where:
- \(\kappa_m \sim \mathcal{U}(0.0007, 0.0016)\) is the asset-specific intrinsic degradation rate.
- \(L_m(t) \in [0.4, 1.25]\) represents production schedule throughput intensity.
- \(\mathbb{I}(\text{operating}) \in \{0, 1\}\) indicates whether the machine is running or in planned downtime.

### 1.2 Sensor Physical Coupling Equations
Telemetry variables couple directly with internal wear \(W_m(t)\) and load \(L_m(t)\):

- **Temperature:**
  $$T_m(t) = T_{\text{base}} + 28 \cdot W_m(t) + 8 \cdot L_m(t) + \epsilon_T, \quad \epsilon_T \sim \mathcal{N}(0, 1.2)$$

- **Vibration Velocity (RMS):**
  $$V_m(t) = V_{\text{base}} \cdot (1 + 2.0 \cdot W_m(t)) + 0.9 \cdot L_m(t) + \epsilon_V, \quad \epsilon_V \sim \mathcal{N}(0, 0.3)$$

- **Pressure Retention:**
  $$P_m(t) = P_{\text{base}} \cdot (1 - 0.25 \cdot W_m(t)) + \epsilon_P, \quad \epsilon_P \sim \mathcal{N}(0, 2.5)$$

- **Acoustic Signature:**
  $$A_m(t) = 58.0 + 35 \cdot W_m(t) + 3.5 \cdot V_m(t) + \epsilon_A, \quad \epsilon_A \sim \mathcal{N}(0, 2.0)$$

When \(W_m(t) \ge 0.75\), the machine enters the P-F degradation window: specific fault modes inject non-linear signatures (e.g., thermal spikes for cooling failure, harmonic vibration for bearing spalling).

---

## 2. Feature Engineering & Signal Processing

Raw point-in-time telemetry fails to capture degradation trajectories. The pipeline engineers 58 dynamic physical features:

### 2.1 Multi-Scale Rolling Statistics
For each sensor signal \(S \in \{T, V, P, \text{Power}, A\}\), rolling statistics are extracted across multiple temporal windows \(k \in \{6, 24\}\) (corresponding to 3 hours and 12 hours at 30-minute intervals):

- **Rolling Mean:**
  $$\mu_{S, k}(t) = \frac{1}{k} \sum_{i=0}^{k-1} S(t - i)$$
- **Rolling Dispersion / Volatility:**
  $$\sigma_{S, k}(t) = \sqrt{\frac{1}{k - 1} \sum_{i=0}^{k-1} (S(t - i) - \mu_{S, k}(t))^2}$$
- **Rolling Peak Envelope:**
  $$\mathcal{M}_{S, k}(t) = \max_{i \in [0, k-1]} S(t - i)$$

### 2.2 Domain Interaction Features
- **Specific Vibration Per Revolution:**
  $$\text{Ratio}_{\text{vib/rpm}} = \frac{V_m(t)}{\text{RPM}_m(t) + 10^{-4}}$$
  *Significance:* Detects structural looseness and eccentricity independent of machine speed.
- **Hydraulic Energy Dissipation Ratio:**
  $$\text{Ratio}_{\text{power/pressure}} = \frac{\text{Power}_m(t)}{P_m(t) + 10^{-4}}$$
  *Significance:* Rising power alongside falling pressure isolates internal seal bypass leakage.
- **Thermal Drift Anomaly:**
  $$\Delta T(t) = T_m(t) - \mu_{T, 24}(t)$$
  *Significance:* Measures sudden thermal runaway relative to ambient 12-hour thermal equilibrium.

---

## 3. Supervised Model & Optimization Formulation

The failure prediction task is formulated as estimating the conditional probability of functional failure within the forecast horizon \(H = 48 \text{ hours}\):

$$\hat{y}(t) = \mathbb{P}(Y(t+H) = 1 \mid \mathbf{x}(t))$$

### 3.1 Cost-Weighted Ensemble Objective
In industrial operations, **False Negatives (missed breakdowns)** are roughly 10x to 15x more costly than **False Positives (unnecessary inspections)**:

$$\mathcal{C}(\hat{y}, y) = c_{\text{FN}} \cdot \mathbb{I}(y=1, \hat{y}=0) + c_{\text{FP}} \cdot \mathbb{I}(y=0, \hat{y}=1)$$

The Random Forest objective optimizes class weight balancing:

$$w_c = \frac{N}{2 \cdot N_c}$$

Ensuring split nodes maximize entropy reduction on the minority critical failure class.

### 3.2 Evaluation Metric Formulations
- **Precision:**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{666}{666 + 5} = 0.9925$$
- **Recall (Sensitivity):**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{666}{666 + 72} = 0.9024$$
- **F1 Score:**
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 0.9454$$

---

## 4. Prescriptive Decision & Financial Optimization Formulation

The recommendation engine translates the raw failure probability \(\hat{p} \in [0, 1]\) into optimal maintenance scheduling.

Let \(C_{\text{breakdown}}\) be the expected financial loss from an unplanned crash:

$$C_{\text{breakdown}} = (\bar{t}_{\text{unplanned}} \cdot R_{\text{downtime}}) + C_{\text{parts, rush}}$$

Where:
- \(\bar{t}_{\text{unplanned}} = 10 \text{ hours}\) average catastrophic repair downtime.
- \(R_{\text{downtime}} = \$850/\text{hour}\) lost production throughput and emergency technician rate.
- \(C_{\text{parts, rush}} = \$3,500\) expedited component replacement fee.
- Total \(C_{\text{breakdown}} = (10 \times 850) + 3,500 = \$12,000\).

Let \(C_{\text{preventive}}\) be the scheduled intervention cost:

$$C_{\text{preventive}} = (\bar{t}_{\text{preventive}} \cdot R_{\text{planned}}) + C_{\text{parts, standard}}$$

Where:
- \(\bar{t}_{\text{preventive}} = 2 \text{ hours}\).
- \(R_{\text{planned}} = \$150/\text{hour}\).
- \(C_{\text{parts, standard}} = \$600\).
- Total \(C_{\text{preventive}} = (2 \times 150) + 600 = \$900\).

The **Net Cost Avoidance** achieved by triggering preventive action is:

$$\Delta C = \mathbb{I}(\hat{p} \ge 0.45) \cdot (C_{\text{breakdown}} - C_{\text{preventive}}) = \$11,100 \text{ per avoided event}$$

