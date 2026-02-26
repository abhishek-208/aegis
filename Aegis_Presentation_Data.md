# Aegis: Enhanced Byzantine-Resilient Federated Learning
*Presentation Data & Slide Outline for IIT Delhi*

---

## Slide 1: Title Slide
**Title:** Aegis: Enhanced Byzantine-Resilient Federated Learning
**Subtitle:** Structural Improvements and Advanced Threat Mitigation
- **Presenter:** [Your Name]
- **Institution:** IIT Delhi
- **Focus:** Recap of Aegis, recent architectural enhancements, and advanced Byzantine attack vectors.

---

## Slide 2: The Challenge in Federated Learning
**Heading:** Vulnerabilities of Standard Aggregation
**Bullet Points:**
- Standard aggregation strategies like FedAvg heavily rely on client trust.
- In distributed environments, Byzantine (malicious) clients can compromise the global model.
- Vulnerabilities exist in scale (sending massive weights), direction (reversing updates), and volume (faking data size).
- A robust, dynamic defense mechanism is strictly required to filter malicious updates without affecting honest non-IID clients.

---

## Slide 3: Recap - Original Aegis Architecture
**Heading:** Original Aegis: A Defensive Aggregator
**Bullet Points:**
- **Goal:** Provide a resilient alternative to FedAvg that filters out Byzantine updates.
- **Mechanism:** Operates by treating model aggregation as a geometric anomaly detection problem.
- **Core Philosophy:** Instead of blind averaging, Aegis evaluates each client's contribution dynamically based on similarity and historical credibility.

---

## Slide 4: Recap - Original Aegis Mechanics (Volume Clipping)
**Heading:** Step 1: Volume Bounding & Clipping
**Bullet Points:**
- **The Threat:** Attackers can inflate their reported dataset size to dominate the final weighted average.
- **The Aegis Solution:** Volume Clipping limits the maximum reported data size.
- **Rule:** Client data sizes are strictly clipped to `2.0 * Average_Data_Size`.
- **Result:** Upper-bounds the influence any single party can exert on the global aggregation step.

---

## Slide 5: Recap - Original Aegis Mechanics (MAD Filtering)
**Heading:** Step 2: Distance-Based Anomaly Detection
**Bullet Points:**
- Calculates the Coordinate-wise Median of all client updates to find a "robust center."
- Computes Euclidean distance from each client's update to this median.
- Uses **Median Absolute Deviation (MAD)** to establish an adaptive threshold dynamically based on the variance of updates in the current round.
- **Filter:** Soft / Hard rejection of clients extending beyond this safe statistical boundary.

---

## Slide 6: Recap - Original Aegis Mechanics (Credit Scoring)
**Heading:** Step 3: Reputation and Credit Scoring
**Bullet Points:**
- Once malicious outliers are hard-filtered, the remaining approved clients are weighted smartly.
- Computes a mathematical **Credit Score** using distance and data volume: `Score = Clipped_Volume / (Euclidean_Distance + Epsilon)`
- Closer, more representative clients receive a higher aggregate priority over borderline clients.
- The global model is updated using a weighted sum driven by these normalized credit scores.

---

## Slide 7: The Need for Enhancements
**Heading:** Why Upgrade Aegis? Identifying Edge Cases
**Bullet Points:**
- **Issue 1 - Late Stage Instability:** Highly skewed Non-IID batches caused sudden accuracy drops in mature models.
- **Issue 2 - Convergence Overshooting:** Fixed step sizes prevented fine-tuning near the optimal global minima.
- **Issue 3 - Directional Blindspots:** Simple magnitude (Euclidean) filters couldn't easily detect clients perfectly reversing the update direction (opposite angle, same distance).
- **Need:** We required state-based server memory, better convergence control, and angle-aware geometry.

---

## Slide 8: Enhancement 1 - Server-Side Global Momentum
**Heading:** Server Momentum: The "Inertia Buff"
**Bullet Points:**
- **Change:** Introduced a `SERVER_MOMENTUM_ENABLED` architectural switch (Default: `0.9` velocity retention).
- **Function:** Standard FedAvg/Aegis has zero memory and reacts wildly to bad rounds. Server momentum stores past update velocity.
- **Impact:** Acts as a stabilizing inertia buff, completely preventing massive late-stage accuracy drops when a specific round samples an extremely skewed batch of clients.

---

## Slide 9: Enhancement 2 - Learning Rate Decay
**Heading:** Stable Convergence via LR Decay
**Bullet Points:**
- **Change:** Implemented a global learning rate decay multiplier per round (`LR_DECAY_RATE = 0.99`).
- **Function:** Gradually reduces the client learning rate dynamically as the rounds progress up to a safe floor limit.
- **Impact:** Eliminates chaotic overshooting near the optimal loss minima and ensures smooth, monotonic convergence in the final stages of the FL simulation.

---

## Slide 10: Enhancement 3 - Delta-Based Representation
**Heading:** Processing Updates (Deltas) over Raw Weights
**Bullet Points:**
- **Change:** Aegis was refactored to compute and analyze **Deltas** (`Client_Weights - Global_Weights`) instead of absolute raw weights.
- **Function:** Isolates the exact directional gradient information introduced by the client during local training.
- **Impact:** Greatly increases the sensitivity and accuracy of the median computation and variance detection (especially critical when DP clipping is involved).

---

## Slide 11: Enhancement 4 - Cosine Similarity Integration
**Heading:** Closing the Directional Loophole
**Bullet Points:**
- **Change:** Integrated a heavy Cosine Similarity check and Penalty against the computed median update.
- **Hard Filter Base:** Immediately rejects updates exhibiting negative cosine similarity (`cos_sim < 0.0`), identifying outright opposing trajectories.
- **Credit Score Penalty:** Applies a massive algorithmic penalty (`Cosine_Penalty * 10`) to the denominator of the credit calculation, effectively nullifying mathematically orthogonal or suspicious inputs.

---

## Slide 12: Attack Feature 1 - The Sign Flip Attack
**Heading:** Testing Defenses: Sign Flip Vector
**Bullet Points:**
- **Mechanism:** Computes the honest gradient delta, completely reverses its sign (direction), and re-applies it to the global weights.
- **Intent:** Forces the global model to step exactly backwards away from the calculated local minimum.
- **Why Added:** To explicitly test our new Cosine Similarity features—verifying that Aegis can detect identical magnitude parameters pointing in the wrong direction.

---

## Slide 13: Attack Feature 2 - Stealthy Additive Noise
**Heading:** Testing Defenses: Additive Mean-Shift Noise
**Bullet Points:**
- **Mechanism:** Injects extreme Gaussian noise into the weights, but scales the noise relative to the norm of the honest delta to stay mathematically plausible.
- **Intent:** Corrupts the weights destructively without blowing up the absolute volume thresholds of the aggregator.
- **Why Added:** To test the robustness of the Median Absolute Deviation (MAD) dynamic thresholding against intelligently constrained variance variance spoofing.

---

## Slide 14: Attack Feature 3 - The Orthogonal Attack (Delta-Aware)
**Heading:** Advanced Evasion: Orthogonal Noise
**Bullet Points:**
- **Mechanism:** Generates random Gaussian noise, performs Gram-Schmidt Orthogonalization against the honest delta, ensuring the dot product is perfectly 0.
- **Intent:** The malicious noise is completely perpendicular to the honest update.
- **Why Added:** This sophisticated attack specifically attempts to bypass angular/directional defenses (Cosine Similarity) while matching honest magnitudes, proving the necessity of combined distance + angle defensive filtering.

---

## Slide 15: Attack Feature 4 - The Volume Spam Attack
**Heading:** Exploiting Aggregation Weighting
**Bullet Points:**
- **Mechanism:** A hybrid attack combining a standard 'Label Flip' with a falsely reported dataset size (e.g., reporting 1 Billion data points).
- **Intent:** To mathematically dominate the final summation step of FedAvg by commanding 99% of the aggregation weight.
- **Why Added:** Demonstrates the catastrophic failure of naive size-based aggregation and validates the absolute necessity of Aegis’s Step 1 `Volume Clipping` safeguard.

---

## Slide 16: Summary & Impact
**Heading:** Delivering a Robust FL Environment
**Bullet Points:**
- **Adaptability:** Upgraded Aegis leverages spatial anomaly detection via both Distance and Angle (Cosine).
- **Stability:** Momentum constraints and decay functions prevent catastrophic unlearning in extreme edge cases.
- **Threat Preparedness:** The inclusion of advanced, delta-aware orthogonal attacks allows continuous stress-testing of Aegis against state-of-the-art poisoning strategies.
- **Conclusion:** Aegis has evolved from a basic statistical filter into a comprehensive, geometry-aware aggregation framework.

---

## Technical Theory: Mathematical Breakdown (Present Aegis)
*The following time-complexity sub-steps are directly formatted to match the style of the original legacy presentation slides, updated for the current codebase functionality (Deltas, Cosine Penalty, MAD, Momentum).*

### **1. Step 1: Distribute Model and Training - $\boldsymbol{O(dn)}$**
a. Server distributes global model ($W_{global}$) among clients.
b. Clients train the model locally across local epochs.
c. Server receives $k$ uploaded model weights ($W_k$).
d. Server also receives $n_k$ (data points per client).

### **2. Step 2: Compute Deltas & Robust Center - $\boldsymbol{O(dk)}$**
a. Compute each client's delta (directional update): $\Delta_k = W_k - W_{global}$
   - Cost per client: $\boldsymbol{O(d)}$ $\rightarrow$ Total: $\boldsymbol{O(dk)}$
b. Stack all $k$ delta vectors.
c. Calculate element-wise coordinate median, $\Delta_{median}$ - $\boldsymbol{O(dk)}$

### **3. Step 3: Dual Anomaly Scores - $\boldsymbol{O(dk)}$**
a. Initialize empty lists for Euclidean Distances ($E_k$) and Cosine Penalties ($P_k$).
b. Euclidean Distance from center, $E_k = ||\Delta_k - \Delta_{median}||_2$
   - Cost per client: $\boldsymbol{O(d)}$ $\rightarrow$ Total: $\boldsymbol{O(dk)}$
c. Cosine Similarity, $Cos_k = \frac{\Delta_k \cdot \Delta_{median}}{||\Delta_k|| \times ||\Delta_{median}||}$
d. Cosine Penalty computation: $P_k = 1.0 - Cos_k$
e. Honest clients $\rightarrow$ Low Distance, Positive Angle ($Cos_k \ge 0$)
f. Byzantine clients $\rightarrow$ High Distance, Negative Angle ($Cos_k < 0$)

### **4. Step 4: Adaptive Thresholding & Filtering - $\boldsymbol{O(k)}$**
*(Applied prior to Credit Calculation for precise exclusion)*
a. Calculate Median of Distances, $E_{median}$ - $\boldsymbol{O(k)}$
b. Find the Deviation: $D_k = |E_k - E_{median}|$ - $\boldsymbol{O(k)}$
c. Find Median Absolute Deviation, $D = Median[D_k]$ - $\boldsymbol{O(k)}$
d. Compute adaptive multiplier $\kappa$ (Based on Warmup Phase & Variance) - $\boldsymbol{O(1)}$
e. Set Rejection Threshold T - $\boldsymbol{O(1)}$
   $T = E_{median} + (\kappa \times D)$
f. Apply Dual-Filtering Logic - $\boldsymbol{O(k)}$
   i. If $E_k > T$ **OR** $Cos_k < 0$ $\rightarrow$ Outlier, Discard
   ii. If $E_k \le T$ **AND** $Cos_k \ge 0$ $\rightarrow$ Honest, Accept.

### **5. Step 5: Volume Bounding & Credit Scores - $\boldsymbol{O(m)}$**
*(For the $m$ approved clients passing Step 4)*
a. Volume Bounding: Calculate global average data size $n_{avg}$.
b. Clip malicious reported bounds: $v_k = \min(n_k, 2.0 \times n_{avg})$
c. Initialize empty list for Credit Scores, $\beta_{k}$
d. Calculate Credit Scores (Combining Volume, Distance, and Penalty) - $\boldsymbol{O(m)}$
   $C_k = \frac{v_k}{E_k + (P_k \times 10.0) + \epsilon}$
e. Sum all approved credit scores: $C_{Total} = \Sigma C_k$ - $\boldsymbol{O(m)}$
f. Normalize to get strict percentages: $\beta_k^t = \frac{C_k}{C_{Total}}$ - $\boldsymbol{O(m)}$

### **6. Step 6: Aggregate & Finalize - $\boldsymbol{O(md)}$**
a. Aggregate approved clients via Weighted Sum: $W_{new} = \Sigma \beta_i \cdot W_i$
   - Multiply vector $W_i$ by scalar $\beta_i$: $\boldsymbol{O(d)}$
   - Sum across $m$ clients: $\boldsymbol{m \cdot O(d)}$ $\rightarrow$ worst case $\boldsymbol{O(kd)}$
b. **Server Momentum Update** *(to retain global inertia)*:
   $V^{t+1} = \mu V^t + (W_{new} - W_{global}^t)$
   $W_{global}^{t+1} = W_{global}^t + \alpha V^{t+1}$
