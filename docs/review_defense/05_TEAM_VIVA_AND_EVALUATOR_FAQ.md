# ASTRA Review Defense & Viva Cheat Sheet
## Master Evaluator Questions, Live Demonstration Walkthrough & Key Numbers

---

### 1. Key Numbers Memorization Table (Every Team Member Must Know These)

| Metric / Specification | Baseline Linear | Gunnar Farneback | Deep RAFT Flow | RAFT + Refinement U-Net | Direct Neural Synthesis (POC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Peak SNR (PSNR)** | 29.19 dB | 31.87 dB | 33.42 dB | **34.21 dB** (Peak) | 26.25 dB (avg) <br> *(up to 30.14 dB)* |
| **Structural Similarity (SSIM)** | 0.7814 | 0.8882 | 0.9210 | **0.9465** (Peak) | 0.8020 (avg) <br> *(up to 0.8836)* |
| **Mean Absolute Error (MAE)** | 5.19 | 4.28 | 3.65 | **2.89** (Lowest) | 7.42 (avg) <br> *(down to 5.50)* |
| **MSE Error Improvement** | 0.0% (Baseline) | **46.1% reduction** | **57.3% reduction** | **63.1% reduction** | Generative, non-warped |
| **Parameter Count** | 0 (Heuristic) | 0 (Analytical) | ~5.3M params | **128K params** (Residual U-Net) | **1.93M params** (Generative U-Net) |
| **Inference Time (CPU)** | < 1 ms | ~45 ms | ~420 ms | ~460 ms | ~95 ms |

---

### 2. High-Stress Evaluator Questions & Tactical Answers

#### Evaluator Question 1:
> *"Why did you spend so much time on classical and deep optical flow if your end goal is neural synthesis and generative diffusion?"*

**Best Team Response (Delivered by Member 2 or Member 4):**
> *"Sir, that is precisely the core scientific finding of our research. In satellite meteorology, you cannot jump blindly into black-box generative models without establishing rigorous physical baselines. Optical flow represents pure fluid advection under the Navier-Stokes approximation. By systematically benchmarking Farneback and RAFT, we proved mathematically where advection succeeds (recovering 63% of motion error) and exactly where and why it breaks down—namely, thermodynamic phase changes, convective cloud genesis, and non-conservation of radiance. In fact, our Direct Synthesis generative model uses optical flow vectors as explicit guidance channels to bridge physics with deep learning."*

---

#### Evaluator Question 2:
> *"In your Refinement network, why did SSIM improve significantly (+0.0255) while PSNR only improved by +0.79 dB?"*

**Best Team Response (Delivered by Member 3):**
> *"Sir, PSNR is a global logarithmic function of Mean Squared Error. Because 90% of the Earth's full disk consists of smooth oceans or slow-moving cloud decks where optical flow is already near-perfect, global pixel MSE changes slowly. However, SSIM measures local structural covariance, high-frequency luminance gradients, and edge sharpness. The Refinement U-Net specifically targets localized warping tears and boundary creases along expanding convective storm tops. Fixing those sharp boundary tears dramatically restores structural fidelity—which SSIM captures immediately—even if the total area of the tear represents only a small fraction of all pixels."*

---

#### Evaluator Question 3:
> *"How do you prove that your metrics are real and not simulated or hardcoded?"*

**Best Team Response (Delivered by Member 1 or Member 4):**
> *"Sir, we invite you to inspect our live backend right now. Every metric displayed on the dashboard is computed dynamically on the server by the `calculate_metrics` function in `app/processing/optical_flow.py` using `scikit-image`'s `structural_similarity` and NumPy array operations directly against the authentic held-out $T_{\text{mid}}$ array loaded from disk. Furthermore, we can switch to any of our 51 real observation triplets from across 2024, and the server recalculates the exact float values in real-time."*

---

#### Evaluator Question 4:
> *"What is the physical meaning of the Viridis and Turbo residual heatmaps shown on your dashboard?"*

**Best Team Response (Delivered by Member 1 or Member 3):**
> *"Sir, the residual heatmap is the spatial difference $|I_{\text{synthesized}}(x, y) - I_{\text{ground\_truth}}(x, y)|$. In Band 13, pixel brightness corresponds to cloud-top temperature. Dark purple/blue indicates near-zero error ($< 3\text{ K}$ error), meaning our model accurately predicted the exact cloud location and temperature. Bright yellow/red regions highlight areas where intense thermodynamic condensation occurred between scans, creating new convective towers that were physically absent in $T_0$. This visually proves where advection holds and where generative modeling is required."*

---

#### Evaluator Question 5:
> *"Why do you have a badge saying 'PLANNED - END-SEM (not yet trained full diffusion)' on your Neural Synthesis tab?"*

**Best Team Response (Delivered by Member 4):**
> *"Sir, scientific honesty is paramount. In our original project proposal, our end-semester goal is a full Latent Diffusion Model (LDM) with a learned epistemic confidence head and a Structural Reliability Index (SRI). For this mid-semester evaluation, we have engineered, trained across 12 epochs, and integrated a 1.93M-parameter conditional generative U-Net (`DirectSynthesisUNet`) as a working proof-of-concept. It directly proves that non-warping generative synthesis works on 6 diverse meteorological regimes. We explicitly keep the planned badge visible to clearly distinguish what is completed today versus what is being scaled for the final semester."*

---

### 3. Step-by-Step Live Demo Runbook (What to Click on Screen)

1. **Start Screen:**
   - URL: `http://localhost:5173/` in full-screen (`F11`).
   - Notice the top header: Point to `SENSOR: NOAA GOES-16 ABI / Band 13 (10.35 µm)`.
   - Point to the green pulsing button: **`NOAA GOES-16 ABI / Band 13 S3 LIVE`**.
2. **Step 1 (Member 1): Open NOAA Dataset Hub:**
   - Click the **`S3 LIVE`** button.
   - Show the modal: AWS S3 `noaa-goes16 (us-east-1)` online with ~1s latency.
   - Show the **51 loaded observation triplets** spanning 10 months of 2024.
   - Click `2024-03-15 18:00 UTC` to load an active storm patch into the workbench.
3. **Step 2 (Member 1): Linear vs Farneback:**
   - Select `Linear` in header $\to$ point to the ghosting and semi-transparent cloud edges.
   - Select `Farneback` $\to$ show the motion vectors advecting the cloud, dropping error by 46%.
4. **Step 3 (Member 2): Deep RAFT Optical Flow:**
   - Select `RAFT` in header.
   - Point out the sharp, fine-grained vector field and how cirrus streaks align with wind shears.
   - Scroll to the Viridis heatmap: point out the small edge tearing artifacts on expanding storm fringes.
5. **Step 4 (Member 3): RAFT + Refinement U-Net:**
   - Select `RAFT + Refinement` in header.
   - Show the metric jump: **34.21 dB PSNR** and **0.9465 SSIM**.
   - Show how the residual error heatmap removes the yellow tearing streaks.
6. **Step 5 (Member 4): Neural Direct Synthesis & Sequences:**
   - In the left sidebar, click the **Neural Synthesis** tab.
   - Point to the **Required Badge**, the **Architecture Diagram**, the **Training Loss Curve**, and switch through the **6 Meteorological Scenario Tabs** (`Cyclone Eyewall`, `Severe Convective Anvil`, `Cold Cloud-Top Core`).
   - In the left sidebar, click the **Temporal Sequences** tab.
   - Click `Play Sequence` to demonstrate fluid, continuous video keyframe animation across $t=0.00$ to $t=1.00$!
