# ASTRA: Satellite Keyframe Interpolation Research Workbench
## Comprehensive Project Demonstration Script & Guide Defense Document

---

## Executive Overview for Project Guide / Evaluator

* **Project Title:** ASTRA (Advanced Satellite Temporal Reconstruction & Advection)
* **Domain:** Remote Sensing, Satellite Meteorology, Deep Learning, Computer Vision.
* **Core Problem:** Geostationary weather satellites (NOAA GOES-16) acquire full-disk imagery at 10-to-15 minute intervals. Rapid atmospheric phenomena—such as convective thunderstorm intensification, hurricane eyewall mesovortices, and frontal shear—evolve non-linearly between consecutive scans. Simple linear blending causes severe ghosting and double-edge blurring, obscuring physical cloud dynamics.
* **Our Solution:** An end-to-end framework integrating classical polynomial optical flow (**Farneback**), deep correlation volume optical flow (**RAFT**), and a deep convolutional residual refinement network (**PatchRefinementUNet**) to synthesize physically plausible intermediate satellite frames, validated against real held-out ground truth observations.
* **Dataset:** NOAA GOES-16 ABI-L2-CMIPF Band 13 (Clean IR Longwave Window, ~10.35 µm, full-disk scans at native $5424 \times 5424$ resolution).

---

## 1. Step-by-Step Live Demo Presentation Script

> **Tip for Delivery:** Speak at a steady pace. Keep the browser open at `http://localhost:5173/#` with the backend running on `localhost:8000`. Maintain the dashboard in full screen (`F11`).

---

### Act I: Introduction & Problem Context (Time: ~1.5 mins)

**What to Say:**
> *"Good morning/afternoon, Sir/Ma'am. Today, I am presenting **ASTRA**—our research workbench for Satellite Keyframe Interpolation.*
> 
> *In satellite meteorology, operational geostationary satellites like NOAA GOES-16 observe the Earth at discrete time intervals, typically every 10 to 15 minutes. While this cadence is sufficient for synoptic weather patterns, mesoscale convective systems and extreme meteorological events evolve rapidly between these scans.*
> 
> *If a forecaster or automated numerical weather prediction model needs intermediate observations—say at $t = 0.5$ (5 minutes in)—the simplest approach is linear cross-dissolving. However, clouds do not dissolve; they advect, deform, and undergo thermodynamic condensation and dissipation.*
> 
> *ASTRA was engineered to evaluate and bridge this gap by combining optical flow motion compensation with deep residual neural refinement, benchmarked against real satellite ground truth."*

---

### Act II: System Architecture & Data Pipeline (Time: ~2 mins)

**What to Point Out on Screen:**
* Point to the **Top Header**: Sensor identifier `NOAA GOES-16 ABI / Band 13 (10.35 µm)` and active UTC timestamp.
* Point to the **Left Sidebar**: Active sensor connection status (`Connected (GOES-16 Ch. 13)`), Dataset Triplet selector, and navigation tabs.

**What to Say:**
> *"Let us inspect the system architecture:*
> 1. **Data Ingestion**: *We ingest native NOAA GOES-16 Band 13 radiance scans. Each full-disk scan is a massive $5424 \times 5424$ float array.*
> 2. **Triplet Organization**: *Each dataset instance is a chronological triplet: $T_0$ (Start, e.g. 18:00 UTC), $T_{\text{mid}}$ (Ground Truth verification frame, 18:10 UTC), and $T_1$ (End, 18:20 UTC).*
> 3. **Dynamic Catalog**: *The backend dynamically catalogs all available scan triplets via a RESTful API (`GET /triplets`). The dropdown on the header and sidebar allows switching between distinct atmospheric events seamlessly.*
> 4. **Methods Available**: *We have implemented four distinct synthesis pipelines: Linear cross-dissolve baseline, Farneback optical flow, RAFT deep neural flow, and RAFT combined with residual U-Net refinement."*

---

### Act III: Live Demonstration of Frame Reconstruction (Time: ~3 mins)

**Actions to Perform:**
1. Select the **Farneback** method button in the top header.
2. Click **Generate Frame** (or observe the already loaded synthesized frame).
3. Walk through the **Frame Trio** layout.

**What to Say:**
> *"Here on the **Frame Reconstruction** page, we see the three-panel comparison:*
> * On the left: **Past Keyframe $T_0$** (18:00 UTC).*
> * On the right: **Future Keyframe $T_1$** (18:20 UTC).*
> * In the center: **Our Synthesized Frame** at intermediate timestamp $t = 0.50$ (18:10 UTC).*
> 
> *Notice that instead of static imagery, we have an interactive temporal scrubber. We can adjust the time fraction $t$ anywhere from 0.0 to 1.0, or toggle the auto-play loop to visualize continuous fluid atmospheric motion."*

**Next Action:**
* Scroll down to **Section 2: Ground Truth Direct Verification & Error Residual Heatmap**.

**What to Point Out:**
* Compare the **Synthesized Frame** with the **Ground Truth Frame** and the **Residual Error Heatmap**.

**What to Say:**
> *"The critical research capability of ASTRA is quantitative validation. In Section 2, the center frame is placed side-by-side with the real held-out satellite observation taken at that exact minute.*
> 
> *The third panel displays a **per-pixel absolute residual error heatmap** rendered in the perceptually uniform Viridis colormap:*
> * Dark purple/blue denotes near-zero reconstruction error ($< 5\text{ K}$ equivalent).*
> * Bright yellow/green highlights regions where non-rigid convective cloud genesis took place between scans.*
> * Notice the **Confidence Map**: the backend computes bidirectional flow consistency to detect occlusions and convective divergence automatically."*

---

### Act IV: Multi-Method Comparison & Metric Panel (Time: ~2.5 mins)

**Actions to Perform:**
1. Switch algorithm to **Linear** by clicking `Linear` in the header $\to$ click `Generate Frame`.
2. Scroll to **Section 3: Scientific Quantitative Metrics Panel**.
3. Point out MSE, PSNR, SSIM, MAE.
4. Switch back to **Farneback** $\to$ point out the dramatic metric leap.

**What to Say:**
> *"Let us contrast the mathematical performance of the algorithms:*
> * Under **Linear Cross-Dissolve**, cloud edges become hazy due to superposition. On native full resolution, Linear yields an MSE of ~36.35 and PSNR of 32.53 dB.
> * When we switch to **Farneback Optical Flow**, the motion vectors track the tropospheric jet stream and cloud rotation. MSE drops to **18.61**—an immediate **48.8% reduction in error**, with PSNR jumping to **35.43 dB** and SSIM reaching **0.9465**.
> * Every metric displayed here is calculated directly from the real 2D matrix arrays on the server—zero hardcoded values."*

---

### Act V: Deep Dive: Residual Analysis View (Time: ~2 mins)

**Action to Perform:**
* In the left sidebar, click the **Residual Analysis** tab (`/residual`).

**What to Point Out:**
* The cloud-active ROI bounding box crop ($256 \times 256$).
* The side-by-side cropped Generated vs Ground Truth.
* The High-Dynamic-Range Heatmap with its explicit quantitative error scale.

**What to Say:**
> *"In full-disk imagery, quiet ocean regions can mask localized errors. The **Residual Analysis** tab isolates the most cloud-active convective cell via automated bounding-box localization.*
> 
> *Here, we see the localized cropped prediction against the ground truth and its high-contrast residual map. This allows meteorological researchers to inspect whether hurricane bands or storm fronts maintained structural morphology."*

---

### Act VI: Benchmark Logs Suite (Native Full-Disk Resolution) (Time: ~2 mins)

**Action to Perform:**
* In the left sidebar, click the **Benchmark Logs** tab (`/benchmark`).

**What to Point Out:**
* The badge: `5424 × 5424 px Native Full Resolution`.
* The comparison table with color-coded ranking.
* The 4-metric grouped comparison bar charts.

**What to Say:**
> *"Finally, we navigate to the **Benchmark Logs Suite**.*
> 
> *A key scientific requirement was that benchmark evaluations must not be run on downscaled previews, as downsampling artificially inflates SSIM by smoothing out high-frequency spatial gradients.*
> 
> *This benchmark page executes all four interpolation methods at the **full native resolution of 5424 × 5424 pixels** on the held-out observation:*
> 1. **Farneback** consistently leads on smooth advection across large synoptic scales.
> 2. **RAFT** provides robust correlation matching on non-rigid motion.
> 3. **RAFT + Refinement** demonstrates our trained deep residual neural network (`PatchRefinementUNet`) applying localized corrections.
> 
> *The interactive grouped bar charts below visually contrast MSE, PSNR, SSIM, and MAE across all algorithms simultaneously."*

---

### Act VII: Conclusion (Time: ~30 secs)

**What to Say:**
> *"In summary, ASTRA bridges satellite meteorology and computer vision. It provides an interactive, full-stack research workbench capable of handling massive multi-gigapixel satellite scans, benchmarking classical vs deep learning motion synthesis, and providing real-time quantitative validation.*
> 
> *Thank you, Sir/Ma'am. I am now open to any questions or architectural discussions."*

---

## 2. Technical Q&A / Guide Defense Preparation

Be prepared for the following technical and theoretical questions your guide or evaluator is likely to ask:

---

### Q1: Why did you choose GOES-16 Band 13 instead of visible channels (Band 2)?
**Answer:**
> *"Visible channels (Band 2, 0.64 µm) rely entirely on reflected solar irradiance and are completely unavailable at night. Band 13 (Clean IR Longwave Window at 10.35 µm) measures planetary thermal brightness temperature continuously, 24 hours a day, with minimal atmospheric water vapor absorption. This ensures our interpolation pipeline operates consistently for daytime and nighttime weather surveillance."*

---

### Q2: Why does Farneback outperform RAFT on the full-disk metrics?
**Answer:**
> *"There are two primary scientific reasons:*
> 1. *Domain Shift*: RAFT was pretrained on optical RGB flow datasets (FlyingChairs and Sintel) featuring rigid objects with Lambertian reflectance. Geostationary infrared satellite imagery exhibits non-Lambertian radiometric changes, cloud-top cooling/warming, and semi-transparent cloud boundaries.
> 2. *Smooth Advection Field*: Atmospheric cloud motion over large oceanic regions is predominantly smooth laminar advection governed by atmospheric fluid dynamics. Farneback's quadratic polynomial expansion regularizes this global advective field effectively, while RAFT's correlation volume can overfit high-frequency radiance noise in low-contrast cloud tops without meteorological domain fine-tuning."*

---

### Q3: How does your Refinement U-Net work, and how did you prevent Out-Of-Memory (OOM) crashes on 5424x5424 images?
**Answer:**
> *"Directly passing a $5424 \times 5424$ single-precision float tensor through a multi-scale U-Net requires over $12\text{ GB}$ of intermediate activation RAM, which causes memory exhaustion on standard hardware.*
> 
> *To solve this, we implemented **tiled inference** with overlapping receptive fields:*
> * The $5424 \times 5424$ frame is partitioned into $512 \times 512$ patches with a $32\text{ px}$ overlap.
> * Each patch is passed through `PatchRefinementUNet`, which outputs a localized residual correction $\Delta I$.
> * The overlapping boundaries are merged using smooth distance-weighted accumulation (`weight` matrix averaging) to eliminate visible seam artifacts.*
> * This keeps peak memory under $1.5\text{ GB}$ and matches the $256 \times 256$ receptive field of our training set."*

---

### Q4: Explain the mathematical formulation of your metrics (MSE, PSNR, SSIM, MAE).
**Answer:**
> * **MSE (Mean Squared Error):**
>   $$\text{MSE} = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W}(I_{\text{synth}}(i,j) - I_{\text{GT}}(i,j))^2$$
> * **PSNR (Peak Signal-to-Noise Ratio):**
>   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) \quad (\text{where } \text{MAX}_I = 255)$$
> * **MAE (Mean Absolute Error):**
>   $$\text{MAE} = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W}|I_{\text{synth}}(i,j) - I_{\text{GT}}(i,j)|$$
> * **SSIM (Structural Similarity Index):**
>   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
>   *Measures luminance, contrast, and structural correlation rather than raw pixel distance, aligning with human perceptual judgment of cloud structures.*

---

### Q5: How does the system handle triplets that were not part of the training set?
**Answer:**
> *"The backend is completely modular and decoupled:*
> * `GET /triplets` dynamically discovers any folder inside `sample_data/triplets/` that contains valid `.npy` or `.png` scans.
> * The `apply_refinement_model` inference is fully resolution-agnostic and uses frozen PyTorch checkpoint weights.
> * If an external or incoming triplet lacks a ground truth midpoint ($T_{\text{mid}}$), the API still synthesizes intermediate frames at any $t \in [0, 1]$, gracefully returning `metrics: null` without failing."*

---

### Q6: What is the optical flow confidence map, and how is it calculated?
**Answer:**
> *"The confidence map measures bidirectional flow consistency:*
> 1. *We forward-warp frame $T_0$ to time $t$: $I_{\text{warped0}} = \mathcal{W}(T_0, t \cdot \mathbf{u})$.*
> 2. *We backward-warp frame $T_1$ to time $t$: $I_{\text{warped1}} = \mathcal{W}(T_1, -(1-t) \cdot \mathbf{u})$.*
> 3. *The intensity residual is $R(x, y) = |I_{\text{warped0}}(x, y) - I_{\text{warped1}}(x, y)|$.*
> 4. *Confidence is computed via exponential decay:*
>    $$\text{Confidence}(x, y) = \exp\left(-\frac{R(x, y)}{\sigma}\right) \times 255$$
> *High confidence (near 255) indicates consistent physical advection. Low confidence highlights occlusions, cloud expansion, or motion boundaries."*

---

## 3. Quick Reference Card: Launching the System

```bash
# Terminal 1: Backend (FastAPI + PyTorch)
cd astra-backend
.venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend (Vite + React + Tailwind)
cd astra-frontend
npm run dev

# Open Browser:
http://localhost:5173/
```
