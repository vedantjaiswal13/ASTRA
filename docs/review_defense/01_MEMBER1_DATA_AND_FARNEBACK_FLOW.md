# ASTRA Defense Guide — Member 1
## Topic: Satellite Radiance Physics, Data Pipeline & Classical Optical Flow (Farneback)

---

### 1. Executive Responsibilities for Member 1

* **Module Ownership:**
  - NOAA GOES-16 Data Ingestion Engine: [`app/data/goes16_fetch.py`](file:///e:/SEM%203/ASTRA/astra-project/astra-backend/app/data/goes16_fetch.py)
  - Dataset Cataloging & Triplets: [`sample_data/triplets/`](file:///e:/SEM%203/ASTRA/astra-project/astra-backend/sample_data/triplets/) & [`sample_data/patches/`](file:///e:/SEM%203/ASTRA/astra-project/astra-backend/sample_data/patches/)
  - Baseline Linear Cross-Dissolve: [`app/processing/optical_flow.py` (`interpolate_linear`)](file:///e:/SEM%203/ASTRA/astra-project/astra-backend/app/processing/optical_flow.py#L220-L245)
  - Gunnar Farneback Multi-Scale Optical Flow: [`app/processing/optical_flow.py` (`compute_farneback_flow`)](file:///e:/SEM%203/ASTRA/astra-project/astra-backend/app/processing/optical_flow.py#L40-L95)
* **Demo Action on Screen:**
  - Open `http://localhost:5173/`.
  - Point to the top header: `NOAA GOES-16 ABI / Band 13 S3 LIVE` badge. Click it to show the **NOAA S3 Dataset Hub** and explain the 51 real observation triplets.
  - Select `Linear` method $\to$ show the double edges/ghosting in the middle frame.
  - Switch to `Farneback` $\to$ show how motion advection removes ghosting and tracks the wind vectors.

---

### 2. Theoretical Background & Equations

#### A. Satellite Sensor Physics: NOAA GOES-16 Band 13
* **Instrument:** Advanced Baseline Imager (ABI) on geostationary satellite GOES-16 (GOES-East stationed at $75.2^\circ\text{W}$ longitude, $35,786\text{ km}$ above the equator).
* **Product:** `ABI-L2-CMIPF` (Cloud and Moisture Imagery Full Disk, native resolution $5424 \times 5424$ pixels, 2 km nadir spatial sampling).
* **Spectral Band:** Band 13 Clean Infrared Window ($\lambda \approx 10.35\ \mu\text{m}$).
  - Clean IR window has minimal absorption by atmospheric water vapor or ozone.
  - Radiance is converted to **Brightness Temperature ($T_b$) in Kelvin** via Planck's Radiation Law inversion:
    $$T_b = \left[ \frac{h \cdot c}{k \cdot \lambda \cdot \ln\left( \frac{2 h c^2}{\lambda^5 R} + 1 \right)} - \beta \right] \cdot \frac{1}{\alpha}$$
  - **Physical Interpretation:**
    * Cold cloud tops (intense convective anvils, hurricane eyewalls): $T_b \approx 180\text{ K} - 220\text{ K}$ (appear bright white).
    * Warm ocean/land surfaces: $T_b \approx 280\text{ K} - 320\text{ K}$ (appear dark grayscale).
    * Outer space: Fill values ($\text{NaN}$ / $-1$), masked to zero.

#### B. The Baseline: Linear Cross-Dissolve Interpolation
The naive baseline computes intermediate pixel radiance as a time-weighted convex linear combination of boundary observations:
$$I_{\text{linear}}(x, y, t) = (1 - t) \cdot I_0(x, y) + t \cdot I_1(x, y) \quad \text{for } t \in [0.0, 1.0]$$

* **Why it fails on satellite clouds:**
  - Atmospheric clouds are fluid parcels moving across spatial coordinates $(x, y)$ due to advection by wind fields $\mathbf{u} = (u, v)$.
  - Linear cross-dissolve does not translate pixels. It fades out cloud $A$ at position $(x_1, y_1)$ while simultaneously fading in cloud $A$ at position $(x_2, y_2)$.
  - **Result:** Severe **ghosting**, **double edges**, and **synthetic transparency** where solid clouds appear semi-transparent. This corrupts automated cloud classification models and numerical weather prediction (NWP) data assimilation.

#### C. Gunnar Farneback Optical Flow (Polynomial Expansion)
* **Core Idea:** Gunnar Farneback (2003) approximates the local neighborhood of each pixel in two consecutive images by a continuous quadratic polynomial surface:
  $$f_1(\mathbf{x}) \approx \mathbf{x}^T A_1 \mathbf{x} + \mathbf{b}_1^T \mathbf{x} + c_1$$
  where $\mathbf{x} = [x, y]^T$, $A_1$ is a symmetric $2 \times 2$ matrix, $\mathbf{b}_1$ is a $2 \times 1$ vector, and $c_1$ is a scalar.
* **Under Translation $\mathbf{d}$:**
  If the image neighborhood undergoes a displacement $\mathbf{d} = (u, v)$, the new surface is:
  $$f_2(\mathbf{x}) = f_1(\mathbf{x} - \mathbf{d}) = (\mathbf{x} - \mathbf{d})^T A_1 (\mathbf{x} - \mathbf{d}) + \mathbf{b}_1^T (\mathbf{x} - \mathbf{d}) + c_1$$
  Expanding and equating coefficients with $f_2(\mathbf{x}) \approx \mathbf{x}^T A_2 \mathbf{x} + \mathbf{b}_2^T \mathbf{x} + c_2$:
  $$A_2 = A_1 = A(\mathbf{x})$$
  $$\mathbf{b}_2 = \mathbf{b}_1 - 2 A_1 \mathbf{d} \implies 2 A(\mathbf{x}) \mathbf{d} = -(\mathbf{b}_2 - \mathbf{b}_1) = \Delta \mathbf{b}(\mathbf{x})$$
* **Pointwise Solution:**
  $$\mathbf{d}(\mathbf{x}) = \frac{1}{2} A(\mathbf{x})^{-1} \Delta \mathbf{b}(\mathbf{x})$$
* **Multi-Scale Gaussian Pyramid:**
  To track large displacements (e.g., jet stream winds up to 30+ pixels over 20 minutes), Farneback constructs an image pyramid (`pyr_scale=0.5, levels=3, winsize=15, iterations=3`). Displacements are estimated at the coarse level and iteratively refined at native scale.

---

### 3. Quantitative Results & What We Improved

* **On Native $5424 \times 5424$ Full Disk Scans:**
  - Linear Cross-Dissolve: $\text{PSNR} = 29.19\text{ dB}$, $\text{SSIM} = 0.7814$, $\text{MSE} = 78.29$.
  - Farneback Optical Flow: $\text{PSNR} = 31.87\text{ dB}$, $\text{SSIM} = 0.8882$, $\text{MSE} = 42.23$.
  - **Improvement:** Farneback achieves a **46.06% reduction in MSE** and an increase of **+2.68 dB PSNR** over the linear baseline.
* **Why Farneback is limited:**
  - Farneback assumes brightness constancy ($\frac{dI}{dt} = 0$) and smooth quadratic deformation.
  - In tropical thunderstorms, cloud tops rapidly cool by $30\text{ K}$ within 10 minutes as convective updrafts punch into the stratosphere. Farneback cannot model this non-advective temperature change.

---

### 4. Top 5 Evaluator Review Questions & Exact Answers

#### Q1: "Why did you choose GOES-16 Band 13 instead of visible channels (Band 2)?"
> **Answer:** *"Sir, Band 2 is a visible channel ($0.64\ \mu\text{m}$) which is only usable during daylight hours; at night, visible imagery goes completely dark. Band 13 is a thermal infrared channel ($10.35\ \mu\text{m}$) that operates 24/7 continuously. Furthermore, Band 13 is in an atmospheric 'clean window' with minimal water vapor absorption, meaning its pixel values directly represent cloud-top brightness temperature according to Planck's Law. This enables continuous day-and-night forecasting."*

#### Q2: "What is the difference between Brightness Temperature and Radiance?"
> **Answer:** *"Radiance is the physical flux of electromagnetic energy received by the sensor in $\text{mW}/(\text{m}^2 \cdot \text{sr} \cdot \text{cm}^{-1})$. Brightness Temperature ($T_b$) is the equivalent temperature in Kelvin that a blackbody would need to radiate that exact radiance at $\lambda = 10.35\ \mu\text{m}$, obtained by inverting Planck's radiation equation with ABI calibration constants. In Band 13, cold temperatures directly correspond to high cloud tops."*

#### Q3: "How does Gunnar Farneback optical flow differ from Lucas-Kanade?"
> **Answer:** *"Lucas-Kanade uses a first-order Taylor expansion and assumes constant flow within a small window, which fails when displacement exceeds a few pixels. Gunnar Farneback uses a second-order quadratic polynomial approximation ($x^T A x + b^T x + c$) for the entire neighborhood. This quadratic formulation provides both curvature ($A$) and slope ($b$), allowing dense, pixel-by-pixel displacement vectors across large rotational and shearing flow fields via multi-level Gaussian pyramids."*

#### Q4: "Why does Linear Blending give double edges while Optical Flow does not?"
> **Answer:** *"Linear blending calculates $I(t) = (1-t)I_0 + tI_1$. Because pixel coordinates are fixed, a cloud that moved 10 pixels to the right appears simultaneously at its old position with opacity $(1-t)$ and at its new position with opacity $t$. Optical flow estimates the velocity vector $(u, v)$ and shifts the actual pixel intensities along the motion trajectory: $I(x, y, t) = I_0(x - t \cdot u, y - t \cdot v)$, preserving sharp cloud boundaries."*

#### Q5: "How are you fetching real GOES-16 data without API keys or AWS credentials?"
> **Answer:** *"We connect directly to NOAA's Open Data Dissemination (NODD) program on AWS S3 (`noaa-goes16`). This bucket is completely public and open. Our backend engine in `goes16_fetch.py` sends anonymous HTTPS REST requests to list and stream NetCDF-4 files. We then parse the internal HDF5 dataset structure using `h5py`, extract the CMI brightness temperature array and projection metadata, and normalize it into our 51 cataloged observation triplets."*
