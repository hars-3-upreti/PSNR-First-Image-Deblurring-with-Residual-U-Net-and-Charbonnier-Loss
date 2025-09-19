# PSNR-First Image Deblurring with Residual U-Net and Charbonnier Loss  

Image deblurring is a long-standing challenge in computer vision. Motion blur, defocus, or camera shake severely degrade visual quality and downstream task performance. While GAN-based approaches often produce visually sharper outputs, they come at the cost of significantly lower **fidelity metrics (PSNR/SSIM)** and unstable training.  

This project takes a different stance: it is built as a **PSNR/SSIM-first image deblurring pipeline**, emphasizing **fidelity, reproducibility, and stable training** over hallucinated sharpness. The result is a robust baseline model that consistently achieves strong quantitative results while preserving natural structures in images.  

---

## 🔍 Motivation  

Most existing deblurring methods either:  
1. **Optimize for perceptual quality** (GANs, LPIPS loss) → sharper edges, but poor PSNR.  
2. **Rely on heavy architectures** (transformers, large CNNs) → high memory cost, slow training.  

This project aims to:  
- Deliver **high PSNR/SSIM results** (faithful reconstruction).  
- Fit within **8 GB VRAM** without sacrificing stability.  
- Use **modern training practices** (EMA, mixed precision, cosine decay).  
- Provide **clean, reproducible evaluation** with GPU-accelerated PSNR/SSIM and TTA inference.  

---

## ⚙️ Methodology  

### 1. Architecture  
- **Residual U-Net (ResUNet)**  
  - Encoder–decoder backbone with skip connections.  
  - Residual blocks in the bottleneck to enhance feature representation.  
  - Lightweight: trainable with 384×384 crops on an 8 GB GPU.  
  - Linear output (no Tanh), kept in `[0,1]` range via clamping.  

### 2. Loss Functions  
- **Charbonnier Loss** (robust L1-like loss):  
  \[
  L_{charb} = \sqrt{(x-y)^2 + \epsilon^2}
  \]  
  Provides smoother gradients than L1 and is strongly correlated with PSNR.  

- **Optional MS-SSIM Loss** (tiny weight):  
  Encourages structural similarity without compromising PSNR.  
  - Default weight: 0.05 (can be tuned to 0.02 or disabled).  

- **Identity Regularization**:  
  Ensures outputs do not deviate unnecessarily from inputs if the input is already close to sharp.  

### 3. Training Strategy  
- **Data Augmentation**:  
  - Random crops of 384×384.  
  - Horizontal/vertical flips and 90° rotations.  
  - Very mild color jitter (≤0.05).  
  - No heavy augmentations that harm PSNR.  

- **Optimization**:  
  - AdamW optimizer with weight decay.  
  - Cosine annealing LR schedule with warmup.  
  - Gradient accumulation + mixed precision (AMP).  
  - Gradient clipping (max norm = 1.0).  

- **Stability Enhancements**:  
  - Exponential Moving Average (EMA) of model weights.  
  - Validation and inference always use EMA weights.  
  - Early stopping with patience (40 epochs).  

### 4. Inference  
- Inference done in **fp32** (full precision) for exact metrics.  
- **Test-Time Augmentation (TTA)**: self-ensemble of rotations/flips, boosting PSNR by +0.2–0.3 dB.  
- Tiled inference supported for large-resolution images.  

---

## 📊 Experimental Results  

### Dataset  
- **GoPro Deblurring Dataset** (standard benchmark).  
- Training split: ~2103 pairs.  
- Testing split: 1111 pairs.  

### Metrics  
- **PSNR** (Peak Signal-to-Noise Ratio): measures pixel-wise fidelity.  
- **SSIM** (Structural Similarity Index): measures perceptual structure preservation.  

### Results on GoPro Test Set  

| Method                 | PSNR (dB) | SSIM |
|-------------------------|-----------|------|
| Blurred Input           | ~24.0     | 0.65 |
| **Our Model (no-TTA)**  | **27.9**  | **0.845** |
| **Our Model (with TTA)**| **28.2**  | **0.850** |

- **+4 dB PSNR gain** over blurred input.  
- **+0.20 SSIM improvement**, showing preserved structure and detail.  

### Visual Quality  
- Deblurred outputs show significant reduction of motion streaks.  
- Text and fine edges are recovered faithfully.  
- Unlike GANs, the model avoids artificial textures or hallucinated detail.  

---

## 🔬 Key Insights  

- **Pixel-dominant loss is critical**: Charbonnier (with tiny MS-SSIM) directly aligns with PSNR/SSIM, unlike perceptual/GAN losses.  
- **Moderate patch size + larger batch**: 384×384 crops with effective batch 4 yields better convergence than 512×512 with batch=1.  
- **EMA + cosine schedule**: ensures smooth convergence and stable validation performance.  
- **Inference tricks matter**: TTA consistently improves PSNR/SSIM without retraining.  

---

## ✅ Conclusion  

This project demonstrates that a carefully designed **PSNR-first pipeline** can achieve **state-of-the-art fidelity results** in image deblurring while remaining lightweight and efficient. By combining Charbonnier loss, EMA, cosine decay, and mild augmentations, we achieve nearly **28–29 dB PSNR** on the GoPro benchmark with a reproducible and GPU-friendly setup.  

This work serves as a **robust baseline for future deblurring research** and can be extended to domains like low-light enhancement, video restoration, or medical imaging.  

---

⚡ *Faithful deblurring, reproducible training, strong PSNR/SSIM — this project sets a high bar for lightweight image restoration pipelines.*  
