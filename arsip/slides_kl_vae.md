# KL-VAE: Variational Autoencoders with KL Divergence
## Lecture Slides (15 slides)

---

## Slide 1: Title Slide

**KL-VAE: Variational Autoencoders with KL Divergence**

Implementing and Analyzing Generative Models for Image Synthesis

*Course: Artificial Intelligence for Computer Vision*  
*Topic: Generative Modeling & Deep Learning*

---

## Slide 2: What is a Variational Autoencoder (VAE)?

**Definition:**
- A generative model that learns to encode data into a latent space
- Reconstructs data from sampled latent representations
- Probabilistic approach to unsupervised learning

**Key Characteristics:**
1. **Encoder**: Maps input → latent distribution (mean μ, variance σ²)
2. **Latent Space**: Continuous, normally distributed for generative capability
3. **Decoder**: Maps latent samples → reconstructed output
4. **Loss Function**: Reconstruction loss + regularization term (KL divergence)

**Advantages:**
- Interpretable latent representations
- Can sample new data by sampling from latent space
- Theoretical foundation in information theory

---

## Slide 3: VAE vs GAN Comparison

| Aspect | VAE | GAN |
|--------|-----|-----|
| **Architecture** | Encoder-Decoder | Generator-Discriminator |
| **Loss Function** | Reconstruction + KL divergence | Adversarial |
| **Latent Space** | Continuous, learned distribution | Random vectors |
| **Training** | More stable | Less stable, requires careful tuning |
| **Interpretability** | High (latent variables are meaningful) | Low (black box) |
| **Reconstruction Quality** | Good but may be blurry | Sharp but ignores input |
| **Sampling** | Smooth transitions | Discrete artifacts |
| **Computational Cost** | Lower | Higher (dual networks) |

---

## Slide 4: KL-VAE Architecture Overview

**Three Main Components:**

1. **Encoder Network (CNN)**
   - Input: Image (3×64×64)
   - Layers: Conv2d with ReLU activations
   - Output: Mean (μ) and log-variance (logσ²) vectors
   - Latent dimension: 32D

2. **Latent Space**
   - Continuous, Gaussian-distributed space
   - Allows smooth interpolation between data points
   - Enables generation of new samples

3. **Decoder Network (Transposed CNN)**
   - Input: Latent vector z (sampled or mean)
   - Layers: ConvTranspose2d with ReLU and final Tanh
   - Output: Reconstructed image (3×64×64)

**Data Flow:**
```
Image → Encoder → (μ, logσ²) → Reparameterize → z → Decoder → Reconstructed Image
```

---

## Slide 5: The Reparameterization Trick

**Problem:**
- Sampling from distribution is non-differentiable
- Cannot backpropagate through sampling operation

**Solution - Reparameterization Trick:**
```
z = μ + σ ⊙ ε  where ε ~ N(0, I)
```
- Instead of sampling directly from N(μ, σ²)
- Sample noise ε from standard normal N(0, I)
- Scale by σ and add μ
- Now differentiable with respect to μ and σ!

**Benefits:**
- Enables gradient-based optimization
- Allows backpropagation through sampling
- Key innovation that makes VAEs trainable

---

## Slide 6: Loss Function - Part 1

**VAE Loss consists of two terms:**

1. **Reconstruction Loss** (L_recon)
   - Measures how well decoder reconstructs input
   - Formula: L1 or MSE loss
   - `L_recon = ||x - x_reconstructed||`

2. **KL Divergence Loss** (L_KL)
   - Regularization term
   - Pushes learned distribution toward prior N(0, I)
   - Formula: `L_KL = -0.5 * Σ(1 + logσ² - μ² - e^(logσ²))`

**Total Loss:**
```
L_total = L_recon + β * L_KL

where β is a weighting factor (often β=1)
```

**Intuition:**
- Reconstruction loss: "Accuracy" - reconstruct input accurately
- KL divergence: "Regularity" - maintain probabilistic structure

---

## Slide 7: Understanding KL Divergence

**What is KL Divergence?**
- Measure of how one probability distribution differs from another
- Always non-negative: KL(P||Q) ≥ 0
- Equals 0 only if P and Q are identical

**In VAE Context:**
- Compares learned posterior q(z|x) with prior p(z) = N(0, I)
- Encourages latent distribution to be "simple"
- Prevents the model from encoding all information without compression

**Effect on Training:**
- Small KL loss → latent space close to N(0, I) → better sampling
- Large reconstruction loss → latent space can encode anything → perfect reconstruction
- Balance between two creates meaningful representations

**Visual Interpretation:**
```
KL loss = 0     : Perfect match with N(0,I)
KL loss = high  : Different from N(0,I) (overfitting)
Optimal         : Balance between reconstruction and regularity
```

---

## Slide 8: VQ-VAE Architecture

**Vector Quantization-VAE:** Discrete alternative to KL-VAE

**Three Components:**

1. **Encoder**
   - Input: Image (3×64×64)
   - Output: Continuous encoding vectors (4×4×64)

2. **Vector Quantizer**
   - Codebook: 512 discrete codes, each 64-dimensional
   - Maps continuous vectors → nearest codebook entry
   - Creates discrete latent representation

3. **Decoder**
   - Input: Quantized vectors (same shape as encoder output)
   - Output: Reconstructed image (3×64×64)

**Key Difference from KL-VAE:**
- **Discrete** latent space instead of continuous
- No probabilistic sampling
- No KL divergence regularization

---

## Slide 9: Vector Quantization Explained

**Process:**
```
For each encoded vector e:
1. Compute distance to all codebook entries: d = ||e - c_i||²
2. Find nearest entry: i* = argmin(d)
3. Replace e with codebook entry: q = c[i*]
```

**Loss Function:**
```
L_total = L_recon + L_commitment + L_codebook

Where:
- L_recon: Reconstruction error
- L_commitment: ||sg[e] - c||² (commit encoder to codebook)
- L_codebook: ||e - sg[c]||² (move codebook toward encoder)
```

**Advantages:**
- Discrete structure enables compression
- Codebook entries can be analyzed/edited
- Stable training without KL annealing

**Challenges:**
- Codebook collapse (unused entries)
- Non-differentiable quantization

---

## Slide 10: KL-VAE vs VQ-VAE Comparison

| Property | KL-VAE | VQ-VAE |
|----------|--------|--------|
| **Latent Space Type** | Continuous (32D) | Discrete (512 codes) |
| **Distribution** | Gaussian N(0,I) | Empirical |
| **Interpolation** | Smooth, linear blend | Discrete transitions |
| **Sampling Quality** | Smooth, blurry | Sharp but artifacts |
| **Parameters** | 1,777,411 | 626,179 ✓ Fewer |
| **Training Stability** | Moderate | Stable ✓ |
| **Interpretability** | High (latent dims) | High (codebook entries) |
| **Use Case** | Generation & interpolation | Compression & discrete codes |

**Conclusion:**
- VQ-VAE: Better for reconstruction and compression
- KL-VAE: Better for sampling and generative modeling

---

## Slide 11: Training Results

**Experimental Setup:**
- Dataset: Anime faces (63,565 images)
- Train/Test split: 80% / 20%
- Batch size: 32
- Epochs: 5
- Optimizer: Adam (lr=0.001)
- GPU: NVIDIA RTX 5070 Ti

**Training Performance:**

| Model | Time | Final Loss | Epochs | Avg/Epoch |
|-------|------|-----------|--------|-----------|
| KL-VAE | 14.6s | 0.3810 | 5 | 2.92s |
| VQ-VAE | 13.1s | 0.2527 | 5 | 2.62s |

**Loss Curves:**
- Both models converge smoothly
- KL-VAE stabilizes around epoch 2
- VQ-VAE shows faster descent
- No signs of instability or divergence

---

## Slide 12: Reconstruction Quality

**Quality Metrics:**

| Model | MSE | PSNR | Quality |
|-------|-----|------|---------|
| KL-VAE | 0.193525 | 13.1630dB | Good |
| VQ-VAE | 0.078507 | 17.0781dB | **Excellent** |

**Performance Analysis:**
- VQ-VAE achieves **~54% lower MSE** than KL-VAE
- PSNR (Peak Signal-to-Noise Ratio) shows VQ-VAE superiority
- Higher PSNR = better reconstruction quality

**Visual Observations:**
- KL-VAE reconstructions: Slightly blurry but natural
- VQ-VAE reconstructions: Sharp with good detail preservation
- Both preserve overall image structure well

**Interpretation:**
- Discrete quantization helps preserve details
- Continuous space may sacrifice sharpness for smoothness
- Trade-off between reconstruction accuracy and generative capability

---

## Slide 13: Latent Space Visualization

**t-SNE Analysis (KL-VAE):**
- 1,272 test images → 1,272 latent vectors (32D)
- Dimensionality reduction: 32D → 2D via t-SNE
- Perplexity: 30, iterations: 1000

**Observations:**
- Clear clustering patterns emerge
- Similar images group together in latent space
- No distinct classes but natural grouping
- Smooth transitions between clusters suggest good interpolation potential

**Codebook Analysis (VQ-VAE):**
- Total codes: 512
- Used codes: 300-380 (60-75% utilization)
- Most used code: Frequency ~50-100 per code
- Distribution shows diversity in learned representations

**Interpretation:**
- KL-VAE latent space is interpretable and smooth
- VQ-VAE uses diverse codebook (no collapse issues)
- Both models learn meaningful representations

---

## Slide 14: Codebook Utilization & Analysis

**VQ-VAE Codebook Statistics:**

```
Total Codebook Entries: 512
Used Entries: ~60-75% (varies per batch)
Most Frequent Code: ID varies, frequency ~50-100
Average Usage: Uniform distribution
Codebook Collapse: Not observed
```

**Findings:**
1. **Diverse Utilization**: Most codes are used
2. **No Collapse**: Unused codes remain available
3. **Balanced Usage**: No single code dominates
4. **Stability**: Usage patterns consistent across batches

**Implications:**
- Codebook learning is healthy and efficient
- Model explores representation space fully
- Appropriate code dimension (64D) for content encoding
- Suitable number of codes (512) for this dataset

**Applications:**
- Codebook can be clustered for semantic analysis
- Entries can be interpolated for smooth transitions
- Discrete nature enables vector arithmetic

---

## Slide 15: Conclusion & Future Work

**Summary:**
- ✓ Implemented both KL-VAE and VQ-VAE successfully
- ✓ KL-VAE: Better for sampling and smooth generation
- ✓ VQ-VAE: Better for reconstruction and compression
- ✓ Both models stable, efficient, and well-trained

**Key Takeaways:**
1. **VAEs are powerful** generative models for unsupervised learning
2. **Reparameterization trick** makes VAEs trainable and practical
3. **Trade-offs exist** between reconstruction and regularity
4. **Multiple architectures** (continuous vs discrete) serve different purposes

**Future Research Directions:**

1. **VQ-VAE-2**: Hierarchical quantization for multi-scale representations
2. **Disentangled VAE**: Learn interpretable, independent latent factors
3. **β-VAE**: Weighted KL divergence for controlling trade-offs
4. **Generative Flow**: Combine VAE with normalizing flows
5. **Cross-Domain**: Apply to audio, text, or multimodal data
6. **Comparison**: Full GAN comparison with same architecture

**Practical Applications:**
- Image compression and transmission
- Data augmentation for downstream tasks
- Anomaly detection in image data
- Style transfer and content manipulation
- Semi-supervised learning with unlabeled data

---

*End of Lecture Slides - Total: 15 slides*
