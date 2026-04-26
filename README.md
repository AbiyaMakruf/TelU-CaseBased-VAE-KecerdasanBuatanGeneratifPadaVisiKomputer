# KL-VAE & VQ-VAE Implementation

Jupyter Notebook untuk implementasi dan perbandingan dua arsitektur Variational Autoencoder (VAE) pada dataset anime-faces.

## Deskripsi Project

Notebook ini mengimplementasikan:
- **KL-VAE**: VAE dengan latent space kontinyu menggunakan KL divergence
- **VQ-VAE**: VAE dengan latent space diskrit menggunakan Vector Quantization

Kedua model akan dilatih pada dataset anime-faces dan dibandingkan berdasarkan:
- Reconstruction Quality (MSE, PSNR)
- Latent/Codebook Structure
- Sampling Ability
- Inference Behavior

## Struktur Notebook

### 1. Environment Setup & GPU Configuration
- Import semua library yang diperlukan
- Deteksi dan konfigurasi GPU (CUDA) untuk akselerasi training

### 2. Data Acquisition & Dataset Setup
- Download dataset anime-faces dari URL
- Extract dan setup folder dataset
- Split data menjadi train (80%) dan test (20%)

### 3. Custom Dataset Class
- Implementasi `AnimeDataset` class untuk loading images
- Setup image transformations (resize, normalization)
- Create DataLoaders untuk training dan testing

### 4. KL-VAE Model Implementation
- CNN Encoder (output: latent mean & logvar)
- Reparameterization trick untuk sampling
- CNN Decoder untuk rekonstruksi image
- Latent dimension: 32

### 5. VQ-VAE Model Implementation
- VectorQuantizer class untuk quantization
- Encoder → quantized vectors dari codebook
- Codebook: 512 codes, dimension 64
- Decoder untuk rekonstruksi image

### 6. Training & Evaluation
- Loss functions (KL-VAE: reconstruction + KL divergence)
- Loss functions (VQ-VAE: reconstruction + commitment + codebook)
- Training loop dengan Adam optimizer dan learning rate scheduler
- Evaluasi menggunakan MSE dan PSNR metrics

### 7. Visualization
- Reconstruction visualization (Original vs KL-VAE vs VQ-VAE)
- Random sampling dari kedua model
- Latent space interpolation (optional)

### 8. Model Comparison Summary
- Tabel perbandingan karakteristik kedua model
- Reconstruction quality metrics
- Training characteristics
- Model parameter counts

### 9. Save Models
- Save trained models ke folder `models/`

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
pillow>=8.0.0
pandas>=1.1.0
```

## Hardware Requirements

- **GPU**: NVIDIA GPU dengan CUDA support (recommended)
- **GPU Memory**: Minimal 2GB VRAM (4GB+ recommended)
- **Disk Space**: ~500MB untuk dataset + checkpoint files

## Cara Menggunakan

1. **Install dependencies**:
```bash
pip install torch torchvision numpy matplotlib pillow pandas
```

2. **Buka Jupyter Notebook**:
```bash
jupyter notebook notebook.ipynb
```

3. **Run cells secara berurutan**:
   - Cell akan otomatis download dataset pada cell pertama
   - GPU akan dideteksi otomatis
   - Training akan berjalan ~20 epochs per model

4. **Waktu Training**:
   - Dengan GPU: ~30-60 menit total
   - Dengan CPU: ~2-4 jam total

## Output

Setelah menjalankan notebook, Anda akan mendapatkan:
- Dataset di folder `dataset/`
- Trained models di folder `models/`
- Loss curves visualization
- Reconstruction comparison images
- Random samples dari kedua model
- Detailed metrics dan comparison table

## Key Features

✓ **Full GPU Optimization**: Explicit CUDA device management
✓ **Comprehensive Implementation**: Complete training pipeline
✓ **Clear Documentation**: Docstrings di setiap fungsi
✓ **Visualization**: Plots dan image comparisons
✓ **Evaluation Metrics**: MSE dan PSNR calculation
✓ **Model Comparison**: Side-by-side analysis

## Hyperparameters

| Parameter | KL-VAE | VQ-VAE |
|-----------|--------|--------|
| Latent Dimension | 32 | 64 |
| Batch Size | 32 | 32 |
| Learning Rate | 1e-3 | 1e-3 |
| Epochs | 20 | 20 |
| Optimizer | Adam | Adam |
| Scheduler | StepLR (step=10) | StepLR (step=10) |
| Codebook Size | - | 512 |

## Architecture Details

### KL-VAE
- **Encoder**: 4 Conv layers (3→32→64→128→256) + 2 FC layers
- **Latent Space**: 32-dimensional continuous distribution
- **Decoder**: 4 ConvTranspose layers (256→128→64→32→3)
- **Total Parameters**: ~25M

### VQ-VAE
- **Encoder**: 4 Conv layers (3→32→64→128→64) + VQ layer
- **Quantizer**: 512 codes × 64 dimensions
- **Decoder**: 4 ConvTranspose layers (64→128→64→32→3)
- **Total Parameters**: ~16M

## Notes

- Notebook menggunakan torch.cuda jika tersedia, fallback ke CPU
- Dataset otomatis di-download pada first run
- Models dapat di-load kembali dari `models/` folder
- Visualization menggunakan matplotlib

## Troubleshooting

**Problem: CUDA out of memory**
- Reduce batch_size di data loading cell
- Reduce latent_dim untuk KL-VAE

**Problem: Dataset download fails**
- Check internet connection
- Download manually dari: https://storage.googleapis.com/learning-datasets/Resources/anime-faces.zip

**Problem: Slow training**
- Ensure GPU is being used (check first cell output)
- Reduce number of epochs untuk testing

## Author

Implementasi untuk tugas kuliah Kecerdasan Buatan Generatif pada Visi Komputer.