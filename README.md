## GANs with Spectral Normalization and Gradient Penalty
### TensorFlow implementation of ["The GAN Landscape: Losses, Architectures, Regularization, and Normalization"](https://arxiv.org/pdf/1807.04720.pdf)

## Architecture

* Generator: ResNet Style
  * Linear: 512 units
  * Unpooling: 2×2 pool size, 2×2 strides
  * Residual Block: 512 filters
    (Pre-Activation, with Batch Normalization, ReLU)
  * Unpooling: 2×2 pool size, 2×2 strides
  * Residual Block: 256 filters
    (Pre-Activation, with Batch Normalization, ReLU)
  * Unpooling: 2×2 pool size, 2×2 strides
  * Residual Block: 256 filters
    (Pre-Activation, with Batch Normalization, ReLU)
  * Unpooling: 2×2 pool size, 2×2 strides
  * Residual Block: 128 filters
    (Pre-Activation, with Batch Normalization, ReLU)
  * Unpooling: 2×2 pool size, 2×2 strides
  * Residual Block: 64 filters
    (Pre-Activation, with Batch Normalization, ReLU)
  * Batch Normalization
  * ReLU
  * Convolution: 3×3 kernel size, 3 filters
  * Sigmoid

* Discriminator: ResNet Style
  * Convolution: 3×3 kernel size, 64 filters
    (with Spectral Normalization)
  * Residual Block: 64 filters
    (Pre-Activation, with Spectral Normalization, ReLU)
  * Average Pooling: 2×2 pool size, 2×2 strides
  * Residual Block: 128 filters
    (Pre-Activation, with Spectral Normalization, ReLU)
  * Average Pooling: 2×2 pool size, 2×2 strides
  * Residual Block: 256 filters
    (Pre-Activation, with Spectral Normalization, ReLU)
  * Average Pooling: 2×2 pool size, 2×2 strides
  * Residual Block: 256 filters
    (Pre-Activation, with Spectral Normalization, ReLU)
  * Average Pooling: 2×2 pool size, 2×2 strides
  * Residual Block: 512 filters
    (Pre-Activation, with Spectral Normalization, ReLU)
  * Average Pooling: 2×2 pool size, 2×2 strides
  * Residual Block: 512 filters
    (Pre-Activation, with Spectral Normalization, ReLU)
  * Average Pooling: 2×2 pool size, 2×2 strides
  * ReLU
  * Global Average Pooling
  * Linear: 1 units
    (with Spectral Normalization)
  * Sigmoid

### Loss Function

* Non-Saturating (NS) GAN

### Regularization

* Gradient Penalty (WGAN-GP)

### Normalization

* Generator: Batch Normalization
* Discriminator: Spectral Normalization
