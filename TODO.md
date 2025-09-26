# Todo

## Performance
- [ ] Add GPU support. CUDA/opencl
- [ ] Optimize batch processing
- [ ] Reduce memory footprint
- [x] ~~Multithreading/Async~~ (v0.6.0) **Complete async/await support with intelligent yielding**

## Data
- [x] ~~Add CSV , images and other datasets support~~ (v0.8.0) **Complete dataset loading system with CSV, image, preprocessing, and PCA**
- [x] ~~Data Preprocessing Pipeline~~ (v0.8.0) **Normalization, standardization, one-hot encoding, missing values, outlier removal**
- [x] ~~Data Augmentation~~ (v0.8.0) **Image augmentation: flip, rotation, brightness, contrast, noise**
- [x] ~~Missing Value Handling~~ (v0.8.0) **Forward/backward fill, mean, median, mode, constant strategies**
- [ ] Cross-Validation
- [ ] Better Metrics (Accuracy, recall , F1, Confusion Matrix)
- [ ] Time Series Dataset Support
- [ ] Text/NLP Dataset Support

## Training
- [x] ~~Early Stopping~~ (v0.7.0) **Early stopping with patience, validation loss monitoring**
- [x] ~~Checkpoints~~ (v0.7.0) **Model checkpointing with bincode serialization**
- [ ] Gradient Clipping
- [ ] Precision Training

## Optimizers
- [x] ~~RMSprop, AdaGrad, AdaDelta~~ (v0.7.0) **Complete implementation with proper momentum and adaptive learning**
- [x] ~~AdamW, Nadam~~ (v0.7.0) **Advanced Adam variants with weight decay and Nesterov momentum**
- [x] ~~Lion, Adabelief~~ (v0.7.0) **Modern optimizers: sign-based Lion and centralized AdaBelief**

## Learning Rate Scheduling
- [ ] StepLR, ExponentialLR
- [ ] CosineAnnealing, ReduceLROnPlateau
- [ ] Warmup Schedulers

## Architectures
- [ ] Convolutional Layers
- [ ] Recurrent Layers
- [ ] Attention Mechanisms
- [ ] Residual Connections

## QoL
- [x] ~~Model Serialization (Save/load models with serde)~~ (v0.7.0) **Model checkpointing with bincode serialization**
- [ ] Visualization
- [ ] Hyperparameter Tuning
- [ ] Real World Examples of Usage for Hextral

## Code Quality
- [x] ~~Remove AI-generated verbose patterns~~ (v0.6.0)  **Optimized code, removed excessive comments**
- [x] ~~Improve Rust idioms~~ (v0.6.0) **More idiomatic async implementation**
- [ ] Add comprehensive error handling
- [ ] Improve documentation coverage
- [ ] Add more unit tests

## Extras
- [ ] Example Methods
- [ ] Transfer Learning
- [ ] AutoML
- [ ] Federated Learning

## Recently Completed (v0.8.0)
- **Complete Dataset Loading System** - CSV and image dataset loaders with async-first API
- **Comprehensive Data Preprocessing** - Normalization, standardization, one-hot encoding with dynamic category discovery
- **Advanced Missing Value Handling** - Forward/backward fill, mean, median, mode, and constant strategies
- **Principal Component Analysis (PCA)** - Full PCA implementation with power iteration and matrix deflation
- **Image Data Augmentation** - Flip, rotation, brightness, contrast, and noise adjustments with proper pixel manipulation
- **Advanced Label Extraction** - Multiple strategies for filename patterns, directory structure, and manual mapping
- **Outlier Detection and Removal** - Statistical outlier removal using IQR method with configurable thresholds
- **Polynomial Feature Engineering** - Automated polynomial feature expansion for improved model capacity
- **Organized Checkpoint System** - Structured checkpoint storage with proper .gitignore configuration
- **Production-Ready Code Quality** - Removed all placeholders, implemented complete functionality, eliminated AI-generated patterns

## Recently Completed (v0.7.0)
- **Clean async-first API** - Removed redundant _sync and _async method suffixes
- **Early stopping** - Configurable early stopping with patience and validation loss monitoring
- **Model checkpointing** - Save and load models with bincode serialization support
- **Enhanced optimizers** - Complete set of 12 modern optimizers with proper implementations
- **Extended activation functions** - 9 activation functions including advanced options
- **Improved loss functions** - 5 comprehensive loss functions for different use cases
- **Breaking changes** - Simplified API with cleaner method names and consistent async patterns
- **Documentation overhaul** - Updated all examples and documentation for new API

## Recently Completed (v0.6.0)
- **Full async/await support** - Complete async API with train, predict, evaluate
- **Intelligent yielding** - Performance-optimized yielding only for large workloads
- **Concurrent batch processing** - Parallel predictions using futures::join_all
- **Code optimization** - Removed verbose AI-generated patterns, cleaner code
- **Async activation functions** - All activation functions support async operations
- **Performance improvements** - Smart async yielding prevents unnecessary overhead
- **Quaternion activation function** - Normalizes quaternion inputs to unit quaternions
- **10 optimizer implementations** - Complete set including Adam, AdamW, Lion, AdaBelief, etc.
