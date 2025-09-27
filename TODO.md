# Todo

## Performance

- [ ] Add GPU support. CUDA/opencl
- [x] ~~Optimize batch processing~~ (v0.8.0) **Memory-efficient batch processing with streaming support**
- [x] ~~Reduce memory footprint~~ (v0.8.0) **Memory pools, object reuse, and in-place operations**
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

## Configuration

- [x] Configuration builder & loader (v0.9.0) **HextralBuilder with TOML/YAML loading and env overrides**
- [x] Training config helpers (v0.9.0) **Automatic `EarlyStopping`/`CheckpointConfig` conversion**
- [ ] Configuration validation tooling **Schema checks, helpful diagnostics**
- [ ] Runtime reload support **Hot-reload networks when config changes**

## Code Quality

- [x] ~~Improve Rust idioms~~ (v0.6.0) **More idiomatic async implementation**
- [x] ~~Add comprehensive error handling~~ (v0.8.0) **Custom error types with recovery strategies and detailed messages**
- [x] ~~Improve documentation coverage~~ (v0.8.0) **Complete API documentation with examples**
- [x] ~~Add more unit tests~~ (v0.8.0) **Comprehensive test suite covering all features**

## Extras

- [ ] Example Methods
- [ ] Transfer Learning
- [ ] AutoML
- [ ] Federated Learning

## Recently Completed (v0.9.0)

- **Configuration-driven workflows** - Declarative network setup via `HextralConfig` and builder
- **Config example & tests** - Added `config_builder_demo` example and integration coverage
- **Training loop cleanup** - Reduced gradient cloning and improved backpropagation stability
- **Dataset validation safeguards** - Error out on mismatched inputs/targets during training and evaluation
