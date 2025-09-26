# Todo

## Performance
- [ ] Add GPU support. CUDA/opencl
- [ ] Optimize batch processing
- [ ] Reduce memory footprint
- [x] ~~Multithreading/Async~~ (v0.6.0) **Complete async/await support with intelligent yielding**

## Data
- [ ] Add CSV , images and other datasets support
- [ ] Cross-Validation
- [ ] Data Augmentation
- [ ] Better Metrics (Accuracy, recall , F1, Confusion Matrix)

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
