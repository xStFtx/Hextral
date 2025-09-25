# Todo

## Performance
- [ ] Add GPU support. CUDA/opencl
- [ ] Optimize batch processing
- [ ] Reduce memory footprint
- [x] ~~Multithreading/Async~~ (v0.6.0) ✅ **Complete async/await support with intelligent yielding**

## Data
- [ ] Add CSV , images and other datasets support
- [ ] Cross-Validation
- [ ] Data Augmentation
- [ ] Better Metrics (Accuracy, recall , F1, Confusion Matrix)

## Training
- [ ] Early Stopping
- [ ] Checkpoints
- [ ] Gradient Clipping
- [ ] Precision Training

## Optimizers
- [ ] RMSprop, AdaGrad, AdaDelta
- [ ] AdamW, Nadam
- [ ] Lion, Adabelief

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
- [ ] Model Serialization (Save/load models with serde)
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

## Recently Completed (v0.6.0)
- ✅ **Full async/await support** - Complete async API with train_async, predict_async, evaluate_async
- ✅ **Intelligent yielding** - Performance-optimized yielding only for large workloads
- ✅ **Concurrent batch processing** - Parallel predictions using futures::join_all
- ✅ **Code optimization** - Removed verbose AI-generated patterns, cleaner code
- ✅ **Async activation functions** - All activation functions support async operations
- ✅ **Performance improvements** - Smart async yielding prevents unnecessary overhead
