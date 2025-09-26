# Contributing to Hextral

Thank you for your interest in contributing to Hextral! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Feature Requests](#feature-requests)
- [Roadmap](#roadmap)

## Code of Conduct

This project adheres to the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Rust 1.70+ (latest stable recommended)
- Git
- Basic understanding of neural networks and machine learning

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hextral.git
   cd hextral
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/xStFtx/hextral.git
   ```

4. **Install dependencies**:
   ```bash
   cargo build
   ```

5. **Run tests** to ensure everything works:
   ```bash
   cargo test
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

#### **Bug Fixes**
- Fix existing bugs or issues
- Improve error handling
- Performance optimizations

#### **New Features**
- New activation functions
- Additional optimizers
- New loss functions
- Architecture improvements

#### **Documentation**
- Improve existing documentation
- Add code examples
- Write tutorials
- Fix typos or unclear explanations

#### **Testing**
- Add unit tests
- Add integration tests
- Improve test coverage
- Add benchmarks

#### **Code Quality**
- Refactor existing code
- Improve code organization
- Add helpful comments
- Optimize performance

### Contribution Workflow

1. **Check existing issues** - Look for issues labeled `good first issue` or `help wanted`
2. **Create a new issue** (if needed) - Discuss your proposed changes
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```
4. **Make your changes** following our coding standards
5. **Add tests** for new functionality
6. **Update documentation** if needed
7. **Run tests and linting**:
   ```bash
   cargo test
   cargo clippy
   cargo fmt
   ```
8. **Commit your changes** with clear commit messages
9. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
10. **Create a Pull Request** with a clear description

## Project Structure

```
hextral/
├── src/
│   ├── lib.rs          # Main library code
│   └── tests.rs        # Unit tests
├── examples/           # Example programs (to be added)
├── docs/              # Additional documentation (to be added)
├── Cargo.toml         # Project configuration
├── README.md          # Project overview
├── TODO.md            # Development roadmap
└── CONTRIBUTING.md    # This file
```

## Coding Standards

### Rust Style

- Follow standard Rust formatting: `cargo fmt`
- Use `cargo clippy` for linting
- Prefer explicit types over type inference when it improves readability
- Use meaningful variable and function names
- Add documentation comments for public APIs

### Code Organization

- Keep functions focused and single-purpose
- Use appropriate data structures (nalgebra types for linear algebra)
- Handle errors gracefully with proper error types
- Add inline comments for complex algorithms

### Async Programming Guidelines

Hextral v0.7.0 uses a clean async-first API design:

- **Async-first API** - All core methods are async by default: `train()`, `predict()`, `evaluate()`, etc.
- **Use intelligent yielding** - Only yield for large workloads (>1000 elements or >10 batches) to prevent unnecessary overhead
- **Leverage parallel processing** - Use `futures::join_all` for concurrent batch operations
- **Clean method naming** - No redundant suffixes, methods are async by default
- **Optimize performance** - Intelligent yielding ensures good performance for both small and large operations
- **Use tokio::task::yield_now()** - For cooperative multitasking in computationally intensive operations

### Async Code Style Examples

```rust
/// Async training method with intelligent yielding
pub async fn train(
    &mut self,
    inputs: &[DVector<f64>],
    targets: &[DVector<f64>],
    learning_rate: f64,
    epochs: usize,
    batch_size: Option<usize>,
) -> Vec<f64> {
    // Only yield for large workloads
    if inputs.len() > 1000 || epochs > 100 {
        tokio::task::yield_now().await;
    }
    // Training implementation...
}

/// Concurrent batch prediction
pub async fn predict_batch(&self, inputs: &[DVector<f64>]) -> Vec<DVector<f64>> {
    if inputs.len() > 10 {
        let futures: Vec<_> = inputs.iter()
            .map(|input| self.forward(input))
            .collect();
        join_all(futures).await
    } else {
        // Still use async but without overhead for small batches
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.forward(input).await);
        }
        results
    }
}
```

### Example Code Style

```rust
/// Computes the sigmoid activation function
/// 
/// # Arguments
/// * `x` - Input value
/// 
/// # Returns
/// The sigmoid of x: 1 / (1 + e^(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run benchmarks (when available)
cargo bench
```

### Writing Tests

- Add unit tests for new functions
- Test edge cases and error conditions
- Use descriptive test names
- Follow the pattern: `test_function_name_scenario`

### Test Examples

```rust
#[test]
fn test_sigmoid_activation() {
    let nn = Hextral::new(
        2, &[4], 1,
        ActivationFunction::Sigmoid,
        Optimizer::default()
    );
    
    let input = DVector::from_vec(vec![1.0, 0.0]);
    let output = nn.predict(&input);
    
    assert_eq!(output.len(), 1);
    assert!(output[0] >= 0.0 && output[0] <= 1.0);
}
```

## Documentation

### Code Documentation

- Use `///` for public API documentation
- Include examples in doc comments when helpful
- Document parameters and return values
- Use markdown formatting in doc comments

### README Updates

- Update README.md when adding new features
- Include usage examples for new functionality
- Update the feature list and version information

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review**
5. **Approval and merge**

## Issue Guidelines

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Windows, Linux, macOS]
- Rust version: [e.g. 1.70.0]
- Hextral version: [e.g. 0.5.1]

**Additional context**
Any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## Feature Requests

### High Priority Features

Check our [TODO.md](TODO.md) for current development priorities:

- **Learning Rate Schedulers** - StepLR, ExponentialLR, etc.
- **Model Serialization** - Save/load models with serde
- **Advanced Optimizers** - RMSprop, AdaGrad, AdamW
- **Early Stopping** - Prevent overfitting automatically
- **Comprehensive Examples** - Real-world use cases

### Feature Implementation Guidelines

1. **Discuss first** - Open an issue to discuss the feature
2. **Design review** - Get feedback on the proposed API
3. **Incremental implementation** - Break large features into smaller PRs
4. **Backward compatibility** - Maintain API compatibility when possible

## Roadmap

### Short Term (Next Release)
- Learning rate schedulers
- Model serialization
- Early stopping and checkpointing
- More comprehensive examples

### Medium Term
- Advanced optimizers (RMSprop, AdaGrad, etc.)
- Convolutional layers
- Recurrent layers (LSTM, GRU)
- GPU acceleration

### Long Term
- Attention mechanisms
- Transfer learning support
- AutoML capabilities
- Federated learning

## Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Email** - noskillz.exe@gmail.com for private matters

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Hextral!