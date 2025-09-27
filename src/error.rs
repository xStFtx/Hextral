use thiserror::Error;

/// Main error type for Hextral neural network operations
#[derive(Error, Debug)]
pub enum HextralError {
    #[error("Network configuration error: {message}")]
    Configuration { message: String },

    #[error("Training error: {message}")]
    Training { message: String },

    #[error("Prediction error: {message}")]
    Prediction { message: String },

    #[error("Invalid input: {context} - {details}")]
    InvalidInput {
        context: String,
        details: String,
        recoverable: bool,
    },

    #[error("Model serialization error: {source}")]
    Serialization {
        #[from]
        source: bincode::Error,
    },

    #[error("IO error during model operations: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("Mathematical computation error: {message}")]
    Computation { message: String },

    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("Numerical instability detected: {context}")]
    NumericalInstability { context: String },

    #[error("Resource exhaustion: {resource}")]
    ResourceExhaustion { resource: String },

    #[cfg(feature = "datasets")]
    #[error("Dataset error: {source}")]
    Dataset {
        #[from]
        source: crate::dataset::DatasetError,
    },
}

impl HextralError {
    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a training error
    pub fn training(message: impl Into<String>) -> Self {
        Self::Training {
            message: message.into(),
        }
    }

    /// Create a prediction error
    pub fn prediction(message: impl Into<String>) -> Self {
        Self::Prediction {
            message: message.into(),
        }
    }

    /// Create a computation error
    pub fn computation(message: impl Into<String>) -> Self {
        Self::Computation {
            message: message.into(),
        }
    }

    /// Create an invalid dimensions error
    pub fn invalid_dimensions(expected: usize, actual: usize) -> Self {
        Self::InvalidDimensions { expected, actual }
    }

    /// Create a numerical instability error
    pub fn numerical_instability(context: impl Into<String>) -> Self {
        Self::NumericalInstability {
            context: context.into(),
        }
    }

    /// Create a resource exhaustion error
    pub fn resource_exhaustion(resource: impl Into<String>) -> Self {
        Self::ResourceExhaustion {
            resource: resource.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            HextralError::Configuration { .. } => false,
            HextralError::Training { .. } => true,
            HextralError::Prediction { .. } => true,
            HextralError::InvalidInput { recoverable, .. } => *recoverable,
            HextralError::Serialization { .. } => false,
            HextralError::Io { .. } => true,
            HextralError::Computation { .. } => true,
            HextralError::InvalidDimensions { .. } => false,
            HextralError::NumericalInstability { .. } => true,
            HextralError::ResourceExhaustion { .. } => true,
            #[cfg(feature = "datasets")]
            HextralError::Dataset { source } => source.is_recoverable(),
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            HextralError::Configuration { .. } => ErrorSeverity::Critical,
            HextralError::Training { .. } => ErrorSeverity::Warning,
            HextralError::Prediction { .. } => ErrorSeverity::Error,
            HextralError::InvalidInput { recoverable, .. } => {
                if *recoverable {
                    ErrorSeverity::Warning
                } else {
                    ErrorSeverity::Error
                }
            }
            HextralError::Serialization { .. } => ErrorSeverity::Error,
            HextralError::Io { .. } => ErrorSeverity::Warning,
            HextralError::Computation { .. } => ErrorSeverity::Error,
            HextralError::InvalidDimensions { .. } => ErrorSeverity::Critical,
            HextralError::NumericalInstability { .. } => ErrorSeverity::Warning,
            HextralError::ResourceExhaustion { .. } => ErrorSeverity::Critical,
            #[cfg(feature = "datasets")]
            HextralError::Dataset { source } => source.severity(),
        }
    }

    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            HextralError::Training { .. } => vec![
                "Try reducing the learning rate".to_string(),
                "Add regularization to prevent overfitting".to_string(),
                "Check your training data for anomalies".to_string(),
            ],
            HextralError::NumericalInstability { .. } => vec![
                "Reduce learning rate".to_string(),
                "Enable gradient clipping".to_string(),
                "Use batch normalization".to_string(),
                "Check for NaN/Inf values in input data".to_string(),
            ],
            HextralError::ResourceExhaustion { .. } => vec![
                "Reduce batch size".to_string(),
                "Use data streaming instead of loading all data".to_string(),
                "Increase available memory".to_string(),
            ],
            _ => vec!["Review error details and documentation".to_string()],
        }
    }
}

/// Error severity levels for proper error handling and logging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Minor issues that don't affect operation
    Info,
    /// Issues that should be noted but don't stop execution
    Warning,
    /// Errors that stop current operation but allow recovery
    Error,
    /// Critical errors that require immediate attention
    Critical,
}

/// Result type alias for Hextral operations
pub type HextralResult<T> = Result<T, HextralError>;

/// Training context for better error reporting
#[derive(Debug, Clone)]
pub struct TrainingContext {
    pub epoch: usize,
    pub batch: Option<usize>,
    pub learning_rate: f64,
    pub loss: Option<f64>,
}

impl TrainingContext {
    pub fn new(epoch: usize, learning_rate: f64) -> Self {
        Self {
            epoch,
            batch: None,
            learning_rate,
            loss: None,
        }
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = Some(batch);
        self
    }

    pub fn with_loss(mut self, loss: f64) -> Self {
        self.loss = Some(loss);
        self
    }
}

/// Enhanced error reporting with context
#[derive(Debug)]
pub struct ContextualError {
    pub error: HextralError,
    pub context: Option<TrainingContext>,
    pub timestamp: std::time::SystemTime,
}

impl ContextualError {
    pub fn new(error: HextralError) -> Self {
        Self {
            error,
            context: None,
            timestamp: std::time::SystemTime::now(),
        }
    }

    pub fn with_context(mut self, context: TrainingContext) -> Self {
        self.context = Some(context);
        self
    }
}

impl std::fmt::Display for ContextualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)?;
        if let Some(ref ctx) = self.context {
            write!(
                f,
                " [Context: epoch={}, lr={}]",
                ctx.epoch, ctx.learning_rate
            )?;
            if let Some(batch) = ctx.batch {
                write!(f, " [batch={}]", batch)?;
            }
            if let Some(loss) = ctx.loss {
                write!(f, " [loss={:.6}]", loss)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// Conversion from Box<dyn std::error::Error> for compatibility with examples
impl From<Box<dyn std::error::Error + Send + Sync>> for HextralError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        HextralError::Configuration {
            message: err.to_string(),
        }
    }
}

impl From<Box<dyn std::error::Error>> for HextralError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        HextralError::Configuration {
            message: err.to_string(),
        }
    }
}
