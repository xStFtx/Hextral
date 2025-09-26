use nalgebra::DVector;

#[cfg(feature = "datasets")]
pub mod csv;

#[cfg(feature = "datasets")]
pub mod image;

pub mod preprocessing;

/// Async trait for dataset loading operations
#[async_trait::async_trait]
pub trait DatasetLoader<T> {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Load a dataset from the given source
    async fn load(&self, source: T) -> Result<Dataset, Self::Error>;
    
    /// Load a dataset with custom preprocessing options
    async fn load_with_preprocessing(
        &self, 
        source: T, 
        preprocessing: &PreprocessingConfig
    ) -> Result<Dataset, Self::Error>;
}

/// A loaded dataset containing features and optional targets
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature vectors (inputs)
    pub features: Vec<DVector<f64>>,
    /// Target vectors (outputs) - optional for unsupervised learning
    pub targets: Option<Vec<DVector<f64>>>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Target names (if available)  
    pub target_names: Option<Vec<String>>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Dataset metadata and statistics
#[derive(Debug, Clone, Default)]
pub struct DatasetMetadata {
    /// Total number of samples
    pub sample_count: usize,
    /// Number of features per sample
    pub feature_count: usize,
    /// Number of target values per sample (if supervised)
    pub target_count: Option<usize>,
    /// Source information (file path, URL, etc.)
    pub source: Option<String>,
    /// Data type information
    pub data_type: Option<String>,
}

/// Configuration for data preprocessing operations
#[derive(Debug, Clone, Default)]
pub struct PreprocessingConfig {
    /// Normalize features to [0, 1] range
    pub normalize: bool,
    /// Standardize features (zero mean, unit variance)
    pub standardize: bool,
    /// One-hot encode categorical variables
    pub one_hot_encode: Vec<usize>, // Column indices to encode
    /// Fill missing values with specified strategy
    pub fill_missing: Option<FillStrategy>,
    /// Shuffle dataset samples
    pub shuffle: bool,
    /// Split dataset into train/validation/test sets
    pub split_ratios: Option<(f64, f64, f64)>, // (train, val, test)
}

/// Strategy for filling missing values
#[derive(Debug, Clone)]
pub enum FillStrategy {
    /// Fill with a constant value
    Constant(f64),
    /// Fill with mean of the column
    Mean,
    /// Fill with median of the column  
    Median,
    /// Fill with mode (most frequent value)
    Mode,
    /// Forward fill (use previous valid value)
    ForwardFill,
    /// Backward fill (use next valid value)
    BackwardFill,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(
        features: Vec<DVector<f64>>,
        targets: Option<Vec<DVector<f64>>>,
    ) -> Self {
        let sample_count = features.len();
        let feature_count = features.first().map(|f| f.len()).unwrap_or(0);
        let target_count = targets.as_ref().and_then(|t| t.first().map(|t| t.len()));
        
        Self {
            features,
            targets,
            feature_names: None,
            target_names: None,
            metadata: DatasetMetadata {
                sample_count,
                feature_count,
                target_count,
                source: None,
                data_type: None,
            },
        }
    }
    
    /// Get a subset of the dataset
    pub fn subset(&self, indices: &[usize]) -> Dataset {
        let features: Vec<DVector<f64>> = indices.iter()
            .filter_map(|&i| self.features.get(i).cloned())
            .collect();
            
        let targets = self.targets.as_ref().map(|targets| {
            indices.iter()
                .filter_map(|&i| targets.get(i).cloned())
                .collect()
        });
        
        let mut subset = Dataset::new(features, targets);
        subset.feature_names = self.feature_names.clone();
        subset.target_names = self.target_names.clone();
        subset.metadata.source = self.metadata.source.clone();
        subset.metadata.data_type = self.metadata.data_type.clone();
        
        subset
    }
    
    /// Split dataset into training and testing sets
    pub fn train_test_split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let n_samples = self.features.len();
        let n_train = (n_samples as f64 * train_ratio) as usize;
        
        let train_indices: Vec<usize> = (0..n_train).collect();
        let test_indices: Vec<usize> = (n_train..n_samples).collect();
        
        (self.subset(&train_indices), self.subset(&test_indices))
    }
    
    /// Get dataset statistics
    pub fn describe(&self) -> DatasetStats {
        DatasetStats::from_dataset(self)
    }
}

/// Dataset statistics and summary information
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub sample_count: usize,
    pub feature_count: usize,
    pub feature_stats: Vec<FeatureStats>,
}

#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub name: Option<String>,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub missing_count: usize,
}

impl DatasetStats {
    pub fn from_dataset(dataset: &Dataset) -> Self {
        let sample_count = dataset.features.len();
        let feature_count = dataset.metadata.feature_count;
        
        let mut feature_stats = Vec::with_capacity(feature_count);
        
        for feature_idx in 0..feature_count {
            let values: Vec<f64> = dataset.features.iter()
                .filter_map(|feature| feature.get(feature_idx).copied())
                .collect();
                
            if values.is_empty() {
                continue;
            }
            
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std = variance.sqrt();
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let missing_count = sample_count - values.len();
            
            let name = dataset.feature_names.as_ref()
                .and_then(|names| names.get(feature_idx))
                .cloned();
            
            feature_stats.push(FeatureStats {
                name,
                mean,
                std,
                min,
                max,
                missing_count,
            });
        }
        
        Self {
            sample_count,
            feature_count,
            feature_stats,
        }
    }
}

/// Error type for dataset operations
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Invalid configuration: {0}")]
    Configuration(String),
    
    #[error("Data validation error: {0}")]
    Validation(String),
    
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    #[error("Data corruption detected: {0}")]
    Corruption(String),
    
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    
    #[error("Schema mismatch: expected {expected}, found {actual}")]
    SchemaMismatch { expected: String, actual: String },
    
    #[cfg(feature = "datasets")]
    #[error("CSV error: {0}")]
    CsvError(String),
    
    #[cfg(feature = "datasets")]
    #[error("Image error: {0}")]
    Image(#[from] ::image::ImageError),
}

impl DatasetError {
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            DatasetError::Io(_) => true,
            DatasetError::Parse(_) => false,
            DatasetError::Configuration(_) => false,
            DatasetError::Validation(_) => false,
            DatasetError::Memory(_) => true,
            DatasetError::Corruption(_) => false,
            DatasetError::UnsupportedFormat(_) => false,
            DatasetError::SchemaMismatch { .. } => false,
            #[cfg(feature = "datasets")]
            DatasetError::CsvError(_) => false,
            #[cfg(feature = "datasets")]
            DatasetError::Image(_) => false,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> crate::error::ErrorSeverity {
        match self {
            DatasetError::Io(_) => crate::error::ErrorSeverity::Warning,
            DatasetError::Parse(_) => crate::error::ErrorSeverity::Error,
            DatasetError::Configuration(_) => crate::error::ErrorSeverity::Critical,
            DatasetError::Validation(_) => crate::error::ErrorSeverity::Error,
            DatasetError::Memory(_) => crate::error::ErrorSeverity::Critical,
            DatasetError::Corruption(_) => crate::error::ErrorSeverity::Critical,
            DatasetError::UnsupportedFormat(_) => crate::error::ErrorSeverity::Error,
            DatasetError::SchemaMismatch { .. } => crate::error::ErrorSeverity::Error,
            #[cfg(feature = "datasets")]
            DatasetError::CsvError(_) => crate::error::ErrorSeverity::Error,
            #[cfg(feature = "datasets")]
            DatasetError::Image(_) => crate::error::ErrorSeverity::Error,
        }
    }
}