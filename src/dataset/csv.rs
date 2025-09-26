use super::{Dataset, DatasetError, DatasetMetadata};
use csv::{ReaderBuilder, StringRecord, Trim};
use nalgebra::DVector;
use std::path::Path;

// CSV dataset loader
#[derive(Debug, Clone)]
pub struct CsvLoader {
    // Whether the CSV has headers
    pub has_headers: bool,
    // Column delimiter
    pub delimiter: u8,
    // Target column indices or names
    pub target_columns: TargetColumns,
    // Columns to skip
    pub skip_columns: Vec<String>,
    // Maximum number of rows to load
    pub max_rows: Option<usize>,
    // Infer data types automatically
    pub infer_types: bool,
    // Custom missing value representations
    pub missing_values: Vec<String>,
}

// Target columns specification
#[derive(Debug, Clone)]
pub enum TargetColumns {
    // No targets
    None,
    // Target columns by index
    Indices(Vec<usize>),
    // Target columns by name
    Names(Vec<String>),
    // Last N columns are targets
    Last(usize),
}

impl Default for CsvLoader {
    fn default() -> Self {
        Self {
            has_headers: true,
            delimiter: b',',
            target_columns: TargetColumns::None,
            skip_columns: Vec::new(),
            max_rows: None,
            infer_types: true,
            missing_values: vec![
                "".to_string(),
                "NA".to_string(),
                "N/A".to_string(),
                "null".to_string(),
                "NULL".to_string(),
                "nan".to_string(),
                "NaN".to_string(),
            ],
        }
    }
}

impl CsvLoader {
    // Create a new CSV loader
    pub fn new() -> Self {
        Self::default()
    }
    
    // Set whether the CSV has headers
    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }
    
    // Set the column delimiter
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }
    
    // Set target columns by indices
    pub fn with_target_indices(mut self, indices: Vec<usize>) -> Self {
        self.target_columns = TargetColumns::Indices(indices);
        self
    }
    
    // Set target columns by names
    pub fn with_target_names(mut self, names: Vec<String>) -> Self {
        self.target_columns = TargetColumns::Names(names);
        self
    }
    
    // Set last N columns as targets
    pub fn with_last_n_targets(mut self, n: usize) -> Self {
        self.target_columns = TargetColumns::Last(n);
        self
    }
    
    // Set columns to skip
    pub fn with_skip_columns(mut self, skip_columns: Vec<String>) -> Self {
        self.skip_columns = skip_columns;
        self
    }
    
    // Set maximum number of rows to load
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = Some(max_rows);
        self
    }
    
    /// Load CSV from file path
    pub async fn from_path<P: AsRef<Path>>(&self, path: P) -> Result<Dataset, DatasetError> {
        let content = tokio::fs::read_to_string(path.as_ref()).await?;
        self.from_string(&content).await
    }
    
    /// Load CSV from string data
    pub async fn from_string(&self, data: &str) -> Result<Dataset, DatasetError> {
        use std::io::Cursor;
        let cursor = Cursor::new(data.as_bytes());
        
        let mut csv_reader = ReaderBuilder::new()
            .has_headers(self.has_headers)
            .delimiter(self.delimiter)
            .trim(Trim::All)
            .from_reader(cursor);
        
        // Get headers if available
        let headers = if self.has_headers {
            match csv_reader.headers() {
                Ok(h) => Some(h.clone()),
                Err(e) => return Err(DatasetError::CsvError(format!("Failed to read headers: {}", e))),
            }
        } else {
            None
        };
        
        // Determine column layout
        let column_info = self.analyze_columns(&headers, &mut csv_reader)?;
        
        // Process records
        let (features, targets, _stats) = self.process_records(csv_reader, &column_info).await?;
        
        // Create dataset metadata
        let metadata = DatasetMetadata {
            sample_count: features.len(),
            feature_count: column_info.feature_columns.len(),
            target_count: if targets.is_some() { Some(column_info.target_columns.len()) } else { None },
            source: None,
            data_type: Some("CSV".to_string()),
        };
        
        let mut dataset = Dataset::new(features, targets);
        dataset.feature_names = Some(column_info.feature_names);
        dataset.target_names = column_info.target_names;
        dataset.metadata = metadata;
        
        Ok(dataset)
    }
    
    /// Analyze column structure and determine feature/target layout
    fn analyze_columns<R>(&self, headers: &Option<StringRecord>, csv_reader: &mut csv::Reader<R>) -> Result<ColumnInfo, DatasetError>
    where
        R: std::io::Read,
    {
        // Get column count from first record or headers
        let total_columns = if let Some(ref headers) = headers {
            headers.len()
        } else {
            // Read first record to determine column count
            let mut record = StringRecord::new();
            if let Err(e) = csv_reader.read_record(&mut record) {
                return Err(DatasetError::CsvError(format!("Failed to read CSV record: {}", e)));
            }
            if record.is_empty() {
                return Err(DatasetError::Validation("Empty CSV file".to_string()));
            }
            record.len()
        };
        
        // Determine target column indices
        let target_indices = match &self.target_columns {
            TargetColumns::None => Vec::new(),
            TargetColumns::Indices(indices) => indices.clone(),
            TargetColumns::Names(names) => {
                if let Some(ref headers) = headers {
                    let mut indices = Vec::new();
                    for name in names {
                        if let Some(index) = headers.iter().position(|h| h == name) {
                            indices.push(index);
                        } else {
                            return Err(DatasetError::Configuration(
                                format!("Target column '{}' not found in headers", name)
                            ));
                        }
                    }
                    indices
                } else {
                    return Err(DatasetError::Configuration(
                        "Cannot use target column names without headers".to_string()
                    ));
                }
            },
            TargetColumns::Last(n) => {
                if *n > total_columns {
                    return Err(DatasetError::Configuration(
                        format!("Cannot use last {} columns as targets, only {} columns available", n, total_columns)
                    ));
                }
                (total_columns - n..total_columns).collect()
            },
        };
        
        // Determine skip column indices
        let skip_indices: Vec<usize> = if !self.skip_columns.is_empty() {
            if let Some(ref headers) = headers {
                let mut indices = Vec::new();
                for name in &self.skip_columns {
                    if let Some(index) = headers.iter().position(|h| h == name) {
                        indices.push(index);
                    }
                }
                indices
            } else {
                Vec::new() // Can't skip by name without headers
            }
        } else {
            Vec::new()
        };
        
        // Build feature column indices (all columns except targets and skipped)
        let mut feature_indices = Vec::new();
        for i in 0..total_columns {
            if !target_indices.contains(&i) && !skip_indices.contains(&i) {
                feature_indices.push(i);
            }
        }
        
        // Build column names
        let feature_names = if let Some(ref headers) = headers {
            feature_indices.iter()
                .map(|&i| headers.get(i).unwrap_or(&format!("feature_{}", i)).to_string())
                .collect()
        } else {
            feature_indices.iter()
                .map(|i| format!("feature_{}", i))
                .collect()
        };
        
        let target_names = if !target_indices.is_empty() {
            Some(if let Some(ref headers) = headers {
                target_indices.iter()
                    .map(|&i| headers.get(i).unwrap_or(&format!("target_{}", i)).to_string())
                    .collect()
            } else {
                target_indices.iter()
                    .map(|i| format!("target_{}", i))
                    .collect()
            })
        } else {
            None
        };
        
        Ok(ColumnInfo {
            feature_columns: feature_indices,
            target_columns: target_indices,
            feature_names,
            target_names,
        })
    }
    
    /// Process CSV records and convert to vectors
    async fn process_records<R>(&self, mut csv_reader: csv::Reader<R>, column_info: &ColumnInfo) -> Result<(Vec<DVector<f64>>, Option<Vec<DVector<f64>>>, DatasetStats), DatasetError>
    where
        R: std::io::Read,
    {
        let mut features = Vec::new();
        let mut targets = if column_info.target_columns.is_empty() { None } else { Some(Vec::new()) };
        let mut record = StringRecord::new();
        let mut row_count = 0;
        
        while let Ok(has_record) = csv_reader.read_record(&mut record) {
            if !has_record {
                break;
            }
            // Check max rows limit
            if let Some(max_rows) = self.max_rows {
                if row_count >= max_rows {
                    break;
                }
            }
            
            // Parse feature values
            let mut feature_values = Vec::with_capacity(column_info.feature_columns.len());
            for &col_idx in &column_info.feature_columns {
                let value = record.get(col_idx).unwrap_or("").trim();
                let parsed_value = self.parse_value(value)?;
                feature_values.push(parsed_value);
            }
            features.push(DVector::from_vec(feature_values));
            
            // Parse target values if present
            if let Some(ref mut targets_vec) = targets {
                let mut target_values = Vec::with_capacity(column_info.target_columns.len());
                for &col_idx in &column_info.target_columns {
                    let value = record.get(col_idx).unwrap_or("").trim();
                    let parsed_value = self.parse_value(value)?;
                    target_values.push(parsed_value);
                }
                targets_vec.push(DVector::from_vec(target_values));
            }
            
            row_count += 1;
            
            // Yield periodically for large datasets
            if row_count % 10000 == 0 {
                tokio::task::yield_now().await;
            }
        }
        
        Ok((features, targets, DatasetStats {}))
    }
    
    /// Parse a single value from string to f64
    fn parse_value(&self, value: &str) -> Result<f64, DatasetError> {
        if self.missing_values.iter().any(|mv| mv == value) {
            Ok(0.0) // Replace missing values with 0.0
        } else if let Ok(parsed) = value.parse::<f64>() {
            Ok(parsed)
        } else {
            match value.to_lowercase().as_str() {
                "true" | "yes" | "1" | "y" => Ok(1.0),
                "false" | "no" | "0" | "n" => Ok(0.0),
                _ => Err(DatasetError::Parse(
                    format!("Cannot parse value '{}' as number", value)
                ))
            }
        }
    }
}

/// Information about column layout in the CSV
#[derive(Debug)]
struct ColumnInfo {
    feature_columns: Vec<usize>,
    target_columns: Vec<usize>,
    feature_names: Vec<String>,
    target_names: Option<Vec<String>>,
}

/// Dataset processing statistics
#[derive(Debug)]
struct DatasetStats {}