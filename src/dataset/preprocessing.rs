use super::{Dataset, DatasetError, FillStrategy};
use nalgebra::DVector;
use rand::{seq::SliceRandom, thread_rng};
use std::collections::HashMap;

// Preprocessing pipeline for datasets
pub struct Preprocessor {
    operations: Vec<PreprocessingOp>,
}

// Preprocessing operation
#[derive(Debug, Clone)]
pub enum PreprocessingOp {
    // Normalize features to [0, 1] range
    Normalize { feature_indices: Option<Vec<usize>> },
    // Standardize features (zero mean, unit variance)
    Standardize { feature_indices: Option<Vec<usize>> },
    // One-hot encode categorical features
    OneHotEncode { feature_indices: Vec<usize> },
    // Fill missing values
    FillMissing { strategy: FillStrategy },
    // Remove outliers using IQR method
    RemoveOutliers { factor: f64 },
    // Apply polynomial features
    PolynomialFeatures { degree: usize },
    // Principal Component Analysis
    PCA { components: usize },
}

// Statistics from training data for preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessingStats {
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
    pub feature_mins: Vec<f64>,
    pub feature_maxs: Vec<f64>,
    pub categorical_mappings: HashMap<usize, HashMap<String, usize>>,
    pub pca_components: Option<Vec<DVector<f64>>>,
}

impl Preprocessor {
    // Create a new preprocessing pipeline
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    // Add normalization
    pub fn normalize(mut self, feature_indices: Option<Vec<usize>>) -> Self {
        self.operations
            .push(PreprocessingOp::Normalize { feature_indices });
        self
    }

    // Add standardization
    pub fn standardize(mut self, feature_indices: Option<Vec<usize>>) -> Self {
        self.operations
            .push(PreprocessingOp::Standardize { feature_indices });
        self
    }

    // Add one-hot encoding
    pub fn one_hot_encode(mut self, feature_indices: Vec<usize>) -> Self {
        self.operations
            .push(PreprocessingOp::OneHotEncode { feature_indices });
        self
    }

    // Add missing value filling
    pub fn fill_missing(mut self, strategy: FillStrategy) -> Self {
        self.operations
            .push(PreprocessingOp::FillMissing { strategy });
        self
    }

    // Add outlier removal
    pub fn remove_outliers(mut self, factor: f64) -> Self {
        self.operations
            .push(PreprocessingOp::RemoveOutliers { factor });
        self
    }

    // Fit pipeline and apply transformations
    pub async fn fit_transform(
        &self,
        dataset: &mut Dataset,
    ) -> Result<PreprocessingStats, DatasetError> {
        let stats = self.fit(dataset).await?;
        self.apply_with_stats(dataset, &stats).await?;
        Ok(stats)
    }

    // Fit preprocessing parameters only
    pub async fn fit(&self, dataset: &Dataset) -> Result<PreprocessingStats, DatasetError> {
        let mut stats = PreprocessingStats {
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
            feature_mins: Vec::new(),
            feature_maxs: Vec::new(),
            categorical_mappings: HashMap::new(),
            pca_components: None,
        };

        if dataset.features.is_empty() {
            return Ok(stats);
        }

        let feature_count = dataset.features[0].len();

        // Basic statistics
        stats.feature_means = vec![0.0; feature_count];
        stats.feature_stds = vec![0.0; feature_count];
        stats.feature_mins = vec![f64::INFINITY; feature_count];
        stats.feature_maxs = vec![f64::NEG_INFINITY; feature_count];

        // Means, mins, maxs
        let sample_count = dataset.features.len() as f64;
        for feature in &dataset.features {
            for (i, &value) in feature.iter().enumerate() {
                stats.feature_means[i] += value;
                stats.feature_mins[i] = stats.feature_mins[i].min(value);
                stats.feature_maxs[i] = stats.feature_maxs[i].max(value);
            }
        }

        // Finalize means
        for mean in &mut stats.feature_means {
            *mean /= sample_count;
        }

        // Compute standard deviations
        for feature in &dataset.features {
            for (i, &value) in feature.iter().enumerate() {
                let diff = value - stats.feature_means[i];
                stats.feature_stds[i] += diff * diff;
            }
        }

        for std in stats.feature_stds.iter_mut() {
            *std = (*std / sample_count).sqrt();
            // Prevent division by zero
            if *std < 1e-8 {
                *std = 1.0;
            }
        }

        // Yield for large datasets
        if dataset.features.len() > 10000 {
            tokio::task::yield_now().await;
        }

        Ok(stats)
    }

    /// Apply preprocessing transformations using pre-computed statistics
    pub async fn apply_with_stats(
        &self,
        dataset: &mut Dataset,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        for operation in &self.operations {
            self.apply_operation(dataset, operation, stats).await?;

            // Yield periodically for long pipelines
            tokio::task::yield_now().await;
        }

        Ok(())
    }

    /// Apply a single preprocessing operation
    async fn apply_operation(
        &self,
        dataset: &mut Dataset,
        operation: &PreprocessingOp,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        match operation {
            PreprocessingOp::Normalize { feature_indices } => {
                self.apply_normalization(dataset, feature_indices.as_deref(), stats)
                    .await?;
            }
            PreprocessingOp::Standardize { feature_indices } => {
                self.apply_standardization(dataset, feature_indices.as_deref(), stats)
                    .await?;
            }
            PreprocessingOp::OneHotEncode { feature_indices } => {
                self.apply_one_hot_encoding(dataset, feature_indices)
                    .await?;
            }
            PreprocessingOp::FillMissing { strategy } => {
                self.apply_missing_value_fill(dataset, strategy, stats)
                    .await?;
            }
            PreprocessingOp::RemoveOutliers { factor } => {
                self.apply_outlier_removal(dataset, *factor, stats).await?;
            }
            PreprocessingOp::PolynomialFeatures { degree } => {
                self.apply_polynomial_features(dataset, *degree).await?;
            }
            PreprocessingOp::PCA { components } => {
                self.apply_pca(dataset, *components, stats).await?;
            }
        }
        Ok(())
    }

    /// Apply normalization to specified features
    async fn apply_normalization(
        &self,
        dataset: &mut Dataset,
        feature_indices: Option<&[usize]>,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        let indices: Vec<usize> = if let Some(indices) = feature_indices {
            indices.to_vec()
        } else {
            (0..dataset.metadata.feature_count).collect()
        };

        for feature in &mut dataset.features {
            for &i in &indices {
                if i < feature.len() {
                    let min_val = stats.feature_mins[i];
                    let max_val = stats.feature_maxs[i];
                    let range = max_val - min_val;

                    if range > 1e-8 {
                        feature[i] = (feature[i] - min_val) / range;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply standardization to specified features
    async fn apply_standardization(
        &self,
        dataset: &mut Dataset,
        feature_indices: Option<&[usize]>,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        let indices: Vec<usize> = if let Some(indices) = feature_indices {
            indices.to_vec()
        } else {
            (0..dataset.metadata.feature_count).collect()
        };

        for feature in &mut dataset.features {
            for &i in &indices {
                if i < feature.len() {
                    let mean = stats.feature_means[i];
                    let std = stats.feature_stds[i];
                    feature[i] = (feature[i] - mean) / std;
                }
            }
        }

        Ok(())
    }

    async fn apply_one_hot_encoding(
        &self,
        dataset: &mut Dataset,
        feature_indices: &[usize],
    ) -> Result<(), DatasetError> {
        if feature_indices.is_empty() {
            return Ok(());
        }

        // First pass: discover unique categories for each feature
        let mut feature_categories: HashMap<usize, std::collections::BTreeSet<i32>> =
            HashMap::new();

        for feature in &dataset.features {
            for &feature_idx in feature_indices {
                if feature_idx < feature.len() {
                    let value = feature[feature_idx] as i32;
                    feature_categories
                        .entry(feature_idx)
                        .or_default()
                        .insert(value);
                }
            }
        }

        // Create mapping from categories to indices
        let mut category_mappings: HashMap<usize, HashMap<i32, usize>> = HashMap::new();

        for (&feature_idx, categories) in &feature_categories {
            let mut mapping = HashMap::new();
            for (idx, &category) in categories.iter().enumerate() {
                mapping.insert(category, idx);
            }
            category_mappings.insert(feature_idx, mapping);
        }

        // Second pass: create one-hot encoded features
        let mut new_features = Vec::new();
        let mut new_feature_names = Vec::new();

        for feature in &dataset.features {
            let mut expanded_feature = Vec::new();

            for (i, &value) in feature.iter().enumerate() {
                if feature_indices.contains(&i) {
                    if let Some(mapping) = category_mappings.get(&i) {
                        let category = value as i32;
                        let num_categories = mapping.len();

                        for j in 0..num_categories {
                            let is_category =
                                mapping.get(&category).map(|&idx| idx == j).unwrap_or(false);
                            expanded_feature.push(if is_category { 1.0 } else { 0.0 });
                        }
                    }
                } else {
                    expanded_feature.push(value);
                }
            }

            new_features.push(DVector::from_vec(expanded_feature));
        }

        // Update feature names
        if let Some(ref old_names) = dataset.feature_names {
            for (i, name) in old_names.iter().enumerate() {
                if feature_indices.contains(&i) {
                    if let Some(categories) = feature_categories.get(&i) {
                        for category in categories {
                            new_feature_names.push(format!("{}_cat_{}", name, category));
                        }
                    }
                } else {
                    new_feature_names.push(name.clone());
                }
            }
            dataset.feature_names = Some(new_feature_names);
        }

        dataset.features = new_features;
        dataset.metadata.feature_count = dataset.features.first().map(|f| f.len()).unwrap_or(0);

        Ok(())
    }

    /// Apply missing value filling
    async fn apply_missing_value_fill(
        &self,
        dataset: &mut Dataset,
        strategy: &FillStrategy,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        match strategy {
            FillStrategy::Constant(value) => {
                for feature in &mut dataset.features {
                    for element in feature.iter_mut() {
                        if element.is_nan() || element.is_infinite() {
                            *element = *value;
                        }
                    }
                }
            }
            FillStrategy::Mean => {
                for feature in &mut dataset.features {
                    for (i, element) in feature.iter_mut().enumerate() {
                        if (element.is_nan() || element.is_infinite())
                            && i < stats.feature_means.len()
                        {
                            *element = stats.feature_means[i];
                        }
                    }
                }
            }
            FillStrategy::Median => {
                // Calculate medians for each feature column
                let mut medians = Vec::new();
                for i in 0..stats.feature_means.len() {
                    let mut values: Vec<f64> = dataset
                        .features
                        .iter()
                        .map(|f| f[i])
                        .filter(|v| !v.is_nan() && !v.is_infinite())
                        .collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = if values.is_empty() {
                        0.0
                    } else if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                    medians.push(median);
                }

                // Apply medians
                for feature in &mut dataset.features {
                    for (i, element) in feature.iter_mut().enumerate() {
                        if (element.is_nan() || element.is_infinite()) && i < medians.len() {
                            *element = medians[i];
                        }
                    }
                }
            }
            FillStrategy::Mode => {
                // Calculate modes for each feature column
                let mut modes = Vec::new();
                for i in 0..stats.feature_means.len() {
                    let mut value_counts = std::collections::HashMap::new();
                    for f in &dataset.features {
                        let val = f[i];
                        if !val.is_nan() && !val.is_infinite() {
                            *value_counts.entry((val * 1000.0) as i64).or_insert(0) += 1;
                        }
                    }
                    let mode = if let Some((&mode_key, _)) =
                        value_counts.iter().max_by_key(|(_, &count)| count)
                    {
                        mode_key as f64 / 1000.0
                    } else {
                        0.0
                    };
                    modes.push(mode);
                }

                // Apply modes
                for feature in &mut dataset.features {
                    for (i, element) in feature.iter_mut().enumerate() {
                        if (element.is_nan() || element.is_infinite()) && i < modes.len() {
                            *element = modes[i];
                        }
                    }
                }
            }
            FillStrategy::ForwardFill => {
                // Fill missing values with the last valid observation
                for feature_idx in 0..stats.feature_means.len() {
                    let mut last_valid_value = None;
                    for sample in &mut dataset.features {
                        if feature_idx < sample.len() {
                            let value = &mut sample[feature_idx];
                            if value.is_nan() || value.is_infinite() {
                                if let Some(valid_val) = last_valid_value {
                                    *value = valid_val;
                                } else {
                                    // If no previous valid value, use feature mean as fallback
                                    *value = stats.feature_means[feature_idx];
                                }
                            } else {
                                last_valid_value = Some(*value);
                            }
                        }
                    }
                }
            }
            FillStrategy::BackwardFill => {
                // Fill missing values with the next valid observation
                for feature_idx in 0..stats.feature_means.len() {
                    let mut next_valid_value = None;
                    // Process samples in reverse order
                    for sample in dataset.features.iter_mut().rev() {
                        if feature_idx < sample.len() {
                            let value = &mut sample[feature_idx];
                            if value.is_nan() || value.is_infinite() {
                                if let Some(valid_val) = next_valid_value {
                                    *value = valid_val;
                                } else {
                                    // If no next valid value, use feature mean as fallback
                                    *value = stats.feature_means[feature_idx];
                                }
                            } else {
                                next_valid_value = Some(*value);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply outlier removal using IQR method
    async fn apply_outlier_removal(
        &self,
        dataset: &mut Dataset,
        factor: f64,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        let mut indices_to_keep = Vec::new();

        'sample_loop: for (sample_idx, feature) in dataset.features.iter().enumerate() {
            for (feature_idx, &value) in feature.iter().enumerate() {
                if feature_idx < stats.feature_means.len() {
                    let mean = stats.feature_means[feature_idx];
                    let std = stats.feature_stds[feature_idx];
                    let z_score = (value - mean).abs() / std;

                    // Remove samples with z-score > factor (default 3.0)
                    if z_score > factor {
                        continue 'sample_loop;
                    }
                }
            }
            indices_to_keep.push(sample_idx);
        }

        // Filter features and targets
        let new_features: Vec<_> = indices_to_keep
            .iter()
            .map(|&idx| dataset.features[idx].clone())
            .collect();

        let new_targets = dataset.targets.as_ref().map(|targets| {
            indices_to_keep
                .iter()
                .map(|&idx| targets[idx].clone())
                .collect()
        });

        dataset.features = new_features;
        dataset.targets = new_targets;
        dataset.metadata.sample_count = dataset.features.len();

        Ok(())
    }

    /// Apply polynomial feature expansion
    async fn apply_polynomial_features(
        &self,
        dataset: &mut Dataset,
        degree: usize,
    ) -> Result<(), DatasetError> {
        if degree <= 1 {
            return Ok(()); // No expansion needed
        }

        let mut new_features = Vec::new();
        let mut new_feature_names = Vec::new();

        for feature in &dataset.features {
            let mut expanded = feature.clone().data.as_vec().clone();
            let original_len = expanded.len();

            // Add polynomial terms up to the specified degree
            for d in 2..=degree {
                for i in 0..original_len {
                    expanded.push(feature[i].powi(d as i32));
                }
            }

            new_features.push(DVector::from_vec(expanded));
        }

        // Update feature names
        if let Some(ref old_names) = dataset.feature_names {
            new_feature_names.extend(old_names.clone());
            for d in 2..=degree {
                for name in old_names {
                    new_feature_names.push(format!("{}_pow_{}", name, d));
                }
            }
            dataset.feature_names = Some(new_feature_names);
        }

        dataset.features = new_features;
        dataset.metadata.feature_count = dataset.features.first().map(|f| f.len()).unwrap_or(0);

        Ok(())
    }

    async fn apply_pca(
        &self,
        dataset: &mut Dataset,
        components: usize,
        stats: &PreprocessingStats,
    ) -> Result<(), DatasetError> {
        if dataset.features.is_empty() || components == 0 {
            return Ok(());
        }

        let n_samples = dataset.features.len();
        let n_features = dataset.features[0].len();
        let components = components.min(n_features.min(n_samples));

        // Center the data using precomputed means
        let mut centered_data = Vec::new();
        for feature in &dataset.features {
            let mut centered = feature.clone();
            for (i, val) in centered.iter_mut().enumerate() {
                if i < stats.feature_means.len() {
                    *val -= stats.feature_means[i];
                }
            }
            centered_data.push(centered);
        }

        // Compute covariance matrix
        let mut covariance = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = 0.0;
                for sample in &centered_data {
                    sum += sample[i] * sample[j];
                }
                let cov_val = sum / (n_samples - 1) as f64;
                covariance[i][j] = cov_val;
                covariance[j][i] = cov_val;
            }
        }

        // Power iteration to find principal components
        let mut components_vectors = Vec::new();
        let mut remaining_cov = covariance.clone();

        for _ in 0..components {
            // Find dominant eigenvector using power iteration
            let mut eigenvector = vec![1.0 / (n_features as f64).sqrt(); n_features];

            for _ in 0..100 {
                // Max iterations
                let mut new_vector = vec![0.0; n_features];

                // Matrix-vector multiplication
                for i in 0..n_features {
                    for (j, value) in eigenvector.iter().enumerate() {
                        new_vector[i] += remaining_cov[i][j] * value;
                    }
                }

                // Normalize
                let norm = new_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-12 {
                    break;
                }

                for val in &mut new_vector {
                    *val /= norm;
                }

                // Check convergence
                let mut converged = true;
                for (old, new) in eigenvector.iter().zip(new_vector.iter()) {
                    if (old - new).abs() > 1e-8 {
                        converged = false;
                        break;
                    }
                }

                eigenvector = new_vector;
                if converged {
                    break;
                }
            }

            components_vectors.push(DVector::from_vec(eigenvector.clone()));

            // Deflate the matrix (remove this component's contribution)
            let eigenvalue = {
                let mut sum = 0.0;
                for i in 0..n_features {
                    for j in 0..n_features {
                        sum += eigenvector[i] * remaining_cov[i][j] * eigenvector[j];
                    }
                }
                sum
            };

            for i in 0..n_features {
                for j in 0..n_features {
                    remaining_cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
                }
            }
        }

        // Transform the data
        let mut transformed_features = Vec::new();
        for sample in &centered_data {
            let mut transformed = Vec::new();
            for component in &components_vectors {
                let mut projection = 0.0;
                for (i, &val) in sample.iter().enumerate() {
                    projection += val * component[i];
                }
                transformed.push(projection);
            }
            transformed_features.push(DVector::from_vec(transformed));
        }

        dataset.features = transformed_features;
        dataset.metadata.feature_count = components;

        // Update feature names
        if dataset.feature_names.is_some() {
            let mut new_names = Vec::new();
            for i in 0..components {
                new_names.push(format!("PC_{}", i + 1));
            }
            dataset.feature_names = Some(new_names);
        }

        Ok(())
    }
}

/// Utility functions for common preprocessing tasks
pub struct PreprocessingUtils;

impl PreprocessingUtils {
    /// Shuffle a dataset in place
    pub async fn shuffle(dataset: &mut Dataset) -> Result<(), DatasetError> {
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..dataset.features.len()).collect();
        let mut shuffled_indices = indices;
        shuffled_indices.shuffle(&mut rng);

        // Reorder features
        let mut new_features = Vec::with_capacity(dataset.features.len());
        for &i in &shuffled_indices {
            new_features.push(dataset.features[i].clone());
        }
        dataset.features = new_features;

        // Reorder targets if present
        if let Some(ref mut targets) = dataset.targets {
            let mut new_targets = Vec::with_capacity(targets.len());
            for &i in &shuffled_indices {
                new_targets.push(targets[i].clone());
            }
            *targets = new_targets;
        }

        Ok(())
    }

    /// Split dataset into train/validation/test sets
    pub async fn train_val_test_split(
        dataset: &Dataset,
        train_ratio: f64,
        val_ratio: f64,
    ) -> Result<(Dataset, Dataset, Dataset), DatasetError> {
        let n_samples = dataset.features.len();
        let n_train = (n_samples as f64 * train_ratio) as usize;
        let n_val = (n_samples as f64 * val_ratio) as usize;

        if train_ratio + val_ratio > 1.0 {
            return Err(DatasetError::Configuration(
                "Train and validation ratios sum to more than 1.0".to_string(),
            ));
        }

        let n_test = n_samples - n_train - n_val;

        let train_indices: Vec<usize> = (0..n_train).collect();
        let val_indices: Vec<usize> = (n_train..n_train + n_val).collect();
        let test_indices: Vec<usize> = (n_train + n_val..n_train + n_val + n_test).collect();

        let train_set = dataset.subset(&train_indices);
        let val_set = dataset.subset(&val_indices);
        let test_set = dataset.subset(&test_indices);

        Ok((train_set, val_set, test_set))
    }

    /// Calculate correlation matrix between features
    pub async fn correlation_matrix(dataset: &Dataset) -> Result<Vec<Vec<f64>>, DatasetError> {
        let n_features = dataset.metadata.feature_count;
        let mut correlation_matrix = vec![vec![0.0; n_features]; n_features];

        // Calculate pairwise correlations
        for i in 0..n_features {
            for j in i..n_features {
                let correlation = Self::calculate_correlation(dataset, i, j)?;
                correlation_matrix[i][j] = correlation;
                correlation_matrix[j][i] = correlation;
            }
        }

        Ok(correlation_matrix)
    }

    /// Calculate Pearson correlation between two features
    fn calculate_correlation(
        dataset: &Dataset,
        feature_i: usize,
        feature_j: usize,
    ) -> Result<f64, DatasetError> {
        if dataset.features.is_empty() {
            return Ok(0.0);
        }

        let n = dataset.features.len() as f64;

        // Extract feature values
        let values_i: Vec<f64> = dataset
            .features
            .iter()
            .filter_map(|f| f.get(feature_i).copied())
            .collect();
        let values_j: Vec<f64> = dataset
            .features
            .iter()
            .filter_map(|f| f.get(feature_j).copied())
            .collect();

        if values_i.len() != values_j.len() || values_i.is_empty() {
            return Ok(0.0);
        }

        // Calculate means
        let mean_i = values_i.iter().sum::<f64>() / n;
        let mean_j = values_j.iter().sum::<f64>() / n;

        // Calculate correlation
        let mut numerator = 0.0;
        let mut sum_sq_i = 0.0;
        let mut sum_sq_j = 0.0;

        for (vi, vj) in values_i.iter().zip(values_j.iter()) {
            let diff_i = vi - mean_i;
            let diff_j = vj - mean_j;
            numerator += diff_i * diff_j;
            sum_sq_i += diff_i * diff_i;
            sum_sq_j += diff_j * diff_j;
        }

        let denominator = (sum_sq_i * sum_sq_j).sqrt();
        if denominator < 1e-8 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}
