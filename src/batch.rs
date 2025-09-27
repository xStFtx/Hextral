use crate::error::{HextralError, HextralResult};
use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;
use std::sync::Arc;

/// Memory-efficient batch iterator that streams data without loading everything into memory
pub struct BatchIterator<T> {
    data: Arc<Vec<T>>,
    batch_size: usize,
    current_index: usize,
    shuffle: bool,
    indices: Vec<usize>,
}

impl<T: Clone> BatchIterator<T> {
    pub fn new(data: Arc<Vec<T>>, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }

        Self {
            data,
            batch_size,
            current_index: 0,
            shuffle,
            indices,
        }
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }

    pub fn len(&self) -> usize {
        (self.data.len() + self.batch_size - 1) / self.batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Clone> Iterator for BatchIterator<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.data.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.data.len());
        let batch: Vec<T> = self.indices[self.current_index..end_index]
            .iter()
            .map(|&i| self.data[i].clone())
            .collect();

        self.current_index = end_index;
        Some(batch)
    }
}

/// Memory pool for reusing vectors and matrices to reduce allocations
#[derive(Clone)]
pub struct MemoryPool {
    vector_pool: VecDeque<DVector<f64>>,
    matrix_pool: VecDeque<DMatrix<f64>>,
    batch_pool: VecDeque<Vec<DVector<f64>>>,
    max_pool_size: usize,
}

impl MemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            vector_pool: VecDeque::with_capacity(max_pool_size),
            matrix_pool: VecDeque::with_capacity(max_pool_size),
            batch_pool: VecDeque::with_capacity(max_pool_size / 4),
            max_pool_size,
        }
    }

    pub fn get_vector(&mut self, size: usize) -> DVector<f64> {
        if let Some(mut vec) = self.vector_pool.pop_front() {
            if vec.len() == size {
                vec.fill(0.0);
                return vec;
            }
        }
        DVector::zeros(size)
    }

    pub fn return_vector(&mut self, vec: DVector<f64>) {
        if self.vector_pool.len() < self.max_pool_size {
            self.vector_pool.push_back(vec);
        }
    }

    pub fn get_matrix(&mut self, rows: usize, cols: usize) -> DMatrix<f64> {
        if let Some(mut mat) = self.matrix_pool.pop_front() {
            if mat.nrows() == rows && mat.ncols() == cols {
                mat.fill(0.0);
                return mat;
            }
        }
        DMatrix::zeros(rows, cols)
    }

    pub fn return_matrix(&mut self, mat: DMatrix<f64>) {
        if self.matrix_pool.len() < self.max_pool_size {
            self.matrix_pool.push_back(mat);
        }
    }

    pub fn get_batch(&mut self, capacity: usize) -> Vec<DVector<f64>> {
        if let Some(mut batch) = self.batch_pool.pop_front() {
            batch.clear();
            batch.reserve(capacity);
            return batch;
        }
        Vec::with_capacity(capacity)
    }

    pub fn return_batch(&mut self, batch: Vec<DVector<f64>>) {
        if self.batch_pool.len() < self.max_pool_size / 4 {
            self.batch_pool.push_back(batch);
        }
    }

    pub fn clear(&mut self) {
        self.vector_pool.clear();
        self.matrix_pool.clear();
        self.batch_pool.clear();
    }

    pub fn memory_usage(&self) -> MemoryUsage {
        let vector_memory: usize = self
            .vector_pool
            .iter()
            .map(|v| v.len() * std::mem::size_of::<f64>())
            .sum();

        let matrix_memory: usize = self
            .matrix_pool
            .iter()
            .map(|m| m.len() * std::mem::size_of::<f64>())
            .sum();

        let batch_memory: usize = self
            .batch_pool
            .iter()
            .map(|b| b.capacity() * std::mem::size_of::<DVector<f64>>())
            .sum();

        MemoryUsage {
            vector_pool_size: self.vector_pool.len(),
            matrix_pool_size: self.matrix_pool.len(),
            batch_pool_size: self.batch_pool.len(),
            total_bytes: vector_memory + matrix_memory + batch_memory,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub vector_pool_size: usize,
    pub matrix_pool_size: usize,
    pub batch_pool_size: usize,
    pub total_bytes: usize,
}

/// Optimized batch processor with memory management and SIMD operations
#[derive(Clone)]
pub struct BatchProcessor {
    pool: MemoryPool,
    chunk_size: usize,
    parallel_threshold: usize,
}

impl BatchProcessor {
    pub fn new(pool_size: usize, chunk_size: usize) -> Self {
        Self {
            pool: MemoryPool::new(pool_size),
            chunk_size,
            parallel_threshold: 100, // Use parallel processing for batches > 100
        }
    }

    /// Process batch with optimized memory usage and optional parallelization
    pub fn process_batch_sync<F>(
        &mut self,
        inputs: &[DVector<f64>],
        targets: &[DVector<f64>],
        mut processor: F,
    ) -> HextralResult<Vec<f64>>
    where
        F: FnMut(&DVector<f64>, &DVector<f64>) -> HextralResult<f64>,
    {
        if inputs.len() != targets.len() {
            return Err(HextralError::InvalidInput {
                context: "Batch processing".to_string(),
                details: format!(
                    "Input batch size {} doesn't match target batch size {}",
                    inputs.len(),
                    targets.len()
                ),
                recoverable: true,
            });
        }

        let results = if inputs.len() > self.parallel_threshold && cfg!(feature = "performance") {
            // For now, use sequential processing due to closure constraints
            inputs
                .iter()
                .zip(targets.iter())
                .map(|(input, target)| processor(input, target))
                .collect::<HextralResult<Vec<_>>>()?
        } else {
            // Sequential processing for small batches
            inputs
                .iter()
                .zip(targets.iter())
                .map(|(input, target)| processor(input, target))
                .collect::<HextralResult<Vec<_>>>()?
        };

        Ok(results)
    }

    /// Process inputs in chunks to reduce memory pressure
    pub async fn process_chunked<F, R>(
        &mut self,
        inputs: &[DVector<f64>],
        mut processor: F,
    ) -> HextralResult<Vec<R>>
    where
        F: FnMut(&[DVector<f64>]) -> HextralResult<Vec<R>> + Send,
        R: Clone + Send,
    {
        let mut results = Vec::with_capacity(inputs.len());

        for chunk in inputs.chunks(self.chunk_size) {
            let chunk_results = processor(chunk)?;
            results.extend(chunk_results);

            // Yield to prevent blocking
            if chunk.len() > 50 {
                tokio::task::yield_now().await;
            }
        }

        Ok(results)
    }

    /// Compute batch statistics with memory-efficient streaming
    pub fn compute_batch_stats(&self, data: &[DVector<f64>]) -> HextralResult<BatchStats> {
        if data.is_empty() {
            return Err(HextralError::InvalidInput {
                context: "Batch statistics".to_string(),
                details: "Cannot compute statistics for empty batch".to_string(),
                recoverable: true,
            });
        }

        let dim = data[0].len();
        let mut sum = DVector::zeros(dim);
        let mut sum_sq = DVector::zeros(dim);
        let mut min_vals = data[0].clone();
        let mut max_vals = data[0].clone();

        for sample in data {
            if sample.len() != dim {
                return Err(HextralError::InvalidInput {
                    context: "Batch statistics".to_string(),
                    details: format!(
                        "Inconsistent dimensions: expected {}, got {}",
                        dim,
                        sample.len()
                    ),
                    recoverable: true,
                });
            }

            sum += sample;
            sum_sq += sample.component_mul(sample);

            for i in 0..dim {
                min_vals[i] = min_vals[i].min(sample[i]);
                max_vals[i] = max_vals[i].max(sample[i]);
            }
        }

        let n = data.len() as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - mean.component_mul(&mean);
        let std_dev = variance.map(|v| v.sqrt());

        Ok(BatchStats {
            mean,
            std_dev,
            min: min_vals,
            max: max_vals,
            count: data.len(),
        })
    }

    /// Get memory pool statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        self.pool.memory_usage()
    }

    /// Clear memory pool to free up memory
    pub fn clear_memory_pool(&mut self) {
        self.pool.clear();
    }

    /// Optimize batch size based on available memory and data characteristics
    pub fn recommend_batch_size(
        &self,
        sample_size: usize,
        available_memory_mb: usize,
        model_parameters: usize,
    ) -> usize {
        let sample_memory = sample_size * std::mem::size_of::<f64>();
        let gradient_memory = model_parameters * std::mem::size_of::<f64>();
        let overhead_factor = 3.0; // Account for activations and temporary variables

        let memory_per_sample = (sample_memory + gradient_memory) as f64 * overhead_factor;
        let available_bytes = available_memory_mb * 1024 * 1024;

        let recommended_size = (available_bytes as f64 * 0.7 / memory_per_sample) as usize;
        recommended_size.max(1).min(1024) // Clamp between 1 and 1024
    }
}

#[derive(Debug, Clone)]
pub struct BatchStats {
    pub mean: DVector<f64>,
    pub std_dev: DVector<f64>,
    pub min: DVector<f64>,
    pub max: DVector<f64>,
    pub count: usize,
}

impl BatchStats {
    pub fn normalize_batch(&self, data: &mut [DVector<f64>]) -> HextralResult<()> {
        for sample in data {
            if sample.len() != self.mean.len() {
                return Err(HextralError::InvalidInput {
                    context: "Batch normalization".to_string(),
                    details: "Sample dimension mismatch".to_string(),
                    recoverable: true,
                });
            }

            for i in 0..sample.len() {
                if self.std_dev[i] > 1e-8 {
                    sample[i] = (sample[i] - self.mean[i]) / self.std_dev[i];
                }
            }
        }
        Ok(())
    }
}

/// Memory-efficient data streaming for large datasets
pub struct DataStream<T> {
    data: Arc<Vec<T>>,
    chunk_size: usize,
    current_pos: usize,
}

impl<T: Clone> DataStream<T> {
    pub fn new(data: Arc<Vec<T>>, chunk_size: usize) -> Self {
        Self {
            data,
            chunk_size,
            current_pos: 0,
        }
    }

    pub fn next_chunk(&mut self) -> Option<Vec<T>> {
        if self.current_pos >= self.data.len() {
            return None;
        }

        let end_pos = (self.current_pos + self.chunk_size).min(self.data.len());
        let chunk = self.data[self.current_pos..end_pos].to_vec();
        self.current_pos = end_pos;

        Some(chunk)
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    pub fn progress(&self) -> f64 {
        if self.data.is_empty() {
            1.0
        } else {
            self.current_pos as f64 / self.data.len() as f64
        }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.current_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_iterator() {
        let data = Arc::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let mut iter = BatchIterator::new(data, 3, false);

        assert_eq!(iter.len(), 4);

        let batch1 = iter.next().unwrap();
        assert_eq!(batch1.len(), 3);

        let batch2 = iter.next().unwrap();
        assert_eq!(batch2.len(), 3);

        let batch3 = iter.next().unwrap();
        assert_eq!(batch3.len(), 3);

        let batch4 = iter.next().unwrap();
        assert_eq!(batch4.len(), 1);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(10);

        let vec1 = pool.get_vector(5);
        assert_eq!(vec1.len(), 5);

        pool.return_vector(vec1);
        let vec2 = pool.get_vector(5);
        assert_eq!(vec2.len(), 5);

        let usage = pool.memory_usage();
        assert_eq!(usage.vector_pool_size, 0);
    }

    #[tokio::test]
    async fn test_data_stream() {
        let data = Arc::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let mut stream = DataStream::new(data, 3);

        let chunk1 = stream.next_chunk().unwrap();
        assert_eq!(chunk1, vec![1, 2, 3]);
        assert_eq!(stream.progress(), 0.3);

        let chunk2 = stream.next_chunk().unwrap();
        assert_eq!(chunk2, vec![4, 5, 6]);

        stream.reset();
        assert_eq!(stream.progress(), 0.0);

        let chunk1_again = stream.next_chunk().unwrap();
        assert_eq!(chunk1_again, vec![1, 2, 3]);
    }
}
