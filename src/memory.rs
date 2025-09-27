use crate::error::{HextralError, HextralResult};
use nalgebra::{DMatrix, DVector};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global memory tracker to monitor allocations
pub struct MemoryTracker {
    allocated: AtomicUsize,
    peak: AtomicUsize,
    allocations: AtomicUsize,
}

impl MemoryTracker {
    pub const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
        }
    }

    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
    }

    fn record_allocation(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let current = self.allocated.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak if necessary
        let mut peak = self.peak.load(Ordering::Relaxed);
        while current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    fn record_deallocation(&self, size: usize) {
        self.allocated.fetch_sub(size, Ordering::Relaxed);
    }
}

static MEMORY_TRACKER: MemoryTracker = MemoryTracker::new();

/// Custom allocator wrapper for tracking memory usage
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            MEMORY_TRACKER.record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        MEMORY_TRACKER.record_deallocation(layout.size());
        System.dealloc(ptr, layout);
    }
}

/// Memory management configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub enable_tracking: bool,
    pub max_pool_size: usize,
    pub enable_gc_hints: bool,
    pub memory_limit_mb: Option<usize>,
    pub cleanup_threshold_mb: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            max_pool_size: 1000,
            enable_gc_hints: true,
            memory_limit_mb: None,
            cleanup_threshold_mb: 100,
        }
    }
}

/// Advanced memory manager with pooling and optimization
#[derive(Clone)]
pub struct MemoryManager {
    config: MemoryConfig,
    vector_pools: HashMap<usize, Vec<DVector<f64>>>,
    matrix_pools: HashMap<(usize, usize), Vec<DMatrix<f64>>>,
    allocation_history: Vec<AllocationEvent>,
    cleanup_threshold: usize,
}

#[derive(Debug, Clone)]
pub struct AllocationEvent {
    pub timestamp: std::time::Instant,
    pub size: usize,
    pub allocation_type: AllocationType,
}

#[derive(Debug, Clone)]
pub enum AllocationType {
    Vector(usize),
    Matrix(usize, usize),
    Gradient,
    Activation,
    Temporary,
}

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Self {
        let cleanup_threshold = config.cleanup_threshold_mb * 1024 * 1024;
        Self {
            config,
            vector_pools: HashMap::new(),
            matrix_pools: HashMap::new(),
            allocation_history: Vec::new(),
            cleanup_threshold,
        }
    }

    /// Get a vector from pool or create new one
    pub fn get_vector(&mut self, size: usize) -> DVector<f64> {
        if let Some(pool) = self.vector_pools.get_mut(&size) {
            if let Some(mut vec) = pool.pop() {
                vec.fill(0.0);
                self.record_allocation(AllocationType::Vector(size));
                return vec;
            }
        }

        self.record_allocation(AllocationType::Vector(size));
        DVector::zeros(size)
    }

    /// Return vector to pool
    pub fn return_vector(&mut self, vec: DVector<f64>) {
        let size = vec.len();
        let pool = self.vector_pools.entry(size).or_insert_with(Vec::new);

        if pool.len() < self.config.max_pool_size / 10 {
            pool.push(vec);
        }

        self.check_cleanup_threshold();
    }

    /// Get a matrix from pool or create new one
    pub fn get_matrix(&mut self, rows: usize, cols: usize) -> DMatrix<f64> {
        let key = (rows, cols);
        if let Some(pool) = self.matrix_pools.get_mut(&key) {
            if let Some(mut mat) = pool.pop() {
                mat.fill(0.0);
                self.record_allocation(AllocationType::Matrix(rows, cols));
                return mat;
            }
        }

        self.record_allocation(AllocationType::Matrix(rows, cols));
        DMatrix::zeros(rows, cols)
    }

    /// Return matrix to pool
    pub fn return_matrix(&mut self, mat: DMatrix<f64>) {
        let key = (mat.nrows(), mat.ncols());
        let pool = self.matrix_pools.entry(key).or_insert_with(Vec::new);

        if pool.len() < self.config.max_pool_size / 10 {
            pool.push(mat);
        }

        self.check_cleanup_threshold();
    }

    /// Create optimized gradient storage
    pub fn create_gradient_storage(&mut self, layer_shapes: &[(usize, usize)]) -> GradientStorage {
        let mut weight_gradients = Vec::with_capacity(layer_shapes.len());
        let mut bias_gradients = Vec::with_capacity(layer_shapes.len());

        for &(rows, cols) in layer_shapes {
            weight_gradients.push(self.get_matrix(rows, cols));
            bias_gradients.push(self.get_vector(rows));
        }

        GradientStorage {
            weight_gradients,
            bias_gradients,
            accumulated: false,
        }
    }

    /// Create activation storage for forward pass
    pub fn create_activation_storage(&mut self, layer_sizes: &[usize]) -> ActivationStorage {
        let mut activations = Vec::with_capacity(layer_sizes.len());

        for &size in layer_sizes {
            activations.push(self.get_vector(size));
        }

        self.record_allocation(AllocationType::Activation);

        ActivationStorage {
            activations,
            current_layer: 0,
        }
    }

    /// Perform in-place matrix operations to reduce memory allocation
    pub fn matrix_multiply_inplace(
        &self,
        result: &mut DMatrix<f64>,
        left: &DMatrix<f64>,
        right: &DMatrix<f64>,
    ) -> HextralResult<()> {
        if left.ncols() != right.nrows() {
            return Err(HextralError::InvalidInput {
                context: "Matrix multiplication".to_string(),
                details: format!(
                    "Incompatible dimensions: {}x{} * {}x{}",
                    left.nrows(),
                    left.ncols(),
                    right.nrows(),
                    right.ncols()
                ),
                recoverable: true,
            });
        }

        if result.nrows() != left.nrows() || result.ncols() != right.ncols() {
            return Err(HextralError::InvalidInput {
                context: "Matrix multiplication".to_string(),
                details: "Result matrix has incorrect dimensions".to_string(),
                recoverable: true,
            });
        }

        result.gemm(1.0, left, right, 0.0);
        Ok(())
    }

    /// Perform in-place vector operations
    pub fn vector_add_inplace(
        &self,
        target: &mut DVector<f64>,
        source: &DVector<f64>,
    ) -> HextralResult<()> {
        if target.len() != source.len() {
            return Err(HextralError::InvalidInput {
                context: "Vector addition".to_string(),
                details: "Vector dimensions don't match".to_string(),
                recoverable: true,
            });
        }

        *target += source;
        Ok(())
    }

    /// Get current memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let vector_pool_memory: usize = self
            .vector_pools
            .iter()
            .map(|(&size, pool)| size * pool.len() * std::mem::size_of::<f64>())
            .sum();

        let matrix_pool_memory: usize = self
            .matrix_pools
            .iter()
            .map(|(&(rows, cols), pool)| rows * cols * pool.len() * std::mem::size_of::<f64>())
            .sum();

        MemoryStats {
            vector_pools: self.vector_pools.len(),
            matrix_pools: self.matrix_pools.len(),
            pool_memory_bytes: vector_pool_memory + matrix_pool_memory,
            total_allocated: MEMORY_TRACKER.allocated(),
            peak_allocated: MEMORY_TRACKER.peak(),
            allocation_count: MEMORY_TRACKER.allocation_count(),
            allocation_history_size: self.allocation_history.len(),
        }
    }

    /// Force cleanup of memory pools
    pub fn force_cleanup(&mut self) {
        // Keep only the most recently used pools
        for pool in self.vector_pools.values_mut() {
            pool.truncate(self.config.max_pool_size / 20);
        }

        for pool in self.matrix_pools.values_mut() {
            pool.truncate(self.config.max_pool_size / 20);
        }

        // Clear old allocation history
        if self.allocation_history.len() > 1000 {
            self.allocation_history
                .drain(..self.allocation_history.len() / 2);
        }

        if self.config.enable_gc_hints {
            // Hint to the garbage collector (on platforms that support it)
            std::hint::black_box(&self.vector_pools);
        }
    }

    /// Clear all pools and reset tracking
    pub fn reset(&mut self) {
        self.vector_pools.clear();
        self.matrix_pools.clear();
        self.allocation_history.clear();
        MEMORY_TRACKER.reset();
    }

    /// Check if we should perform cleanup
    fn check_cleanup_threshold(&mut self) {
        if MEMORY_TRACKER.allocated() > self.cleanup_threshold {
            self.force_cleanup();
        }
    }

    /// Record allocation for tracking
    fn record_allocation(&mut self, allocation_type: AllocationType) {
        if self.config.enable_tracking {
            let size = match &allocation_type {
                AllocationType::Vector(s) => s * std::mem::size_of::<f64>(),
                AllocationType::Matrix(r, c) => r * c * std::mem::size_of::<f64>(),
                _ => 0,
            };

            self.allocation_history.push(AllocationEvent {
                timestamp: std::time::Instant::now(),
                size,
                allocation_type,
            });

            // Limit history size
            if self.allocation_history.len() > 10000 {
                self.allocation_history.drain(..1000);
            }
        }
    }

    /// Analyze allocation patterns for optimization recommendations
    pub fn analyze_patterns(&self) -> AllocationAnalysis {
        let mut vector_sizes = HashMap::new();
        let mut matrix_sizes = HashMap::new();

        for event in &self.allocation_history {
            match &event.allocation_type {
                AllocationType::Vector(size) => {
                    *vector_sizes.entry(*size).or_insert(0) += 1;
                }
                AllocationType::Matrix(rows, cols) => {
                    *matrix_sizes.entry((*rows, *cols)).or_insert(0) += 1;
                }
                _ => {}
            }
        }

        AllocationAnalysis {
            most_common_vector_size: vector_sizes
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(size, _)| *size),
            most_common_matrix_size: matrix_sizes
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(size, _)| *size),
            total_allocations: self.allocation_history.len(),
            avg_allocation_rate: self.calculate_allocation_rate(),
        }
    }

    fn calculate_allocation_rate(&self) -> f64 {
        if self.allocation_history.len() < 2 {
            return 0.0;
        }

        let first = self.allocation_history.first().unwrap();
        let last = self.allocation_history.last().unwrap();
        let duration = last.timestamp.duration_since(first.timestamp).as_secs_f64();

        if duration > 0.0 {
            self.allocation_history.len() as f64 / duration
        } else {
            0.0
        }
    }
}

/// Gradient storage with memory management
pub struct GradientStorage {
    pub weight_gradients: Vec<DMatrix<f64>>,
    pub bias_gradients: Vec<DVector<f64>>,
    pub accumulated: bool,
}

impl GradientStorage {
    pub fn clear(&mut self) {
        for grad in &mut self.weight_gradients {
            grad.fill(0.0);
        }
        for grad in &mut self.bias_gradients {
            grad.fill(0.0);
        }
        self.accumulated = false;
    }

    pub fn accumulate(&mut self, other: &GradientStorage) -> HextralResult<()> {
        if self.weight_gradients.len() != other.weight_gradients.len() {
            return Err(HextralError::InvalidInput {
                context: "Gradient accumulation".to_string(),
                details: "Mismatched gradient storage sizes".to_string(),
                recoverable: true,
            });
        }

        for (self_grad, other_grad) in self
            .weight_gradients
            .iter_mut()
            .zip(&other.weight_gradients)
        {
            *self_grad += other_grad;
        }

        for (self_grad, other_grad) in self.bias_gradients.iter_mut().zip(&other.bias_gradients) {
            *self_grad += other_grad;
        }

        self.accumulated = true;
        Ok(())
    }

    pub fn scale(&mut self, factor: f64) {
        for grad in &mut self.weight_gradients {
            *grad *= factor;
        }
        for grad in &mut self.bias_gradients {
            *grad *= factor;
        }
    }
}

/// Activation storage for forward pass
pub struct ActivationStorage {
    pub activations: Vec<DVector<f64>>,
    pub current_layer: usize,
}

impl ActivationStorage {
    pub fn get_current(&self) -> &DVector<f64> {
        &self.activations[self.current_layer]
    }

    pub fn get_current_mut(&mut self) -> &mut DVector<f64> {
        &mut self.activations[self.current_layer]
    }

    pub fn advance(&mut self) {
        if self.current_layer < self.activations.len() - 1 {
            self.current_layer += 1;
        }
    }

    pub fn reset(&mut self) {
        self.current_layer = 0;
        for activation in &mut self.activations {
            activation.fill(0.0);
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub vector_pools: usize,
    pub matrix_pools: usize,
    pub pool_memory_bytes: usize,
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub allocation_count: usize,
    pub allocation_history_size: usize,
}

impl MemoryStats {
    pub fn pool_memory_mb(&self) -> f64 {
        self.pool_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn total_allocated_mb(&self) -> f64 {
        self.total_allocated as f64 / (1024.0 * 1024.0)
    }

    pub fn peak_allocated_mb(&self) -> f64 {
        self.peak_allocated as f64 / (1024.0 * 1024.0)
    }
}

#[derive(Debug, Clone)]
pub struct AllocationAnalysis {
    pub most_common_vector_size: Option<usize>,
    pub most_common_matrix_size: Option<(usize, usize)>,
    pub total_allocations: usize,
    pub avg_allocation_rate: f64,
}

/// Global functions for memory tracking
pub fn get_memory_stats() -> MemoryStats {
    MemoryStats {
        vector_pools: 0,
        matrix_pools: 0,
        pool_memory_bytes: 0,
        total_allocated: MEMORY_TRACKER.allocated(),
        peak_allocated: MEMORY_TRACKER.peak(),
        allocation_count: MEMORY_TRACKER.allocation_count(),
        allocation_history_size: 0,
    }
}

pub fn reset_memory_tracking() {
    MEMORY_TRACKER.reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager() {
        let config = MemoryConfig::default();
        let mut manager = MemoryManager::new(config);

        let vec1 = manager.get_vector(10);
        assert_eq!(vec1.len(), 10);

        let mat1 = manager.get_matrix(3, 4);
        assert_eq!((mat1.nrows(), mat1.ncols()), (3, 4));

        manager.return_vector(vec1);
        manager.return_matrix(mat1);

        let stats = manager.memory_stats();
        assert_eq!(stats.vector_pools, 1);
        assert_eq!(stats.matrix_pools, 1);
    }

    #[test]
    fn test_gradient_storage() {
        let layer_shapes = vec![(3, 2), (2, 1)];
        let config = MemoryConfig::default();
        let mut manager = MemoryManager::new(config);

        let mut storage = manager.create_gradient_storage(&layer_shapes);
        assert_eq!(storage.weight_gradients.len(), 2);
        assert_eq!(storage.bias_gradients.len(), 2);

        storage.clear();
        assert!(!storage.accumulated);

        // Test accumulation
        let storage2 = manager.create_gradient_storage(&layer_shapes);
        storage.accumulate(&storage2).unwrap();
        assert!(storage.accumulated);
    }

    #[test]
    fn test_in_place_operations() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        // Create test matrices for multiplication: A * B = C
        // A = [1 3]    B = [2 1]    C = [1*2+3*1  1*0+3*2] = [5 6]
        //     [2 4]        [1 2]          [2*2+4*1  2*0+4*2]   [8 8]
        let left = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let right = DMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let mut result = DMatrix::zeros(2, 2);

        manager
            .matrix_multiply_inplace(&mut result, &left, &right)
            .unwrap();

        // Verify result: [1*2+3*1, 1*1+3*2; 2*2+4*1, 2*1+4*2] = [5, 7; 8, 10]
        assert!((result[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((result[(0, 1)] - 7.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 8.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 10.0).abs() < 1e-10);
    }
}
