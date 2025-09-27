use crate::{HextralError, HextralResult, TrainingContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "monitoring")]
use metrics::{counter, gauge, histogram};
#[cfg(feature = "monitoring")]
use tracing::{debug, error, info, span, warn, Level};

/// Training progress callback trait for custom monitoring
pub trait ProgressCallback: Send + Sync {
    /// Called at the start of training
    fn on_training_start(&mut self, context: &TrainingContext);

    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, context: &TrainingContext, metrics: &EpochMetrics);

    /// Called when training completes (successfully or with errors)
    fn on_training_end(&mut self, context: &TrainingContext, result: &TrainingResult);

    /// Called when an error occurs during training
    fn on_error(&mut self, context: &TrainingContext, error: &HextralError);
}

/// Comprehensive metrics collected during an epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
    pub learning_rate: f64,
    pub batch_time_ms: Vec<f64>,
    pub forward_pass_time_ms: f64,
    pub backward_pass_time_ms: f64,
    pub optimizer_time_ms: f64,
    pub memory_usage_mb: Option<f64>,
    pub throughput_samples_per_sec: f64,
    pub gradient_norm: Option<f64>,
    pub weight_norm: Option<f64>,
    pub timestamp: SystemTime,
}

impl EpochMetrics {
    pub fn new(epoch: usize, train_loss: f64, learning_rate: f64) -> Self {
        Self {
            epoch,
            train_loss,
            validation_loss: None,
            learning_rate,
            batch_time_ms: Vec::new(),
            forward_pass_time_ms: 0.0,
            backward_pass_time_ms: 0.0,
            optimizer_time_ms: 0.0,
            memory_usage_mb: None,
            throughput_samples_per_sec: 0.0,
            gradient_norm: None,
            weight_norm: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Calculate average batch processing time
    pub fn avg_batch_time_ms(&self) -> f64 {
        if self.batch_time_ms.is_empty() {
            0.0
        } else {
            self.batch_time_ms.iter().sum::<f64>() / self.batch_time_ms.len() as f64
        }
    }

    /// Get total epoch time in milliseconds
    pub fn total_epoch_time_ms(&self) -> f64 {
        self.forward_pass_time_ms + self.backward_pass_time_ms + self.optimizer_time_ms
    }
}

/// Final training results and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub success: bool,
    pub total_epochs: usize,
    pub total_time_ms: f64,
    pub final_train_loss: f64,
    pub final_validation_loss: Option<f64>,
    pub best_validation_loss: Option<f64>,
    pub early_stopped: bool,
    pub convergence_epoch: Option<usize>,
    pub avg_epoch_time_ms: f64,
    pub total_samples_processed: usize,
    pub overall_throughput: f64,
    pub error: Option<String>,
}

/// Performance profiler for detailed timing analysis
#[derive(Debug)]
pub struct PerformanceProfiler {
    timers: HashMap<String, Instant>,
    durations: HashMap<String, Vec<Duration>>,
    enabled: bool,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            durations: HashMap::new(),
            enabled: true,
        }
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn start_timer(&mut self, name: impl Into<String>) {
        if self.enabled {
            self.timers.insert(name.into(), Instant::now());
        }
    }

    pub fn stop_timer(&mut self, name: impl Into<String>) -> Option<Duration> {
        if !self.enabled {
            return None;
        }

        let name = name.into();
        if let Some(start_time) = self.timers.remove(&name) {
            let duration = start_time.elapsed();
            self.durations
                .entry(name)
                .or_insert_with(Vec::new)
                .push(duration);
            Some(duration)
        } else {
            None
        }
    }

    pub fn get_average_duration(&self, name: &str) -> Option<Duration> {
        self.durations.get(name).map(|durations| {
            let total: Duration = durations.iter().sum();
            total / durations.len() as u32
        })
    }

    pub fn get_total_duration(&self, name: &str) -> Option<Duration> {
        self.durations
            .get(name)
            .map(|durations| durations.iter().sum())
    }

    pub fn reset(&mut self) {
        self.timers.clear();
        self.durations.clear();
    }
}

/// Production-ready training monitor with comprehensive observability
pub struct TrainingMonitor {
    callbacks: Vec<Box<dyn ProgressCallback>>,
    profiler: PerformanceProfiler,
    metrics_history: Vec<EpochMetrics>,
    start_time: Option<Instant>,
    sample_count: usize,
}

impl Default for TrainingMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingMonitor {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            profiler: PerformanceProfiler::new(),
            metrics_history: Vec::new(),
            start_time: None,
            sample_count: 0,
        }
    }

    /// Enable Prometheus metrics export
    #[cfg(feature = "monitoring")]
    pub fn with_prometheus(mut self, port: u16) -> HextralResult<Self> {
        let builder = metrics_exporter_prometheus::PrometheusBuilder::new()
            .with_http_listener(([0, 0, 0, 0], port));

        match builder.install() {
            Ok(_handle) => {
                // Store the handle if needed for later use
                info!("Prometheus metrics server started on port {}", port);
                Ok(self)
            }
            Err(e) => Err(HextralError::config(format!(
                "Failed to start Prometheus server: {}",
                e
            ))),
        }
    }

    /// Add a custom progress callback
    pub fn add_callback<C: ProgressCallback + 'static>(mut self, callback: C) -> Self {
        self.callbacks.push(Box::new(callback));
        self
    }

    /// Initialize tracing for structured logging
    #[cfg(feature = "monitoring")]
    pub fn init_tracing() -> HextralResult<()> {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_target(false)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .init();

        info!("Hextral tracing initialized");
        Ok(())
    }

    /// Record the start of training
    pub fn training_started(&mut self, context: &TrainingContext, sample_count: usize) {
        self.start_time = Some(Instant::now());
        self.sample_count = sample_count;

        #[cfg(feature = "monitoring")]
        {
            info!(
                epoch = context.epoch,
                learning_rate = context.learning_rate,
                samples = sample_count,
                "Training started"
            );
            metrics::counter!("hextral_training_sessions_total", 1);
            metrics::gauge!("hextral_training_samples_total", sample_count as f64);
        }

        for callback in &mut self.callbacks {
            callback.on_training_start(context);
        }
    }

    /// Record epoch completion with comprehensive metrics
    pub fn epoch_completed(&mut self, context: &TrainingContext, metrics: EpochMetrics) {
        #[cfg(feature = "monitoring")]
        {
            info!(
                epoch = metrics.epoch,
                train_loss = metrics.train_loss,
                validation_loss = metrics.validation_loss,
                learning_rate = metrics.learning_rate,
                avg_batch_time_ms = metrics.avg_batch_time_ms(),
                throughput = metrics.throughput_samples_per_sec,
                "Epoch completed"
            );

            // Record metrics
            metrics::gauge!("hextral_train_loss", metrics.train_loss);
            if let Some(val_loss) = metrics.validation_loss {
                metrics::gauge!("hextral_validation_loss", val_loss);
            }
            metrics::gauge!("hextral_learning_rate", metrics.learning_rate);
            metrics::gauge!(
                "hextral_throughput_samples_per_sec",
                metrics.throughput_samples_per_sec
            );
            metrics::histogram!("hextral_epoch_time_ms", metrics.total_epoch_time_ms());

            if let Some(memory_mb) = metrics.memory_usage_mb {
                metrics::gauge!("hextral_memory_usage_mb", memory_mb);
            }
        }

        for callback in &mut self.callbacks {
            callback.on_epoch_end(context, &metrics);
        }

        self.metrics_history.push(metrics);
    }

    /// Record training completion
    pub fn training_completed(&mut self, context: &TrainingContext, result: TrainingResult) {
        #[cfg(feature = "monitoring")]
        {
            if result.success {
                info!(
                    total_epochs = result.total_epochs,
                    total_time_ms = result.total_time_ms,
                    final_loss = result.final_train_loss,
                    throughput = result.overall_throughput,
                    early_stopped = result.early_stopped,
                    "Training completed successfully"
                );
                metrics::counter!("hextral_training_success_total", 1);
            } else {
                error!(
                    total_epochs = result.total_epochs,
                    error = result.error.as_deref().unwrap_or("Unknown error"),
                    "Training failed"
                );
                metrics::counter!("hextral_training_failures_total", 1);
            }

            metrics::histogram!("hextral_training_duration_ms", result.total_time_ms);
            metrics::gauge!("hextral_final_loss", result.final_train_loss);
        }

        for callback in &mut self.callbacks {
            callback.on_training_end(context, &result);
        }
    }

    /// Record error occurrence
    pub fn error_occurred(&mut self, context: &TrainingContext, error: &HextralError) {
        #[cfg(feature = "monitoring")]
        {
            error!(
                epoch = context.epoch,
                error = %error,
                severity = ?error.severity(),
                recoverable = error.is_recoverable(),
                "Training error occurred"
            );
            metrics::counter!("hextral_errors_total", 1);
        }

        for callback in &mut self.callbacks {
            callback.on_error(context, error);
        }
    }

    /// Get training metrics history
    pub fn get_metrics_history(&self) -> &[EpochMetrics] {
        &self.metrics_history
    }

    /// Export metrics to JSON
    pub fn export_metrics_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.metrics_history)
    }

    /// Get profiler for custom timing
    pub fn profiler(&mut self) -> &mut PerformanceProfiler {
        &mut self.profiler
    }
}

/// Default console progress callback
pub struct ConsoleProgressCallback {
    show_progress_bar: bool,
    log_every_n_epochs: usize,
}

impl Default for ConsoleProgressCallback {
    fn default() -> Self {
        Self {
            show_progress_bar: true,
            log_every_n_epochs: 10,
        }
    }
}

impl ConsoleProgressCallback {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_progress_bar(mut self, show: bool) -> Self {
        self.show_progress_bar = show;
        self
    }

    pub fn log_every_n_epochs(mut self, n: usize) -> Self {
        self.log_every_n_epochs = n;
        self
    }
}

impl ProgressCallback for ConsoleProgressCallback {
    fn on_training_start(&mut self, context: &TrainingContext) {
        println!(
            "Training started - Epoch: {}, LR: {:.6}",
            context.epoch, context.learning_rate
        );
    }

    fn on_epoch_end(&mut self, _context: &TrainingContext, metrics: &EpochMetrics) {
        if metrics.epoch % self.log_every_n_epochs == 0 {
            let val_loss_str = metrics
                .validation_loss
                .map(|loss| format!(", Val Loss: {:.6}", loss))
                .unwrap_or_default();

            println!(
                "Epoch {}: Train Loss: {:.6}{}, Throughput: {:.1} samples/sec",
                metrics.epoch, metrics.train_loss, val_loss_str, metrics.throughput_samples_per_sec
            );
        }
    }

    fn on_training_end(&mut self, _context: &TrainingContext, result: &TrainingResult) {
        if result.success {
            println!(
                "Training completed! Final Loss: {:.6}, Total Time: {:.2}s",
                result.final_train_loss,
                result.total_time_ms / 1000.0
            );
            if result.early_stopped {
                println!("Early stopping triggered");
            }
        } else {
            println!(
                "Training failed: {}",
                result.error.as_deref().unwrap_or("Unknown error")
            );
        }
    }

    fn on_error(&mut self, context: &TrainingContext, error: &HextralError) {
        eprintln!(
            "Training error at epoch {}: {} (Recoverable: {})",
            context.epoch,
            error,
            error.is_recoverable()
        );

        if error.is_recoverable() {
            println!("Suggestions:");
            for suggestion in error.recovery_suggestions() {
                println!("   • {}", suggestion);
            }
        }
    }
}
