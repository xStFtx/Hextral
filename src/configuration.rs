use std::path::Path;

use crate::{
    activation::ActivationFunction,
    error::{HextralError, HextralResult},
    optimizer::Optimizer,
    CheckpointConfig, EarlyStopping, Hextral, LossFunction, Regularization,
};

#[cfg(feature = "config")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "config")]
use config as config_crate;

#[cfg(feature = "performance")]
use crate::memory::MemoryConfig;

#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HextralConfig {
    pub network: NetworkConfig,
    #[serde(default)]
    pub training: Option<TrainingConfig>,
    #[cfg(feature = "performance")]
    #[serde(default)]
    pub performance: Option<PerformanceConfig>,
}

#[cfg(feature = "config")]
impl HextralConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> HextralResult<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(HextralError::config(format!(
                "Configuration file '{}' does not exist",
                path.display()
            )));
        }

        let builder = config_crate::Config::builder()
            .add_source(config_crate::File::from(path))
            .add_source(config_crate::Environment::with_prefix("HEXTRAL").separator("__"));

        let settings = builder
            .build()
            .map_err(|e| HextralError::config(format!("Failed to load configuration: {e}")))?;

        settings
            .try_deserialize::<Self>()
            .map_err(|e| HextralError::config(format!("Failed to deserialize configuration: {e}")))
    }

    pub fn from_str(content: &str, format: ConfigFormat) -> HextralResult<Self> {
        match format {
            ConfigFormat::Toml => toml::from_str(content)
                .map_err(|e| HextralError::config(format!("Failed to parse TOML: {e}"))),
            ConfigFormat::Yaml => serde_yaml::from_str(content)
                .map_err(|e| HextralError::config(format!("Failed to parse YAML: {e}"))),
        }
    }

    pub fn builder(self) -> HextralBuilder {
        HextralBuilder::from_config(self)
    }
}

#[cfg(feature = "config")]
#[derive(Debug, Clone, Copy)]
pub enum ConfigFormat {
    Toml,
    Yaml,
}

#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_size: usize,
    #[serde(default)]
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    #[serde(default = "ActivationFunction::default")]
    pub activation: ActivationFunction,
    #[serde(default = "Optimizer::default")]
    pub optimizer: Optimizer,
    #[serde(default = "LossFunction::default")]
    pub loss_function: LossFunction,
    #[serde(default = "Regularization::default")]
    pub regularization: Regularization,
    #[serde(default)]
    pub batch_norm: BatchNormConfig,
}

#[cfg(feature = "config")]
impl NetworkConfig {
    pub fn validate(&self) -> HextralResult<()> {
        if self.input_size == 0 {
            return Err(HextralError::config("input_size must be greater than zero"));
        }
        if self.output_size == 0 {
            return Err(HextralError::config(
                "output_size must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchNormConfig {
    #[serde(default)]
    pub enabled: bool,
}

#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    #[serde(default)]
    pub epochs: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub early_stopping: Option<EarlyStoppingConfig>,
    #[serde(default)]
    pub checkpoint: Option<CheckpointFileConfig>,
}

#[cfg(feature = "config")]
impl TrainingConfig {
    pub fn early_stopping(&self) -> Option<EarlyStopping> {
        self.early_stopping.clone().map(|cfg| cfg.into())
    }

    pub fn checkpoint(&self) -> Option<CheckpointConfig> {
        self.checkpoint.clone().map(|cfg| cfg.into())
    }
}

#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    #[serde(default = "EarlyStoppingConfig::default_min_delta")]
    pub min_delta: f64,
    #[serde(default)]
    pub restore_best_weights: bool,
}

#[cfg(feature = "config")]
impl EarlyStoppingConfig {
    fn default_min_delta() -> f64 {
        1e-4
    }
}

#[cfg(feature = "config")]
impl From<EarlyStoppingConfig> for EarlyStopping {
    fn from(cfg: EarlyStoppingConfig) -> Self {
        EarlyStopping::new(cfg.patience, cfg.min_delta, cfg.restore_best_weights)
    }
}

#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointFileConfig {
    pub filepath: String,
    #[serde(default = "CheckpointFileConfig::default_save_best")]
    pub save_best: bool,
    #[serde(default)]
    pub save_every: Option<usize>,
    #[serde(default = "CheckpointFileConfig::default_monitor_loss")]
    pub monitor_loss: bool,
}

#[cfg(feature = "config")]
impl CheckpointFileConfig {
    fn default_save_best() -> bool {
        true
    }

    fn default_monitor_loss() -> bool {
        true
    }
}

#[cfg(feature = "config")]
impl From<CheckpointFileConfig> for CheckpointConfig {
    fn from(cfg: CheckpointFileConfig) -> Self {
        let mut checkpoint = CheckpointConfig::new(cfg.filepath);
        if let Some(every) = cfg.save_every {
            checkpoint = checkpoint.save_every(every);
        }
        checkpoint.save_best = cfg.save_best;
        checkpoint.monitor_loss = cfg.monitor_loss;
        checkpoint
    }
}

#[cfg(feature = "performance")]
#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceConfig {
    #[serde(default)]
    pub enable_memory_optimization: bool,
    #[serde(default)]
    pub memory: Option<MemorySettings>,
}

#[cfg(feature = "performance")]
#[cfg(feature = "config")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemorySettings {
    #[serde(default)]
    pub enable_tracking: Option<bool>,
    #[serde(default)]
    pub max_pool_size: Option<usize>,
    #[serde(default)]
    pub enable_gc_hints: Option<bool>,
    #[serde(default)]
    pub memory_limit_mb: Option<usize>,
    #[serde(default)]
    pub cleanup_threshold_mb: Option<usize>,
}

#[cfg(feature = "performance")]
#[cfg(feature = "config")]
impl MemorySettings {
    fn to_memory_config(&self) -> MemoryConfig {
        let mut config = MemoryConfig::default();
        if let Some(enable_tracking) = self.enable_tracking {
            config.enable_tracking = enable_tracking;
        }
        if let Some(max_pool_size) = self.max_pool_size {
            config.max_pool_size = max_pool_size;
        }
        if let Some(enable_gc_hints) = self.enable_gc_hints {
            config.enable_gc_hints = enable_gc_hints;
        }
        if let Some(memory_limit) = self.memory_limit_mb {
            config.memory_limit_mb = Some(memory_limit);
        }
        if let Some(cleanup_threshold) = self.cleanup_threshold_mb {
            config.cleanup_threshold_mb = cleanup_threshold;
        }
        config
    }
}

#[cfg(feature = "config")]
#[derive(Debug, Clone)]
pub struct HextralBuilder {
    network: NetworkConfig,
    training: Option<TrainingConfig>,
    #[cfg(feature = "performance")]
    performance: Option<PerformanceConfig>,
}

#[cfg(feature = "config")]
impl HextralBuilder {
    pub fn new(network: NetworkConfig) -> Self {
        Self {
            network,
            training: None,
            #[cfg(feature = "performance")]
            performance: None,
        }
    }

    pub fn from_architecture(
        input_size: usize,
        hidden_layers: Vec<usize>,
        output_size: usize,
    ) -> Self {
        Self::new(NetworkConfig {
            input_size,
            hidden_layers,
            output_size,
            activation: ActivationFunction::default(),
            optimizer: Optimizer::default(),
            loss_function: LossFunction::default(),
            regularization: Regularization::default(),
            batch_norm: BatchNormConfig::default(),
        })
    }

    pub fn from_config(config: HextralConfig) -> Self {
        Self {
            network: config.network,
            training: config.training,
            #[cfg(feature = "performance")]
            performance: config.performance,
        }
    }

    pub fn with_training(mut self, training: TrainingConfig) -> Self {
        self.training = Some(training);
        self
    }

    #[cfg(feature = "performance")]
    pub fn with_performance(mut self, performance: PerformanceConfig) -> Self {
        self.performance = Some(performance);
        self
    }

    pub fn build(self) -> HextralResult<HextralBuild> {
        self.network.validate()?;

        let mut model = Hextral::new(
            self.network.input_size,
            &self.network.hidden_layers,
            self.network.output_size,
            self.network.activation.clone(),
            self.network.optimizer.clone(),
        );

        model.set_loss_function(self.network.loss_function.clone());
        model.set_regularization(self.network.regularization.clone());

        if self.network.batch_norm.enabled {
            model.enable_batch_norm();
        }

        #[cfg(feature = "performance")]
        if let Some(performance) = &self.performance {
            if performance.enable_memory_optimization {
                let memory_config = performance
                    .memory
                    .as_ref()
                    .map(|settings| settings.to_memory_config());
                model.enable_memory_optimization(memory_config);
            }
        }

        Ok(HextralBuild {
            model,
            training: self.training,
        })
    }

    pub fn build_model(self) -> HextralResult<Hextral> {
        self.build().map(|result| result.model)
    }
}

#[cfg(feature = "config")]
pub struct HextralBuild {
    pub model: Hextral,
    pub training: Option<TrainingConfig>,
}

#[cfg(feature = "config")]
impl HextralBuild {
    pub fn into_parts(self) -> (Hextral, Option<TrainingConfig>) {
        (self.model, self.training)
    }

    pub fn training(&self) -> Option<&TrainingConfig> {
        self.training.as_ref()
    }
}
