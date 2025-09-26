use super::{Dataset, DatasetError, DatasetMetadata};
use image::{DynamicImage, imageops::FilterType};
use nalgebra::DVector;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

// Image dataset loader
#[derive(Debug, Clone)]
pub struct ImageLoader {
    // Target image dimensions
    pub target_size: Option<(u32, u32)>,
    // Resize filter type
    pub resize_filter: FilterType,
    // Normalize pixel values
    pub normalize: bool,
    // Convert to grayscale
    pub grayscale: bool,
    // Label extraction strategy
    pub label_strategy: LabelStrategy,
    // Supported file extensions
    pub extensions: Vec<String>,
    // Maximum number of images to load
    pub max_images: Option<usize>,
    // Data augmentation options
    pub augmentation: AugmentationConfig,
}

// Strategy for extracting labels
#[derive(Debug, Clone)]
pub enum LabelStrategy {
    // No labels
    None,
    // Extract from parent directory
    FromDirectory,
    // Extract from filename pattern
    FromFilename(String),
    // Use provided label mapping
    Manual(HashMap<String, usize>),
    // Load labels from separate file
    FromFile(PathBuf),
}

// Image augmentation configuration
#[derive(Debug, Clone, Default)]
pub struct AugmentationConfig {
    // Random horizontal flip probability
    pub horizontal_flip: Option<f32>,
    // Random vertical flip probability
    pub vertical_flip: Option<f32>,
    // Random rotation range
    pub rotation_range: Option<f32>,
    // Random brightness adjustment range
    pub brightness_range: Option<(f32, f32)>,
    // Random contrast adjustment range
    pub contrast_range: Option<(f32, f32)>,
    /// Random noise addition
    pub noise_level: Option<f32>,
}

impl Default for ImageLoader {
    fn default() -> Self {
        Self {
            target_size: Some((224, 224)), // Common CNN input size
            resize_filter: FilterType::Lanczos3,
            normalize: true,
            grayscale: false,
            label_strategy: LabelStrategy::FromDirectory,
            extensions: vec![
                "jpg".to_string(), "jpeg".to_string(), "png".to_string(), 
                "bmp".to_string(), "tiff".to_string(), "tif".to_string(),
                "webp".to_string(),
            ],
            max_images: None,
            augmentation: AugmentationConfig::default(),
        }
    }
}

impl ImageLoader {
    /// Create a new image loader with default settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set target image size for resizing
    pub fn with_target_size(mut self, width: u32, height: u32) -> Self {
        self.target_size = Some((width, height));
        self
    }
    
    /// Disable image resizing
    pub fn no_resize(mut self) -> Self {
        self.target_size = None;
        self
    }
    
    /// Set resize filter type
    pub fn with_resize_filter(mut self, filter: FilterType) -> Self {
        self.resize_filter = filter;
        self
    }
    
    /// Enable/disable pixel normalization
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    
    /// Enable/disable grayscale conversion
    pub fn with_grayscale(mut self, grayscale: bool) -> Self {
        self.grayscale = grayscale;
        self
    }
    
    /// Set label extraction strategy
    pub fn with_label_strategy(mut self, strategy: LabelStrategy) -> Self {
        self.label_strategy = strategy;
        self
    }
    
    /// Set supported file extensions
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }
    
    /// Set maximum number of images to load
    pub fn with_max_images(mut self, max_images: usize) -> Self {
        self.max_images = Some(max_images);
        self
    }
    
    /// Set augmentation configuration
    pub fn with_augmentation(mut self, augmentation: AugmentationConfig) -> Self {
        self.augmentation = augmentation;
        self
    }
    
    /// Load images from a directory
    pub async fn from_directory<P: AsRef<Path>>(&self, dir_path: P) -> Result<Dataset, DatasetError> {
        let paths = self.collect_image_paths(dir_path.as_ref()).await?;
        self.load_from_paths(&paths, Some(dir_path.as_ref().to_string_lossy().to_string())).await
    }
    
    /// Load images from a list of file paths
    pub async fn from_paths(&self, paths: &[PathBuf]) -> Result<Dataset, DatasetError> {
        self.load_from_paths(paths, None).await
    }
    
    /// Load a single image
    pub async fn load_image<P: AsRef<Path>>(&self, path: P) -> Result<DVector<f64>, DatasetError> {
        let img = image::open(path.as_ref())?;
        self.process_image(img).await
    }
    
    /// Collect all valid image paths from a directory (recursive using iterative approach)
    async fn collect_image_paths(&self, dir_path: &Path) -> Result<Vec<PathBuf>, DatasetError> {
        let mut paths = Vec::new();
        let mut dirs_to_process = vec![dir_path.to_path_buf()];
        let mut depth_map = std::collections::HashMap::new();
        depth_map.insert(dir_path.to_path_buf(), 0);
        
        while let Some(current_dir) = dirs_to_process.pop() {
            let depth = depth_map[&current_dir];
            
            // Limit recursion depth to avoid infinite loops
            if depth > 10 {
                continue;
            }
            
            let mut dir = fs::read_dir(&current_dir).await?;
            
            while let Some(entry) = dir.next_entry().await? {
                let path = entry.path();
                
                if path.is_dir() {
                    // Add subdirectory for processing
                    dirs_to_process.push(path.clone());
                    depth_map.insert(path, depth + 1);
                } else if let Some(extension) = path.extension() {
                    let ext = extension.to_string_lossy().to_lowercase();
                    if self.extensions.contains(&ext) {
                        paths.push(path);
                    }
                }
                
                // Apply max images limit
                if let Some(max) = self.max_images {
                    if paths.len() >= max {
                        return Ok(paths);
                    }
                }
                
                // Yield periodically for large directories
                if paths.len() % 100 == 0 {
                    tokio::task::yield_now().await;
                }
            }
        }
        
        Ok(paths)
    }
    
    /// Load images from collected paths
    async fn load_from_paths(&self, paths: &[PathBuf], source: Option<String>) -> Result<Dataset, DatasetError> {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        let mut label_map = HashMap::new();
        let mut next_label_id = 0;
        
        for (index, path) in paths.iter().enumerate() {
            // Load and process image
            let img = image::open(path)?;
            let feature_vector = self.process_image(img).await?;
            features.push(feature_vector);
            
            // Extract label if needed
            let label = match &self.label_strategy {
                LabelStrategy::None => None,
                LabelStrategy::FromDirectory => {
                    if let Some(parent) = path.parent() {
                        if let Some(dir_name) = parent.file_name() {
                            let dir_str = dir_name.to_string_lossy().to_string();
                            let label_id = *label_map.entry(dir_str.clone()).or_insert_with(|| {
                                let id = next_label_id;
                                next_label_id += 1;
                                id
                            });
                            Some(label_id as f64)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                LabelStrategy::FromFilename(pattern) => {
                    if let Some(filename) = path.file_stem() {
                        let filename_str = filename.to_string_lossy().to_string();
                        
                        // Extract label based on pattern type
                        let extracted_label = if pattern == "digit" {
                            // Find first digit in filename
                            filename_str.chars()
                                .find(|c| c.is_ascii_digit())
                                .and_then(|c| c.to_digit(10))
                                .map(|d| d.to_string())
                        } else if pattern == "number" {
                            // Extract first number sequence
                            let mut number_str = String::new();
                            let mut found_digit = false;
                            for c in filename_str.chars() {
                                if c.is_ascii_digit() {
                                    number_str.push(c);
                                    found_digit = true;
                                } else if found_digit {
                                    break;
                                }
                            }
                            if !number_str.is_empty() {
                                Some(number_str)
                            } else {
                                None
                            }
                        } else if pattern.starts_with("split:") {
                            // Split by delimiter and extract part
                            let delimiter = pattern.strip_prefix("split:").unwrap_or("_");
                            filename_str.split(delimiter).next().map(|s| s.to_string())
                        } else if pattern.starts_with("prefix:") {
                            // Extract by prefix length
                            let prefix_len: usize = pattern.strip_prefix("prefix:")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(1);
                            if filename_str.len() >= prefix_len {
                                Some(filename_str[..prefix_len].to_string())
                            } else {
                                Some(filename_str.clone())
                            }
                        } else if pattern.starts_with("suffix:") {
                            // Extract by suffix length
                            let suffix_len: usize = pattern.strip_prefix("suffix:")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(1);
                            if filename_str.len() >= suffix_len {
                                let start = filename_str.len() - suffix_len;
                                Some(filename_str[start..].to_string())
                            } else {
                                Some(filename_str.clone())
                            }
                        } else {
                            // Default: look for the pattern as a substring and extract surrounding context
                            if filename_str.contains(pattern) {
                                Some(pattern.to_string())
                            } else {
                                None
                            }
                        };
                        
                        if let Some(label_str) = extracted_label {
                            let label_id = *label_map.entry(label_str).or_insert_with(|| {
                                let id = next_label_id;
                                next_label_id += 1;
                                id
                            });
                            Some(label_id as f64)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                LabelStrategy::Manual(mapping) => {
                    if let Some(filename) = path.file_stem() {
                        let filename_str = filename.to_string_lossy().to_string();
                        mapping.get(&filename_str).map(|&id| id as f64)
                    } else {
                        None
                    }
                },
                LabelStrategy::FromFile(label_file) => {
                    let mut result = None;
                    if let Ok(content) = std::fs::read_to_string(label_file) {
                        if let Some(filename) = path.file_stem() {
                            let filename_str = filename.to_string_lossy().to_string();
                            for line in content.lines() {
                                let parts: Vec<&str> = line.split_whitespace().collect();
                                if parts.len() >= 2 && parts[0] == filename_str {
                                    if let Ok(label_id) = parts[1].parse::<usize>() {
                                        result = Some(label_id as f64);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    result
                },
            };
            
            if let Some(label_val) = label {
                targets.push(DVector::from_vec(vec![label_val]));
            }
            
            // Yield periodically for large datasets
            if index % 50 == 0 {
                tokio::task::yield_now().await;
            }
        }
        
        // Create targets vector if we have labels
        let final_targets = if targets.is_empty() {
            None
        } else {
            Some(targets)
        };
        
        // Calculate image dimensions
        let image_dims = if let Some((width, height)) = self.target_size {
            let channels = if self.grayscale { 1 } else { 3 };
            width * height * channels
        } else {
            // Use dimensions from first image if no target size
            features.first().map(|f| f.len() as u32).unwrap_or(0)
        };
        
        // Create metadata
        let metadata = DatasetMetadata {
            sample_count: features.len(),
            feature_count: image_dims as usize,
            target_count: if !label_map.is_empty() { Some(label_map.len()) } else { None },
            source,
            data_type: Some("Image".to_string()),
        };
        
        // Create class names from label map
        let target_names = if !label_map.is_empty() {
            let mut class_names = vec!["".to_string(); label_map.len()];
            for (name, &id) in &label_map {
                if id < class_names.len() {
                    class_names[id] = name.clone();
                }
            }
            Some(class_names)
        } else {
            None
        };
        
        let mut dataset = Dataset::new(features, final_targets);
        dataset.feature_names = Some(vec!["pixel".to_string(); image_dims as usize]); // Generic pixel names
        dataset.target_names = target_names;
        dataset.metadata = metadata;
        
        Ok(dataset)
    }
    
    /// Process a single image into a feature vector
    async fn process_image(&self, mut img: DynamicImage) -> Result<DVector<f64>, DatasetError> {
        // Convert to grayscale if requested
        if self.grayscale {
            img = img.grayscale();
        }
        
        // Resize if target size is specified
        if let Some((width, height)) = self.target_size {
            img = img.resize_exact(width, height, self.resize_filter);
        }
        
        // Apply augmentation if configured
        img = self.apply_augmentation(img).await;
        
        // Convert to pixel values
        let rgb_img = img.to_rgb8();
        let (_width, _height) = rgb_img.dimensions();
        let pixels = rgb_img.into_raw();
        
        // Convert to f64 and normalize if requested
        let pixel_values: Vec<f64> = if self.normalize {
            pixels.iter().map(|&p| p as f64 / 255.0).collect()
        } else {
            pixels.iter().map(|&p| p as f64).collect()
        };
        
        Ok(DVector::from_vec(pixel_values))
    }
    
    /// Apply augmentation to an image
    async fn apply_augmentation(&self, mut img: DynamicImage) -> DynamicImage {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Horizontal flip
        if let Some(prob) = self.augmentation.horizontal_flip {
            if rng.gen::<f32>() < prob {
                img = img.fliph();
            }
        }
        
        // Vertical flip
        if let Some(prob) = self.augmentation.vertical_flip {
            if rng.gen::<f32>() < prob {
                img = img.flipv();
            }
        }
        
        if let Some(range) = self.augmentation.rotation_range {
            let rotation = rng.gen_range(-range..=range);
            if rotation.abs() >= 45.0 {
                let times = ((rotation / 90.0).round() as i32).rem_euclid(4);
                for _ in 0..times {
                    img = img.rotate90();
                }
            }
        }
        
        // Brightness adjustment
        if let Some((min_brightness, max_brightness)) = self.augmentation.brightness_range {
            let brightness_factor = rng.gen_range(min_brightness..=max_brightness);
            img = self.adjust_brightness(img, brightness_factor);
        }
        
        // Contrast adjustment
        if let Some((min_contrast, max_contrast)) = self.augmentation.contrast_range {
            let contrast_factor = rng.gen_range(min_contrast..=max_contrast);
            img = self.adjust_contrast(img, contrast_factor);
        }
        
        // Noise addition
        if let Some(noise_level) = self.augmentation.noise_level {
            if noise_level > 0.0 {
                img = self.add_noise(img, noise_level, &mut rng);
            }
        }

        img
    }
    
    /// Adjust image brightness
    fn adjust_brightness(&self, img: DynamicImage, factor: f32) -> DynamicImage {
        let mut rgb_img = img.to_rgb8();
        
        for pixel in rgb_img.pixels_mut() {
            for channel in &mut pixel.0 {
                let new_value = (*channel as f32 * factor).clamp(0.0, 255.0) as u8;
                *channel = new_value;
            }
        }
        
        DynamicImage::ImageRgb8(rgb_img)
    }
    
    /// Adjust image contrast
    fn adjust_contrast(&self, img: DynamicImage, factor: f32) -> DynamicImage {
        let mut rgb_img = img.to_rgb8();
        
        // Calculate mean pixel value for contrast adjustment
        let mut sum = 0u32;
        let mut count = 0u32;
        for pixel in rgb_img.pixels() {
            for &channel in &pixel.0 {
                sum += channel as u32;
                count += 1;
            }
        }
        let mean = (sum as f32) / (count as f32);
        
        for pixel in rgb_img.pixels_mut() {
            for channel in &mut pixel.0 {
                let diff = *channel as f32 - mean;
                let new_value = (mean + diff * factor).clamp(0.0, 255.0) as u8;
                *channel = new_value;
            }
        }
        
        DynamicImage::ImageRgb8(rgb_img)
    }
    
    /// Add random noise to image
    fn add_noise(&self, img: DynamicImage, noise_level: f32, rng: &mut impl rand::Rng) -> DynamicImage {
        let mut rgb_img = img.to_rgb8();
        
        for pixel in rgb_img.pixels_mut() {
            for channel in &mut pixel.0 {
                let noise = rng.gen_range(-noise_level..=noise_level) * 255.0;
                let new_value = (*channel as f32 + noise).clamp(0.0, 255.0) as u8;
                *channel = new_value;
            }
        }
        
        DynamicImage::ImageRgb8(rgb_img)
    }
}

/// Image augmentation functions
impl AugmentationConfig {
    /// Create a new augmentation config
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set horizontal flip probability
    pub fn with_horizontal_flip(mut self, probability: f32) -> Self {
        self.horizontal_flip = Some(probability);
        self
    }
    
    /// Set vertical flip probability
    pub fn with_vertical_flip(mut self, probability: f32) -> Self {
        self.vertical_flip = Some(probability);
        self
    }
    
    /// Set rotation range
    pub fn with_rotation(mut self, degrees: f32) -> Self {
        self.rotation_range = Some(degrees);
        self
    }
    
    /// Set brightness adjustment range
    pub fn with_brightness(mut self, min: f32, max: f32) -> Self {
        self.brightness_range = Some((min, max));
        self
    }
    
    /// Set contrast adjustment range
    pub fn with_contrast(mut self, min: f32, max: f32) -> Self {
        self.contrast_range = Some((min, max));
        self
    }
    
    /// Set noise level
    pub fn with_noise(mut self, level: f32) -> Self {
        self.noise_level = Some(level);
        self
    }
}