# Task 1.3: Visual EDA Figures and Captions

### Figure 1: Sample Grid of Images per Class (`sample_grid.png`)
**Caption:** A 2x4 visual inspection grid displaying four randomly chosen images from AiArtData (top row) and RealArt (bottom row), annotated with original pixel dimensions and storage size.

**Interpretation:** The visual samples reveal distinct thematic characteristics across classes. AI-generated artworks demonstrate hyper-smooth textures, surreal lighting, and synthetic stylistic coherence typical of modern diffusion models. In contrast, authentic real artworks display natural brush stroke grain, historical wear, and photographical imperfections. This confirms the perceptual validity of the dataset while underscoring subtle boundary artifacts that a convolutional neural network can exploit.

### Figure 2: Image Width and Height Distributions (`histograms_dimensions.png`)
**Caption:** Histograms and kernel density estimates comparing spatial dimensions (width and height in pixels) across the AiArtData and RealArt classes, marked with the overall median.

**Interpretation:** Both classes exhibit high variance in spatial dimensions, spanning from small thumbnails (~183x135 px) to ultra-high-resolution files (>8000 px), with strong density concentration around standard web resolutions (median width 1024 px for AI vs 920 px for Real). The significant dimensional disparity necessitates an explicit resizing and aspect-ratio handling strategy in the tf.data input pipeline (such as resizing to 224x224 with bilinear interpolation or padding) to prevent geometric distortion during model training.

### Figure 3: File Size Distributions (`histograms_filesize.png`)
**Caption:** Linear (<2000 KB) and logarithmic histograms comparing file storage sizes across AI-generated and authentic artworks.

**Interpretation:** The file size distributions peak sharply between 100 KB and 300 KB (median 165.1 KB), reflecting standard web compression ratios. However, a long heavy right tail extends up to 9.9 MB in the RealArt class due to uncompressed PNG and high-resolution JPEG scans. This underscores the necessity of establishing a strict payload size limit (e.g., 5 MB) on the Flask inference web application to safeguard against denial-of-service and memory spikes during deployment.

### Figure 4: Aspect Ratio and File Formats (`aspect_ratio_and_formats.png`)
**Caption:** Box plots of aspect ratios (width/height) and grouped count charts of image encodings (JPEG vs PNG) per class.

**Interpretation:** The aspect ratio distributions cluster closely around 1.0 (square) and 1.33 (4:3 landscape), though several panoramic and vertical portrait outliers exist in both classes. Furthermore, JPEG dominates both categories (>80%), while PNG comprises the remaining portion. This confirms that the training pipeline must support multi-channel PNG transparency channels by stripping alpha channels down to 3-channel RGB.
