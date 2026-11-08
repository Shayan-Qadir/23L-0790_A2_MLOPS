# Bonus Task: Grad-CAM Interpretability Analysis

### What the Model Looks At:
1. **Correct Predictions (Top Row):**
   - When successfully identifying AI-generated imagery, the Grad-CAM activation heatmaps demonstrate intense localized focus on fine semantic boundaries, overly smooth skin/surface gradients, and background bokeh regions where diffusion synthesis leaves characteristic frequency anomalies.
   - For authentic artworks, attention is distributed across natural canvas texture, physical pigment brush strokes, and realistic light refraction patterns.

2. **Misclassified Predictions (Bottom Row):**
   - In failure modes, model attention frequently drifts away from central subjects toward peripheral borders, watermarks, or high-contrast artistic signatures.
   - Highly stylized real digital art with clean vector gradients is misdiagnosed because the feature maps falsely trigger on the glossy, noise-free pixel regions typically associated with generative diffusion prompts.
