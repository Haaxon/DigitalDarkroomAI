# Digital Darkroom AI 
## An open-source alternative for inverting C41 color film scans

**DigitalDarkroomAI** is an open-source, AI-powered proof-of-concept that explores automated color correction of analog film scans. Traditional approaches use algorithmic software that is sometimes described as automatic, but often still requires human intervention to produce good results. This project investigates two AI-driven methods for adjusting the color of already-inverted film scans, with a focus on preserving the original structure, detail, and grain.

> Note: This project is currently in a prototype phase. It demonstrates the core ideas and model architecture, but it requires further development, real-world data, and optimization to reach a usable quality.

---

## Features

- Two distinct AI-based color correction models:  
  - **Color Matrix Model**: Generates a 3×3 color transformation matrix that is multiplied with the input image.  
  - **Parametric Adjustment Model**: Predicts 9 parameters used to adjust the color of the input image in a flexible way.  
- Tools to generate synthetic training data.  
- Preserves analog detail — no image synthesis or image generation.  
- Fully open-source and reproducible.

---

## Background

Color correction for scanned film negatives is traditionally a manual, time-consuming process. Each film stock, scanner, and exposure introduces different color distortions — especially due to the **orange mask** inherent in C41 negatives. Just inverting the image does not compensate for the mask or the specific color imbalances.

Photographers, both in home setups and in professional lab environments, typically adjust each image by hand to achieve proper color balance. This manual process can take several minutes per image and requires visual judgment and familiarity with film characteristics.

**DigitalDarkroomAI** aims to automate this process using machine learning, while preserving the detail and natural grain of analog film. Unlike many AI tools, this project does not synthesize or "enhance" images, but focuses strictly on improving color information already present in film scans.

---

## Model Architectures

### 1. Color Matrix Model

This model generate a single 3×3 matrix from the input image that is applied uniformly across all pixels in the image. The matrix performs global color correction while leaving image structure untouched.

- Simpl
- Fast inference
- Limited adaptability to local color variance

### 2. Parametric Adjustment Model

This model predicts 9 interpretable color manipulation parameters to apply a more flexible, nonlinear transformation to the image. Parameters control highlights, shadows, and midtones for each RGB channel, offering more nuanced adjustments.

- More expressive than a matrix
- Still interpretable and controllable
- Better suited for variable lighting and exposure conditions

---

## Limitations

- Current models are trained on synthetic data using the Intel Image Dataset and may not generalize well to real-world scans.
- No support for per-film-stock calibration.
- Performance and quality are not yet suitable for real-world use.

---

---

## License

See `LICENSE.md` for full terms.