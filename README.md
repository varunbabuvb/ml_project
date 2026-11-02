# Cross-Cultural and Demographic Robustness of Set-of-Vision Prompting for Zero-Shot Emotion Recognition

## Project Overview

This project investigates Set-of-Vision (SoV) prompting for zero-shot facial emotion recognition and evaluates its performance across different visual prompting configurations, demographic groups, and multi-face settings.

The study evaluates two open vision-language models:

- **LLaVA-1.5-7B**
- **Qwen2.5-VL-7B-Instruct**

The project is organized into two phases:

- **Phase 1 – Pipeline Implementation:** Face detection, image preprocessing, context-image generation, prompt construction, and model inference.
- **Phase 2 – Experimental Evaluation:** Performance analysis, demographic robustness evaluation, multi-face experiments, tag probing, and ablation studies.

The objective is to examine how visual prompting configurations influence emotion recognition performance and whether the observed behaviour varies across demographic groups and experimental settings.

## Team Gradient

| Name | Roll Number |
|---|---|
| Thammandra Saketh Ram | B24DS034 |
| Veeramallu Varun Babu | B24DS037 |
| Yellampati Lukesh | B24DS039 |

## Models Evaluated

| Model | Configuration |
|---|---|
| LLaVA-1.5-7B | 4-bit quantized inference |
| Qwen2.5-VL-7B-Instruct | 4-bit quantized inference |

Both models were evaluated using the same experimental framework to support comparison across prompting conditions.

## Experimental Conditions

The experiments investigate the following visual prompting configurations:

1. **Plain:** Original face image without additional visual annotations.
2. **Box:** Face image with bounding-box annotations.
3. **Box + Number:** Bounding boxes with numerical identifiers.
4. **SoV:** Set-of-Vision prompting configuration.

Additional experiments include:

- **Half-dot ablation:** Evaluates the effect of modifying the visual dot representation.
- **Multi-face evaluation:** Examines performance when multiple faces appear in the same context image.
- **Tag probing:** Investigates model responses to demographic and visual tags.
- **Demographic analysis:** Evaluates performance across estimated demographic groups.

## Repository Structure

```text
ML_PROJECT/
│
├── sov pipeline/                 # Phase 1 pipeline
│   ├── sov_stage1_2_detect.py
│   ├── Sov_stage3_5_overlay.py
│   ├── sov_stage6_prompts.py
│   └── sov_stage7_infer.py
│
├── phase2/
│   ├── preprocessing/            # Data preprocessing and image preparation
│   ├── inference/                # Job generation and model inference utilities
│   ├── evaluation/               # Evaluation and analysis scripts
│   └── notebooks/                # Phase 2 experiment notebooks
│
├── results/
│   ├── raw/
│   │   ├── llava/                # LLaVA inference outputs
│   │   ├── qwen/                 # Qwen inference outputs
│   │   ├── tag_probes/           # Tag-probing outputs
│   │   └── jobs/                 # Generated inference jobs
│   │
│   ├── analysis/
│   │   ├── comparison/
│   │   ├── llava_core/
│   │   ├── llava_multi/
│   │   ├── qwen_core/
│   │   └── qwen_multi/
│   │
│   ├── ablation/
│   │   └── half_dot/
│   │
│   ├── class_contributions/
│   │   ├── llava/
│   │   └── qwen/
│   │
│   └── figures/
│
├── reports/
│   ├── main.tex
│   ├── references.bib
│   ├── report ML PROJECT.pdf
│   └── figures/
│
├── images/                       # Project images and visual assets
├── requirements.txt
├── .gitignore
└── README.md
```

## Phase 1 – Pipeline Implementation

Phase 1 establishes the SoV processing and inference pipeline.

The main stages include:

1. Face detection and preprocessing.
2. Face cropping and dataset preparation.
3. Context-image generation.
4. Visual overlay and prompt construction.
5. Vision-language model inference.

The implementation is available in the `sov pipeline/` directory.

## Phase 2 – Experimental Evaluation

Phase 2 extends the pipeline to support systematic experiments and model evaluation.

### Preprocessing

- Face cropping and dataset preparation.
- Construction of core and multi-face evaluation sets.
- Context-image generation.
- Preparation of inference jobs.

### Inference

- Generation of model-specific inference jobs.
- LLaVA and Qwen inference.
- Core and multi-face experiment execution.
- Half-dot ablation inference.

### Evaluation and Analysis

- Accuracy and balanced accuracy evaluation.
- Macro-F1 and class-level performance analysis.
- Demographic group comparisons.
- Multi-face evaluation.
- Tag-probing analysis.
- Model comparison and ablation analysis.

The corresponding scripts, notebooks, and experiment outputs are organized under `phase2/` and `results/`.

## Key Experimental Findings

The core evaluation produced the following balanced accuracy results:

| Condition | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| Plain | 0.428 | 0.493 |
| Box | 0.419 | 0.492 |
| Box + Number | 0.417 | 0.498 |
| SoV | 0.374 | 0.412 |

Additional observations:

- SoV prompting did not improve core balanced accuracy over the plain-image condition in either evaluated model.
- The half-dot ablation showed improvements over the corresponding SoV condition in both models.
- Neutral-class prediction behaviour differed substantially between LLaVA and Qwen.
- Multi-face experiments showed different performance changes across the two models.
- Demographic group analysis revealed variation across estimated demographic categories.

These findings are specific to the evaluated datasets, models, and experimental configurations. They should not be interpreted as universal conclusions about the models.

## Code Reproducibility

The complete experimental workflow is documented in the Jupyter notebooks provided in the repository, particularly those under `phase2/notebooks/`.

To reproduce the experiments:

1. Clone the repository and install the required dependencies using `requirements.txt`.
2. Open the relevant Jupyter notebook in Google Colab.
3. Upload the required input files and datasets to the corresponding Google Colab `/content/` and Google Drive folders, following the directory structure expected by the notebook.
4. Configure the required model and dataset paths.
5. Execute the notebook cells in sequence to reproduce the preprocessing, inference, and evaluation stages.

**Note:** The datasets and pretrained model weights are not included in this repository and must be obtained separately. GPU resources are required for model inference.

The repository also contains saved inference outputs and analysis results, allowing the reported experiments to be inspected without rerunning the complete inference pipeline.

## Results and Analysis

The `results/` directory contains the experimental outputs:

- `raw/` – Raw model predictions, inference jobs, and tag-probing outputs.
- `analysis/` – Processed metrics and experimental analysis.
- `ablation/` – Half-dot ablation results.
- `class_contributions/` – Class-level contribution analysis.
- `figures/` – Generated experimental visualizations.

## Final Report

The final project report is available at:

`reports/report ML PROJECT.pdf`

The LaTeX source and bibliography are also provided:

- `reports/main.tex`
- `reports/references.bib`

All figures used in the report are stored in `reports/figures/`.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Hugging Face Transformers
- Vision-Language Models
- Jupyter Notebook
- Google Colab
- LaTeX

## Acknowledgements

We acknowledge the authors of the datasets, models, and research resources used in this project.