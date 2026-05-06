# KLoRA

This repository contains the core implementation of **KLORA**, a unified framework that integrates LoRA initialization and adaptive layer allocation to achieve a preferable starting point, faster convergence, and improved performance. 

By projecting knowledge vectors derived from a fine-tuned small model into the parameter space of a larger target model, KLORA establishes a superior optimization starting point for downstream tasks. To resolve architectural mismatches and minimize GPU memory usage, KLORA formulates a projection error-aware mechanism, which is efficiently solved via dynamic programming to selectively allocate LoRA to the most critical layers.

## Repository Structure

The provided core codes focus on the two main stages of the KLORA framework:

1. **`ILP_DP.py` (Stage 2: Error-aware Adaptive LoRA Allocation)**
   - Computes the projection-error cost matrix between the layers of the small model and the large model.
   - Solves an Integer Linear Programming (ILP) problem with monotonicity constraints using a Dynamic Programming (DP) algorithm.
   - Outputs the optimal monotonic layer mapping to identify the most significant layers in the large model for knowledge-informed LoRA initialization and adaptation.

2. **`KLoRA_Init.py` (Stage 1: Knowledge-informed LoRA Initialization)**
   - Extracts a task vector (knowledge vector) from a fine-tuned small model.
   - Factorizes the knowledge vector via Singular Value Decomposition (SVD).
   - Projects the decomposed matrices into the parameter space of the large model using the mappings identified in Stage 2.
   - Generates the initialized LoRA weights (`.pth` file) ready to be loaded into the target large language model.

## Installation & Dependencies

Ensure you have the following core libraries installed:

```
pip install torch transformers numpy matplotlib
```

*Note: For the full training and inference pipelines, you will also need `peft` and the PiSSA framework codebase. See the Acknowledgements section below.*

## Usage Workflow

The full KLORA pipeline operates in the following sequential steps:

### Step 1: Extract Target-Domain Knowledge (Small Model SFT)
First, fine-tune a small model (e.g., Qwen2.5-0.5B) on your target-domain dataset for a few epochs to capture domain-specific knowledge vectors. Save the fine-tuned checkpoint.

### Step 2: Determine Optimal Layer Mapping (`ILP_DP.py`)
Run the layer allocation script to calculate the minimal projection error path between the small model and your target large model (e.g., Qwen2.5-3B).
```
python ILP_DP.py
```
* **Input**: Base Small Model and Base Large Model paths.
* **Output**: The exact layer indices mapping for minimal projection error.

### Step 3: Initialize KLORA Weights (`KLoRA_Init.py`)
Using the mapping from Step 2, project the fine-tuned small model's knowledge into the large model's parameter space.
```bash
python KLoRA_Init.py
```
* **Input**: Base Small Model, Fine-tuned Small Model, and Base Large Model paths.
* **Output**: A `.pth` file containing the initialized LoRA matrices (e.g., `svd_r8_mmlu_qwen2505b23b.pth`).

### Step 4: Fine-tune the Large Model
Load the generated `.pth` initialization weights into your target large model. You can seamlessly integrate these weights by loading them into standard LoRA components within the PEFT/PiSSA framework. Finally, perform the adaptation on the large model using your target task data.

## Acknowledgements

The training and inference pipelines for KLORA build upon the robust parameter-efficient fine-tuning ecosystems established by the community. We strictly base our training loops and model loading mechanisms on these repositories:

* **[PiSSA](https://github.com/MuLabPKU/PiSSA):** Principal Singular Values and Singular Vectors Adaptation of Large Language Models.
* **[PEFT](https://huggingface.co/docs/peft/index):** Hugging Face's Parameter-Efficient Fine-Tuning library. 

We extend our sincere gratitude to the authors and maintainers of these projects for their foundational tools.
