# Text Summarization with PEGASUS Model: Generative AI and Abstractive Summarization

## Overview
This project leverages the PEGASUS model to perform abstractive text summarization on the SAMSum dataset. The goal is to preprocess, fine-tune, and evaluate the model to generate high-quality summaries of dialogues. The project emphasizes generative AI techniques for creating concise and coherent summaries.

## Data Collection

### SAMSum Dataset
- **Dataset Source:** SAMSum dataset, which consists of dialogues and their corresponding summaries.
- **Purpose:** Utilized for training and evaluating the summarization model.

## Data Preprocessing

### Cleaning and Tokenization
- **Text Cleaning:** Removed extraneous elements such as special characters and non-textual data.
- **Tokenization:** Converted text into tokens using the PEGASUS tokenizer.

### Token Length Analysis
- Analyzed token lengths for both dialogues and summaries to understand dataset characteristics and ensure proper model input handling.

## Model Training

### PEGASUS Model
- **Model Choice:** Implemented the PEGASUS model from Hugging Face's Transformers library.
- **Training:** Fine-tuned the model on the SAMSum dataset using GPU acceleration to optimize performance.

### Evaluation
- **Metrics:** Evaluated model performance using ROUGE metrics to assess the quality of generated summaries.
- **Optimization:** Tuned training parameters and utilized GPU resources for efficient model training.

## Skills Acquired
- **Transformers Library:** Proficient use of Hugging Face’s Transformers for NLP tasks.
- **Generative AI:** Advanced understanding of generative AI techniques for text summarization.
- **GPU Resource Management:** Experience in managing GPU resources to accelerate model training.

## Dependencies
- `transformers`
- `datasets`
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`

## Learning Outcomes
Working on this project provided insights into:
- **Preprocessing Text Data:** Effective techniques for text cleaning and tokenization.
- **Model Fine-Tuning:** Practical experience in fine-tuning state-of-the-art NLP models.
- **Evaluation Metrics:** Applying and interpreting ROUGE metrics for summarization quality.
