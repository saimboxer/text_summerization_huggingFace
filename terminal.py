# Import the pipeline and set_seed function from the transformers library
# The pipeline function provides an easy way to use pretrained models for various tasks
# such as text generation, summarization, translation, and more
# The set_seed function is used to ensure reproducibility of results by setting a random seed
from transformers import pipeline, set_seed

# Import matplotlib for plotting graphs and visualizations
# This library is used to create static, animated, and interactive visualizations in Python
# Useful for displaying data and model performance
import matplotlib.pyplot as plt

# Import pandas for data manipulation and analysis
# Pandas is a powerful data analysis and manipulation library for Python
# Useful for handling datasets, reading/writing CSV files, and data preprocessing
import pandas as pd

# Import the AutoModelForSeq2SeqLM and AutoTokenizer classes from the transformers library
# AutoModelForSeq2SeqLM is a generic model class for sequence-to-sequence language modeling
# Useful for tasks such as translation, summarization, and text generation
# AutoTokenizer is used for tokenizing input text to the format required by the model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Import the nltk library and the sent_tokenize function for sentence tokenization
# nltk (Natural Language Toolkit) is a suite of libraries and programs for natural language processing
# Useful for various text processing tasks like tokenization, lemmatization, and more
# sent_tokenize is used to split a text into a list of sentences
import nltk
from nltk.tokenize import sent_tokenize
print("All libraries imported successfully")

# Import the tqdm library for creating progress bars
# tqdm is used to show progress bars for loops, making it easier to track the progress of operations
# Useful for monitoring the progress of tasks such as data processing and model training
from tqdm import tqdm

# Import the torch library for PyTorch functionalities
# PyTorch is an open-source machine learning library used for applications such as computer vision and natural language processing
# Provides tools for tensor computation, automatic differentiation, and more
import torch

# Download the "punkt" tokenizer model from nltk
# The "punkt" tokenizer is a pre-trained model for tokenizing text into sentences
# Useful for splitting a large text into individual sentences for further processing
# nltk.download("punkt")

# Import the function to load datasets from the 'datasets' library
# The 'datasets' library provides a wide range of datasets and tools for handling and processing data,
# making it easier to access datasets from various sources including the Hugging Face hub or local files.
from datasets import load_dataset

# Import the function to load evaluation metrics from the 'datasets' library
# Metrics are used to evaluate the performance of machine learning models. The 'datasets' library includes 
# several standard metrics, allowing you to assess how well your model's predictions align with human evaluations.
from datasets import load_metric

# Importing AutoModelForSeq2SeqLM and AutoTokenizer from the 'transformers' library

# AutoModelForSeq2SeqLM is a class that provides a generic interface to any pre-trained sequence-to-sequence model.
# Sequence-to-sequence models are used for tasks like text summarization, translation, and other tasks where 
# an input sequence is transformed into an output sequence.
from transformers import AutoModelForSeq2SeqLM

# AutoTokenizer is a class that provides a tokenizer for any pre-trained model.
# Tokenizers convert text into a format that the model can understand (e.g., converting text to tokens or IDs).
# This is a crucial step before passing text data to the model for processing.
from transformers import AutoTokenizer

# Set the device to "cuda" if a GPU with CUDA support is available, otherwise use "cpu".
# This allows you to leverage GPU acceleration for faster model training and inference if a compatible GPU is present.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the model checkpoint identifier for a pre-trained model.
# In this case, 'google/pegasus-cnn_dailymail' refers to a specific pre-trained PEGASUS model fine-tuned on the CNN/DailyMail dataset.
# This model is used for tasks like text summarization, leveraging its pre-trained capabilities to generate summaries from input text.
model_ckpt = "google/pegasus-cnn_dailymail"

# Load the tokenizer associated with the pre-trained model checkpoint.
# The tokenizer converts text into tokens or IDs that the model can process.
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Load the pre-trained model specified by the checkpoint and move it to the specified device (CPU or GPU).
# The model is used for generating predictions based on the input data.
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)


print(f"Using device: {device}")
print(f"Using model checkpoint: {model_ckpt}")
print(f"Tokenizer type: {type(tokenizer)}")
print(f"Model type: {type(model_pegasus)}")


def generate_batch_sized_chunks(list_of_elements, batch_size):
    """
    Splits a list into smaller batches of a specified size.

    This function is useful for processing large datasets in smaller,
    more manageable chunks, especially when dealing with memory constraints
    or parallelizing computations.

    Args:
        list_of_elements: The list to be split into batches.
        batch_size: The desired size of each batch.

    Yields:
        Successive batch-sized chunks from the input list.
    """
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                               batch_size=8, device=device,
                               column_text="article",
                               column_summary="highlights"):
    """
    Evaluates a summarization model on a test dataset using the specified metric.

    Args:
    - dataset: The test dataset containing the text and summaries.
    - metric: The evaluation metric (e.g., ROUGE) to compute.
    - model: The pre-trained summarization model.
    - tokenizer: The tokenizer associated with the model.
    - batch_size: The number of samples to process in each batch.
    - device: The device (CPU/GPU) to run the model on.
    - column_text: The column name in the dataset containing the articles.
    - column_summary: The column name in the dataset containing the reference summaries.

    Returns:
    - score: The computed metric score.
    """

    # Split the articles and summaries into batches
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    # Loop through each batch of articles and corresponding summaries
    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):

        # Tokenize the articles in the batch
        inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                           padding="max_length", return_tensors="pt")

        # Generate summaries using the model
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                   attention_mask=inputs["attention_mask"].to(device),
                                   length_penalty=0.8, num_beams=8, max_length=128)
        # length_penalty ensures that the model does not generate sequences that are too long

        # Decode the generated summaries into readable text
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
                             for s in summaries]

        # Replace empty strings with a space to avoid issues
        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

        # Add the decoded summaries and the reference summaries to the metric
        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    # Compute and return the final metric score
    score = metric.compute()
    return score

# Load the 'samsum' dataset using the 'datasets' library.
# The 'samsum' dataset is a dialogue summarization dataset consisting of dialogues and their corresponding summaries.
# This dataset is useful for training and evaluating models on tasks such as text summarization and dialogue summarization.
dataset_samsum = load_dataset("samsum")
print(dataset_samsum)

# Initialize the summarization pipeline
# The 'pipeline' function from the 'transformers' library simplifies the process of using pre-trained models.
# 'summarization' indicates that we are creating a pipeline for generating summaries of text.
# 'model=model_ckpt' specifies the pre-trained model to be used for the summarization task.
pipe = pipeline('summarization', model=model_ckpt)

# Load the ROUGE metric for evaluation
rouge_metric = load_metric('rouge', trust_remote_code=True)

# # Calculate the ROUGE score on the test dataset
# score = calculate_metric_on_test_ds(
#     dataset_samsum['test'],  # The test dataset split
#     rouge_metric,            # The ROUGE metric to use
#     model_pegasus,          # The pre-trained summarization model
#     tokenizer,              # The corresponding tokenizer
#     column_text='dialogue',   # The column containing the input text
#     column_summary='summary', # The column containing the reference summaries
#     batch_size=8             # Batch size for processing
# )

# rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
# rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

# pd.DataFrame(rouge_dict, index = ['pegasus'])