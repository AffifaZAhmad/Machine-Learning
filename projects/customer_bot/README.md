# Customer Support Chatbot using DistilBERT

This project implements an AI-powered customer support chatbot using a fine-tuned DistilBERT transformer model for intent classification. The chatbot is trained on a custom intents dataset and deployed using Gradio for interactive use.

## Features

- Transformer-based NLP model (DistilBERT)
- Custom intent classification from JSON dataset
- End-to-end training pipeline using HuggingFace Trainer
- GPU support for faster training and inference
- Interactive web interface using Gradio
- Automatic response mapping based on predicted intent

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas & NumPy
- Gradio

## Project Workflow

1. Load and preprocess intent data from JSON
2. Encode intent labels using LabelEncoder
3. Tokenize text using DistilBERT tokenizer
4. Train DistilBERT for sequence classification
5. Save trained model and label encoder
6. Load model for inference
7. Build a chatbot interface using Gradio

## How to Run

### Install dependencies
```bash
pip install torch transformers pandas numpy scikit-learn gradio joblib
