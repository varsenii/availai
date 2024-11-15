# availai

**`availai`** is a Python library for managing and building computer vision workflows, including tools for dataset management, preprocessing, and seamless integration with [Weights & Biases (W&B)](https://wandb.ai/) and [Roboflow](https://roboflow.com/). Future updates will expand functionality to include broader machine learning, deep learning, audio processing, NLP, and large language models (LLMs).

## Features

- **Dataset Management**: 
  - Download datasets from **Weights & Biases** and **Roboflow** directly into your projects.
  - Upload and log datasets as artifacts on W&B for easy tracking and versioning.
  
- **Integration with Roboflow**:
  - Download datasets in a variety of formats (e.g., YOLOv8).
  - Specify workspaces and projects for precise dataset retrieval.

- **Integration with Weights & Biases**:
  - Fetch specific dataset versions using artifact versioning.
  - Log and visualize data tables and create reproducible dataset workflows.

## Installation

You can install `availai` from PyPI:

```bash
pip install availai
