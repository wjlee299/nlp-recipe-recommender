## NLP Recipe Reccomendation System

## Overview

Self-directed NLP learning project that implements a **hybrid recipe recommendation system** that combines semantic ingredient embeddings with fine-grained, per-term re-ranking to deliver more relevant and robust recipe suggestions. At its core is a tailored Word2Vec ingredient embedding (model **m7**) augmented with TF-IDF weighting to down-weigh common ingredients (eg. salt, sugar).

The pipeline balances global similarity (via average ingredient vectors) with local precision (per-word matching), achieving a multi-stage retrieval that is both efficient and semantically performant.

### Dataset Sourced from Kaggle
[Dataset Source Link](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)

## Demo

Try the live demo on HuggingFace Spaces:  
https://huggingface.co/spaces/wenjin-lee/nlp-recipe-recommender-demo
<img width="1920" height="987" alt="demo" src="https://github.com/user-attachments/assets/a266a747-6c5d-4e3f-a182-7de748f18139" />

## Features

- Custom-trained Word2Vec embeddings on ingredient tokens (model m7)
- Hybrid ranking: initial coarse retrieval followed by TF-IDF-weighted, per-term re-ranking
- Penalization for missing ingredients in similarity scoring
- Gradio demo for interactive exploration

## Learning Outcomes
- Data exploration of embedding neighborhoods and model behavior with matplotlib
- Data pre-processing and formatting (eg. cleaning, tokenization, lemmatization, fixing spelling with SymSpell)
- Word embeddings trained with Word2Vec and document-level representation via average word vectors
- Scoring approach aggregating per-term cosine similarity with TF-IDF
- Designing a multi-stage retrieval pipeline that balances coarse-grained and fine-grained relevance signals
- Gradio demo creation to simulate end-to-end workflow

## Model Details

### Selected Embedding: `model m7`

After evaluating seven Word2Vec variants (m1–m7) on neighbor coherence and similarity strength, **model m7** was chosen for its better semantic fidelity and highest overall similarity scores to good neighbours.

Training configuration for `model m7`:

```python
Word2Vec(
    sentences=df["ingredient_tokens"],
    vector_size=50,
    window=3,
    min_count=3,
    sg=1,
    negative=9,
    sample=1e-4,
    epochs=200,
    workers=3,
)
