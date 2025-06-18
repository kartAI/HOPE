# A New HOPE

<div class="image-container">
    <img src="cover.png" alt="A small robot jedi knight slashing text with a lightsaber for A new HOPE logo" />
</div>

# 

This repository contains the code implementation for the HOPE method introduced in "A new HOPE: Domain-agnostic Automatic Evaluation of Text Chunking" which is accepted for the SIGIR25 conference. 


Full paper: https://arxiv.org/abs/2505.02171


## Abstract
Document chunking fundamentally impacts Retrieval-Augmented Generation (RAG) by determining how source materials are segmented before indexing. Despite evidence that Large Language Models (LLMs) are sensitive to the layout and structure of retrieved data, there is currently no framework to analyze the impact of different chunking methods. In this paper, we introduce a novel methodology that defines essential characteristics of the chunking process at three levels: intrinsic passage properties, extrinsic passage properties, and passages-document coherence. We propose HOPE (Holistic Passage Evaluation), a domain-agnostic, automatic evaluation metric that quantifies and aggregates these characteristics. Our empirical evaluations across seven domains demonstrate that the HOPE metric correlates significantly (p > 0.13) with various RAG performance indicators, revealing contrasts between the importance of extrinsic and intrinsic properties of passages. Semantic independence between passages proves essential for system performance with a performance gain of up to 56.2% in factual correctness and 21.1% in answer correctness. On the contrary, traditional assumptions about maintaining concept unity within passages show minimal impact. These findings provide actionable insights for optimizing chunking strategies, thus improving RAG system design to produce more factually correct responses.


## Setup
Everything is configured in a python package. Perform local installation using: 

```bash
pip install -e .
```

### Environment 
The code is developed with the following:
- OS: Linux
- Python: 3.10
- CUDA: 11.8


## Configuration
See the provided `.env` file for reference. 


## Usage

We recommend to only use the `Hope` class to calculate the HOPE metric or any of the sub-metrics. The class requires a LangChain chat model and a LangChain embedding model as input parameters. The class returns a `HopeScore` object when the `evaluate` method in invoked.

```py
from langchain_core.documents import Document

from hope.hope import Hope

hope = Hope(
    my_llm_model,
    my_embedding_model
)

...

document = Document("my text ...")
chunks = my_chunking_method(document)

...

hope_score = hope.evaluate(
    document,
    chunks,
    document_name = "My document",
)
```