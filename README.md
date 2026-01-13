# Prerequisites

## Install VGT
Please follow VGT's [documentation](https://github.com/huridocs/pdf-document-layout-analysis).

## Install packages
Under the root directory `SHTRAG/`:
```
$ conda env create --file my_env.yml
$ conda activate shtrag
$ pip install -r raptor/requirements.txt
$ pip install --upgrade pymupdf
$ pip install anytree
$ pip install python-dotenv
$ pip install datasets
$ pip install rank_bm25
$ pip install colorlog
$ pip install matplotlib
```

### Issues
[[issue](https://github.com/easydiffusion/easydiffusion/issues/1851)] If you encounter this error, you may directly remove `cached_download` from huggingface_hub import line:
```
> cannot import name 'cached_download' from 'huggingface_hub'
```
or set the version
```
$ pip install --upgrade huggingface_hub==0.25.2
```

[[issue](https://community.openai.com/t/subject-error-openai-object-has-no-attribute-batches/745724/2)] If you encounter this error when using batch api:
```
> AttributeError: 'OpenAI' object has no attribute 'batches'
```
you may update openai:
```
$ pip install --upgrade openai --quiet
```

## OpenAI key
Create a `.env` file under the root directory `SHTRAG/`:
```
API_KEY=your-openai-key
```

# Example Usage
## Data preparation
Create your dataset under folder `data/`. All data will be stored under folder `data/your-dataset/`. Store the pdf files under folder `data/your-dataset/pdf/`. Store the queries as a list of dictonaries in `data/your-dataset/queries.json`. 

Example: [01262022-1835.pdf](./data/example/pdf/01262022-1835.pdf), [queries.json](./data/example/queries.json)

## Heading identification
To identify the headings of a pdf, we use VGT:
```
$ curl -X POST -F 'file=@path-to-your-pdf' localhost:5060
```
VGT returns a json file, classifying the texts of the pdf into ten categories. Store this json file under folder `data/your-dataset/heading_identification`. 

Example: [01262022-1835.json](./data/example/heading_identification/01262022-1835.json)

## Sound SHT extraction & Structured-RAG
After identifying the headings, it is time to build the SHT and to incorporate it into Structured-RAG. For implementation details , please read [structured_rag/README.md](./structured_rag/README.md). 
```
> python run_structured_rag.py --root-dir path-to-your-dataset
```
For example, you can use `data/your-dataset/`. It is recommended to **always** use **absolute path**. 

Below lists the arguments to set configuration:


|flag|required|default|explanation|example|
|:-:|:-:|:-:|:-:|:-:|
|root-dir|True|-|(absolute) path to your dataset folder|`--root-dir ./data/example/`|
|chunk-size|False|100|size of a chunk (i.e., the #tokens of the context of a newly added leaf)|`--chunk-size 100`|
|summary-len|False|100|length of a recursively generated summary (i.e., the #tokens of the context of an original SHT node)|`--summary-len 100`|
|node-embedding-model|False|"sbert"|the embedding model for SHT nodes (choices: "sbert", "dpr", "te3small")|`--node-embedding-model "sbert"`|
|query-embedding-model|False|"sbert"|the embedding model for the query (choices: "sbert", "dpr", "te3small")|`--query-embedding-model "sbert"`|
|summarization-model|False|"gpt-4o-mini"|the summarization model (choices: "gpt-4o-mini", "empty"(return an empty string as the summary))|`--summarization-model "gpt-4o-mini"`|
|distance-metric|False|"cosine"|he distance metric in the embedding space (choices: "cosine", "L1", "L2", "Linf")|`--distance-metric "cosine"`|
|context-hierarchy|False|True|whether to recover hierarchical information in the final context (choices: True, False)|`--context-hierarchy True`|
|embed-hierarchy|False|True|whether to embed the hierarchical information (choices: True, False)|`--embed-hierarchy True`|
|context-raw|False|True|whether to retrieve the newly added leaves (i.e., the chunks of the document) for the final context (choices: True, False)|`--context-raw True`|
|context-len|False|1000|the #tokens of the final context|`--context-len 1000`|

After running this command, you will find:
- an SHT for each queried pdf

    (Example: for [01262022-1835.pdf](./data/example/pdf/01262022-1835.pdf), [01262022-1835.json](./data/example/sbert.empty.c100.s100/sht/01262022-1835.json) is an SHT using "empty" as the summarization model, and [01262022-1835.json](./data/example/sbert.gpt-4o-mini.c100.s100/sht/01262022-1835.json) is another SHT using "sbert" as the summarization model)

- a visualized SHT for each generated SHT

    (Example: [01262022-1835.vis](./data/example/sbert.empty.c100.s100/sht_vis/01262022-1835.vis))

- an indexing that sorts the SHT nodes in ascending order of their distances to a query in the embedding space

    (Example: [index.jsonl](./data/example/sbert.empty.c100.s100/sbert.cosine.h1/index.jsonl))

- the retrieved context for each query

    (Example: [context.jsonl](./data/example/sbert.empty.c100.s100/sbert.cosine.h1/1000.l1.h1/context.jsonl))

You can then use these contexts to answer queries.

# Development
Structured-RAG is implemented in folder [structured_rag/](./structured_rag/). For implementation details , please read [structured_rag/README.md](./structured_rag/README.md). 

## Add a new embedding model
- In [SHTBuilder.py](./structured_rag/SHTBuilder.py)
    - import your new model from `.EmbeddingModels`
    - add your new model to `self.embedder` in `SHTBuilder.__init__`
- In [SHTIndexer.py](./structured_rag/SHTIndexer.py)
    - import your new model from `.EmbeddingModels`
    - add your new model to `self.embedder` in `SHTIndexer.__init__`
- In [StructuredRAG.py](./structured_rag/StructuredRAG.py)
    - Add your new model to `candid_embedding_models` in `Structured_RAG.build_sht`
- In [run_structured_rag.py](./run_structured_rag.py)
    - add your new model to the choices of cmd argument `--node-embedding-model` and/or `--query-embedding-model`


## Add a new summarization model
- In [SummarizationModels.py](./structured_rag/SummarizationModels.py)
    - define your new model as a derived class of `BaseSummarizationModel`
    - you can refer to `BaseGPTSummarizationModel`
- In [SHTBuilder.py](./structured_rag/SHTBuilder.py)
    - import your new model from `.SummarizationModels`
    - add your new model to `self.summarizer` in  `SHTBuilder.__init__`
- In [structured_rag.py](./run_structured_rag.py)
    - add your new model to the choices of cmd argument `--summarization-model`

# Data
Our experiment data is stored under `./data`. You can download it from [Google drive](https://drive.google.com/file/d/1xdIsK-seMbHapyHr2VTfzPGpkzz-vReE/view?usp=sharing) (9.14GB). The unzipped dataset is 39GB. 

## Shared data
```
data
└── <your-dataset>
    ├── pdf
    ├── heading_identification
    ├── node_clustering
    └── queries.json
```
- `pdf/` stored the queried files.
- `heading_identifiaction/` stores the VGT results for the pdfs.
- `node_clustering/` stores the clustered headings (e.g., SHT nodes) for the pdfs.
- `queries.json` stores all the queries.

## Baselines results
Folder `baselines/` stores the results for the baselines. 
```
data
└── <your-dataset>
    └── baselines
        ├── <node-embedding-model>.<summarization-model>.c<chunk-size>.s<summarization-len>
        │   └── <query-embedding-model>.<distance-metric>.raptor<is-raptor>
        │       ├── <context-len>.o<is-ordered>
        │       │   ├── context.jsonl
        │       │   ├── answer.jsonl
        │       │   ├── (qa_job.jsonl)
        │       │   ├── (qa_result.jsonl)
        │       │   ├── (rating.jsonl)
        │       │   ├── (rating_job.jsonl)
        │       │   └── (rating_result.jsonl)
        │       └── index.jsonl
        └── raptor_tree
```
- `raptor_tree/` stores the tree generated by raptor for the pdfs.
- `index.jsonl` stores the SHT nodes sorted in indexing order for the queries.
- `context.jsonl` stores the generated contexts for the queries.
- `answer.jsonl` stores the answers for the queries using the generated contexts.
- `rating.jsonl` stores the LLM's ratings for the generated answers. Only Qasper dataset has this file.
- `qa_job.jsonl`, `qa_result.jsonl`, `rating_job.jsonl`, and `rating_result.jsonl` are intermediate files for openai batch api. 

The configurations for the baselines can be indicated from the path to the results.

## Structured-RAG results
```
data
└── <your-dataset>
    └── <node-embedding-model>.<summarization-model>.c<chunk-size>.s<summarization-len>
        ├── <query-embedding-model>.<distance-metric>.h<embed-hierarchy>
        │   ├── <context-len>.l<context-raw>.h<context-hierarchy>
        │   │   ├── context.jsonl
        │   │   ├── answer.jsonl
        │   │   ├── (qa_job.jsonl)
        │   │   ├── (qa_result.jsonl)
        │   │   ├── (rating.jsonl)
        │   │   ├── (rating_job.jsonl)
        │   │   └── (rating_result.jsonl)
        │   └── index.jsonl
        ├── sht
        └── sht_vis
```
- `sht/` stores the SHTs for the queried pdfs.
- `sht_vis/` stores the visualized SHTs.

The configurations for Structured-RAG can be inferred from the path to the results.

## GROBID results
Folder `grobid/` stores the results for SHTs generated by [GROBID](https://github.com/allenai/s2orc-doc2json). 
```
data
└── <your-dataset>
    └── grobid
        ├── <node-embedding-model>.<summarization-model>.c<chunk-size>.s<summarization-len>
        │   └── <query-embedding-model>.<distance-metric>.h<embed-hierarchy>
        │       ├── <context-len>.l<context-raw>.h<context-hierarchy>
        │       │   ├── context.jsonl
        │       │   ├── answer.jsonl
        │       │   ├── (qa_job.jsonl)
        │       │   ├── (qa_result.jsonl)
        │       │   ├── (rating.jsonl)
        │       │   ├── (rating_job.jsonl)
        │       │   └── (rating_result.jsonl)
        │       └── index.jsonl
        ├── grobid
        └── node_clustering
```

- `grobid/` stores the immediate results returned by GROBID.
- `node_clustering/` stores the imtermediate results that help build SHTs from GROBID results.

## Intrinsic SHT results
Folder `intrinsic/` stores the results for the human-labeled intrinsic SHTs. 
```
data
└── <your-dataset>
    └── intrinsic
        ├── <node-embedding-model>.<summarization-model>.c<chunk-size>.s<summarization-len>
        │   └── <query-embedding-model>.<distance-metric>.raptor<is-raptor>
        │       ├── <context-len>.o<is-ordered>
        │       │   ├── context.jsonl
        │       │   ├── answer.jsonl
        │       │   ├── (qa_job.jsonl)
        │       │   ├── (qa_result.jsonl)
        │       │   ├── (rating.jsonl)
        │       │   ├── (rating_job.jsonl)
        │       │   └── (rating_result.jsonl)
        │       └── index.jsonl
        ├── heading_identification
        ├── human_label
        └── node_clustering
```
- `human_label/` stores the human labels.
- `heading_idenfication/` and `node_clustering/` are deducted from the human labels. 

# Reproduction
## Evaluation scripts
Folder: [eval/](./eval/)
- `eval_{dataset}.py` evaluate the *accuracy*, *f1*, or *ratings by LLM-as-a-judge* of the corresponding dataset.

## Data Generation scripts
- [run_structured_rag.py](./run_structured_rag.py) generates the Strcuctured-RAG results.
- [run_grobid.py](./run_grobid.py) generates the GROBID SHTs. Then you can use `StructuredRAG` to generate (similar to [run_structured_rag.py](./run_structured_rag.py)) their indexes and contexts.
- [run_raptor.py](./run_raptor.py) generates the RAPTOR trees and their indexes and contexts.
- [run_vanilla.py](./run_vanilla.py) generates the vanilla chunks and their indexes. Then you can use `StructuredRAG` to generate (similar to [run_structured_rag.py](./run_structured_rag.py)) their contexts.
- [run_bm25.py](./run_bm25.py) generates the results of Structured-RAG and baselines (vanilla & raptor) using BM25 as the embedder for nodes and queries. 

- [compare.py](./compare.py) calculates the number of $C$-templatizeda and well-formatted files (`calc()`). It also caculates the average precentage of SHT nodes that have robust hierarchical information (AVG(%hierarchy-robust nodes)), or that of SHT nodes that have hierarchical information exactly the same as the intrinsic ones (AVG(%hierarchy-intrinsic nodes)) (`count()`). 

## QA scripts
Folder: [batches/](./batches/)

We use openai batch api for question-answering and LLM-as-a-judge. 
- [batch.py](./batches/batch.py) generates a batch file storing a batch of qa/rating (llm-as-a-judge) tasks, uploads it to the server, and creates a corresponding batch job. You can also check the status of a batch, retrieve the result of the tasks from the server, and delete the batch job from the server.
- [llm_judge_prompt.txt](./batches/llm_judge_prompt.txt) stores the prompt template for LLM-as-a-judge to rate an answer compared with the gold answer.

# References
## Raptor
directory: [raptor](./raptor/)

This was copied from raptor's [codebase](https://github.com/parthsarthi03/raptor).