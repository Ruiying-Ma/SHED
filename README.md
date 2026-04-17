# SHED

## Prerequisites

### Install Huridocs for document layout analysis
Please follow the [documentation](https://github.com/huridocs/pdf-document-layout-analysis) to install huridocs and then run
```
$ curl -X POST -F 'file=@path-to-your-pdf' localhost:5060
```
to analyze the layout of a document (e.g., identify headers).

### Install dependencies
```
$ conda create --file environment.yaml
```
Python version used in our experiment: 3.10.18

### Setup API credentials
Create a `.env` file under the root and add (we use Azure):
```
AZURE_ENDPOINT=xxx
AZURE_API_KEY=xxx
AZURE_API_VERSION=xxx
```


## SHT Inference: Local-First
Ensure that you have already identified the headers of a PDF (e.g., via [Huridocs](https://github.com/huridocs/pdf-document-layout-analysis)).


### Node Clustering
[Visual pattern extraction](structured_rag/FeatureExtractor.py): Extract visual patterns of the identified headers. A visual pattern includes:
    - font_size (rounded to 2)
    - font_name
    - font_color
    - is_all_cap (alphabetic characters only)
    - is_centered (|mid_bbox - mid_page| <= 2)
    - list_type
    - is_underlined

[Node clustering](structure_rag/ClusteringOracle.py): Cluster nodes based on their visual patterns.

[SHT assembly](structure_rag/SHTBuilder.py): Infer an SHT using local-first approach.


## Application: Agentic Document QA
Our [SHT-based agent](agents/react_agent_clean.py) is provided with an SHT in its user prompt, and uses `read_section` tool to retrieve relevant sections on demand.


## Datasets
The four datasets for evaluation are stored under [data/](data/) with the following structure:

```
data
└── civic                   <= dataset name
    ├── pdf                     <= the documents
    ├── heading_identification  <= results of layout analysis (e.g., via Huridocs)
    ├── node_clustering         <= node clusters
    └── queries.json            <= the QA set
```

## Results
Agentic document QA [results](agents/results/) store agent answers and LLM-as-judge results (for Finance and Papers).