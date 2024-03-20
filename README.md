This repository can be used to extract ESG metrics from a company's annual report. 

### Installation

```
# Set up virtual environment
export ENVIRONMENT_NAME = <env-name>
conda create --name $ENVIRONMENT_NAME python=3.10 pip
conda activate $ENVIRONMENT_NAME

# Install package and dependencies 
pip install .
```

### Usage

To extract the ESG metrics from a company's annual report:

1. Save the company's annual report pdf to the `data/reports` directory.
2. Update the `esg_retriever.config.py` configuration file with an entry for the company. 
3. Use [Microsoft Azure's AI Document Intelligence]([url](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) to extract the text passages and tables from the pdf, run the `convert_report.py` script:
```
python azure_ai_doc_intelligence/convert_report.py --pdf_filename <file name of report>
```
This saves the data to a `json` file in `azure_ai_doc_intelligence/outputs`. 

4. Extract the ESG metrics from the report:
```
python esg_retriever/query.py --company <company> --year <year> --save_path <path to directory to save the output csv file>
```
This outputs a `.csv` file containing the extracted ESG metrics from the report. 

