## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Installation](#2-installation)
* [3. How to Use the Repository](#3-how-to-use-the-repository)
* [4. Data Sources](#4-data-sources)
* [Project Tree](#project-tree)

## 1. Introduction

This repository is a prototype of the [MEDomicsLab](https://github.com/MEDomics-UdeS/MEDomicsLab) project. It aims to
illustrate the functionalities of the three main modules of the future application:

- Extraction
- Input
- Learning

The demonstration is based on the [MIMIC](https://mimic.mit.edu/) dataset.

## 2. Installation

This repository requires *Python 3.8* or higher to run.

Install the requirements as follows:

```
pip install -r requirements.txt
```


Additionally, you need to add the data locally.

Copy the following files into *csv/original_data*:

- *admissions.csv*
- *chartevents.csv*
- *d_items.csv*
- *d_labitems.csv*
- *labevents.csv*
- *patients.csv*
- *procedureevents.csv*
- *radiology.csv*

To generate note embeddings, you also need to add the pre-trained biomedical language representation model called BioBERT to the project. Obtain the BioBERT sources from the following [link](https://github.com/EmilyAlsentzer/clinicalBERT). Add the *pretrained_bert_tf* folder under *extraction_models/*. The folder must contain at least a subfolder *biobert_pretrain_output_all_notes_150000*. Make sure the config *json* file in this subfolder is named *config.json*.

## 3. How to Use the Repository

Once you have completed all the requirements mentioned in the previous section, run the following notebooks in order:

1. The **Extraction.ipynb** notebook illustrates the functionalities of the MEDomicsLab Extraction module by generating embeddings from time series events and text notes. At the end of the execution, the notebook will produce the following CSV files from the extracted embeddings in the *csv/extracted_features/* folder:

   - *chart_events.csv*
   - *lab_events.csv*
   - *procedure_events.csv*
   - *rad_notes.csv*
   

2. The **Input.ipynb** notebook illustrates the functionalities of the MEDomicsLab Input module by performing basic operations on CSV tables from the *csv/original_data* and *csv/extracted_features* folders to create machine learning tables for the Learning module. One objective of this notebook is to demonstrate the use of the [MEDprofiles](https://github.com/MEDomics-UdeS/MEDprofiles/tree/main) package. For now, the functionalities of this package are only reproduced, but we are currently working on integrating the real package into our prototype. At the end of the execution, the following CSV files will be created in the *csv/static* folder:

   - *holdout_time_point_1.csv*
   - *holdout_time_point_2.csv*
   - *holdout_time_point_3.csv*
   - *holdout_time_point_4.csv*
   - *train_time_point_1.csv*
   - *train_time_point_2.csv*
   - *train_time_point_3.csv*
   - *train_time_point_4.csv*
   - *pred_holdout.csv*
   - *pred_train.csv*
   

3. The **MachineLearning.ipynb** notebook illustrates the functionalities of the MEDomicsLab Learning module by showing 
   a pipeline of machine learning experiments using the [PyCaret](https://pycaret.gitbook.io/docs/) library. It uses the 
   previously created CSV tables in the *csv/static/* folder.

## 4. Data Sources

The dataset used in this repository is confidential data and is available on [PhysioNet](https://physionet.org/content/mimiciv/2.2/).

## Project Tree

```
|--- csv                         <- Will contain all the CSV files generated during the execution of the prototype
|     |--- extracted_features    <- Will contain the CSV files generated by the Extraction notebook
|     |--- original_data         <- Must contain the original CSV files mentioned in Installation before any execution
|     |--- static                <- Will contain the static CSV files generated by the Input notebook
|
|--- extraction_models           <- Must contain models mentioned in Installation for the Extraction
|
|--- src                         <- Contains all the utility functions for the three notebooks
|     |--- extraction.py         <- Utility functions for the Extraction notebook
|     |--- input.py              <- Utility functions for the Input notebook
|     |--- machine_learning.py   <- Utility functions for the MachineLearning notebook
|
|--- Extraction.ipynb            <- Demonstrates functionalities of the Extraction module, must be executed before the other notebooks
|--- Input.ipynb                 <- Demonstrates functionalities of the Input module, must be executed after the Extraction notebook and before the MachineLearning notebook
|--- MachineLearning.ipynb       <- Demonstrates functionalities of the Learning module, must be executed after the other notebooks
```