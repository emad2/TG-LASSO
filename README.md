# *TG-LASSO* - Tissue Guided LASSO
#### Amin Emad (email: amin (dot) emad (at) mcgill (dot) ca)
#### McGill University

# Motivation
Cancer is one of the leading causes of death globally and is expected to be the most important obstacle in increasing the life expectancy in the 21st century. Prediction of clinical drug response (CDR) of cancer patients based on their clinical and molecular profiles, before the administration of the drug, can play a significant role in individualized medicine. Machine learning models have the potential to address this issue, yet they require a large number of labeled samples for training. While large databases of drug response and molecular profiles of preclinical in-vitro cancer cell lines (CCLs) exist for a large number of drugs, it is unclear whether preclinical samples can be used to predict CDR of real patients. In this study, we first designed a systematic approach to evaluate the performance of different algorithms trained on gene expression and IC50 values of CCLs to predict CDR of patients. This framework is depicted in the figure below. Using data from two large databases, we evaluated various linear and non-linear algorithms, some of which utilized information on gene interactions. Our results showed that regularized regression methods provide a relatively good prediction. 

Then, we developed a new algorithm called TG-LASSO (Tissue-Guided LASSO) that explicitly integrates information on samples’ tissue of origin with samples gene expression profiles to improve prediction performance (see figure below). The idea behind this approach is to use all CCLs originating from different tissue types in training the LASSO model, while choose the hyperparameter of the LASSO model, α, in a tissue-specific manner. Since α controls the number of features (i.e. genes) used by the LASSO model, this approach allows us to optimally select the number of predictive genes for each tissue type, yet use all CCLs to train these tissue-specific regression models. This approach resulted in the best performance among all the methods tested in our study. 
 
![Method Overview](/image_pipeline.png)



# Requirements

In order to run the code, you need to have Python 3.7 installed. In addition, the code uses the following python modules/libraries which need to be installed:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Sklearn (scikit-learn)](http://scikit-learn.org/stable/)

Instead of installing all these libraries independently, you can use prebulit Python distributions such as [Anaconda](https://www.continuum.io/downloads), which provides a free academic subscription.

# Input Files
### Description of required inputs:

#### Gene expression files :
These are genes x samples csv files where the first column contains name of genes and the first row contains name/IDs of the samples. The training gene expression file is obtained from GDSC and the test file is obtained from TGCA. 
The files are homogenized using ComBat for batch effect removal to homogenize the gene expression data from GDSC (microarray) and TGCA (RNA-seq)
As input examples, we have have provided these files [here](https://www.dropbox.com/s/vwhtcf7rdko26tw/TG_LASSO_GeneExpressionInput.zip?dl=0). 

Example of Gene expression file - 

|  | sample_1 | sample_2 | sample_3 |
| :--- | :--- | :--- | :--- |
| G1 | 2.24 | 7.67 | 2.12 |  
| G2 | 8.34 | 9.34 | 8.45 |
| G3 | 4.51 | 2.05 | 2.22 |
| G4 | 3.03 | 1.55 | 2.15 |
| G5 | 6.23 | 2.23 | 6.55 |
| G6 | 5.94 | 8.33 | 4.12 |

#### Drug Response files:
These are drugs x samples csv files where first column contains name of drugs and the first row contains name/IDs of the samples. The training drug response file is obtained from GDSC and the test file is obtained from TGCA. 

Example of Drug Response file-

|  | sample_1 | sample_2 | sample_3 |
| :--- | :--- | :--- | :--- |
| bicalutamide | 4.04 | 4.21 | 4.04 |  
| bleomycin | 3.22 | -1.21 | -0.25 |
| cetuximab | 6.89 | 5.29 | 5.80 |
| cisplatin | 2.10 | 0.79 | 2.76 |
| dabrafenib | 1.95 | 4.04 | 2.76 |
| docetaxel | -3.27 | -6.42 | -5.50 |

#### Origin Tissue files:
These are tissue x samples csv files where first column contains name of Tissues and the first row contains name/IDs of the samples. It introduces a binary feature to each sample, representing whether the sample belongs to that tissue ('1') or not ('0').

Example of Tissue file-

|  | sample_1 | sample_2 | sample_3 |
| :--- | :--- | :--- | :--- |
| B_cell_lymphoma |0 | 0 | 1 |  
| Bladder | 0 | 0 | 0 |
| Blood | 0 | 1 | 0 |
| Brain | 1 | 0 | 0 |


# Running TG-LASSO
To run the program place the input files in one folder and then specify the following arguments:
- 'input_directory' : Address of the directory containing the input files (short: '-id') eg. "./Data"
- 'train_gene_expression' : Name of the input gene expression file from GDSC (short: '-trg') eg. "gdsc_expr_postCB.csv"
- 'test_gene_expression' : Name of the gene expression file from TCGA to be tested (short: '-teg') eg. "tcga_expr_postCB.csv"
- 'train_drug_response' : Name of the input drug response file from GDSC (short: '-trd') eg. "gdsc_dr.csv"
- 'test_drug_response' : Name of the drug response file from TCGA to be tested (short: '-ted') eg. "tcga_dr.csv"
- 'gdsc_tissue' : Name of the input gdsc tissue by sample file (short: '-gt') eg. "gdsc_tissue_by_sample_2.csv"
- 'tcga_tissue' : Name of the input tcga tissue by sample file (short: '-tt') eg. "tcga_tissue_by_sample2.csv "



```
python3 lasso_script.py --input_directory ./Data --train_gene_expression gdsc_expr_postCB.csv --test_gene_expression tcga_expr_postCB.csv --train_drug_response gdsc_dr.csv --test_drug_response tcga_dr.csv --gdsc_tissue gdsc_tissue_by_sample_2.csv --tcga_tissue tcga_tissue_by_sample2.csv 
```

By default, TG-LASSO writes the intermediate outputs generated in this step into a directory called "Results" in the current directory by the name GDSC_TCGA_lasso_tissue.csv. To change its location, see below:

- 'output_directory' : Name of the directory where the result would be saved (short: '-od')
- 'output_file' : Name of the output file generated (short: '-of')

