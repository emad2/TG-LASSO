import argparse
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error

# CSV files names.
#TRAIN_GENE_EXPRESSION = 'gdsc_expr_postCB.csv'
#TEST_GENE_EXPRESSION = 'tcga_expr_postCB.csv'
#TRAIN_DRUG_RESPONSE = 'gdsc_dr.csv'
#TEST_DRUG_RESPONSE = 'tcga_dr.csv'
#
#GDSC_TISSUE_FNAME = 'gdsc_tissue_by_sample_2.csv'
#TCGA_TISSUE_FNAME = 'tcga_tissue_by_sample2.csv'
#
#OUTPUT_FNAME = 'GDSC_TCGA_lasso_tissue.csv'


def parse_args():
    """
    Parse the arguments.
    Parse the command line arguments/options using the argparse module
    and return the parsed arguments (as an argparse.Namespace object,
    as returned by argparse.parse_args()).
    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for running TG_LASSO')
    parser.add_argument('-id', '--input_directory', default='./Data', help = 'Address of the directory containing the input files')
    parser.add_argument('-od', '--output_directory', default = './Results', help = 'output directory adddress')
    parser.add_argument('-trg','--train_gene_expression', type = str, help = 'name of the input gene expression file from GDSC', default = 'gdsc_expr_postCB.csv')
    parser.add_argument('-teg','--test_gene_expression',type = str,
                        help = 'name of the gene expression file from TCGA to be tested',default = 'tcga_expr_postCB.csv')
    parser.add_argument('-trd','--train_drug_response', type = str,
                        help = 'name of the input drug response file from GDSC',default = 'gdsc_dr.csv')
    parser.add_argument('-ted','--test_drug_response', type = str,
                        help = 'name of the drug response file from TCGA to be tested',default = 'tcga_dr.csv')
    parser.add_argument('-gt','--gdsc_tissue', type = str,
                        help = 'name of the input gdsc tissue by sample file',default ='gdsc_tissue_by_sample_2.csv')
    parser.add_argument('-tt','--tcga_tissue', type = str,
                        help = 'name of the input tcga tissue by sample file',default ='tcga_tissue_by_sample2.csv') 
    parser.add_argument('-of','--output_file', type = str,
                        help = 'name of the output file',default = 'GDSC_TCGA_lasso_tissue.csv')
    
    args = parser.parse_args()
    return args

def get_drug_lasso_alpha(train_expr, train_resp):
    """
    Trains a lasso CV for each drug, and returns dictionary.
    Key: drug name -> str
    Value: alpha -> float
    """

    drug_alpha_dct = {}
    

    X_train = train_expr.values.T
    # Train a lasso CV for each drug.
    with warnings.catch_warnings():
        for drug in train_resp.index.values:
            y_train_tmp = train_resp.loc[drug].values
            not_nan_ind = ~np.isnan(y_train_tmp)
            y_train_tmp = y_train_tmp[not_nan_ind]
            X_train_tmp = X_train[not_nan_ind,:]
            
            warnings.filterwarnings("ignore") 
            reg_model = LassoCV(n_jobs=-1)
            
            reg_model.fit(X_train_tmp, y_train_tmp)
            
            drug_alpha_dct[drug] = reg_model.alpha_
        
    return drug_alpha_dct

def fit_model(*args):

    X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp, alpha = args[0]
    reg_model = Lasso(alpha=alpha, random_state=0)
    reg_model.fit(X_train_tmp, y_train_tmp)
    
    y_pred = reg_model.predict(X_test_tmp)
    rmse = mean_squared_error(y_test_tmp, y_pred)
    return rmse

def tune_alpha(train_expr_non_tissue, train_resp_non_tissue, train_expr_tissue,
    train_resp_tissue, all_tissue_drug_alpha_dct):
    """
    Given the training expression and training response as well as a tissue
    type, find the alpha that trains on all other tissue types and predicts on
    the given tissue type.
    """

    drug_alpha_dct = {}

    X_train = train_expr_non_tissue.values.T
    X_test = train_expr_tissue.values.T
    # Get the best alpha for each drug.
    for drug in train_resp_non_tissue.index.values:
        # Get the training set.
        y_train_tmp = train_resp_non_tissue.loc[drug].values
        not_nan_ind = ~np.isnan(y_train_tmp)
        y_train_tmp = y_train_tmp[not_nan_ind]
        X_train_tmp = X_train[not_nan_ind,:]
        
        # Get the validation set.
        y_test_tmp = train_resp_tissue.loc[drug].values
        not_nan_ind = ~np.isnan(y_test_tmp)
        y_test_tmp = y_test_tmp[not_nan_ind]
        X_test_tmp = X_test[not_nan_ind,:]
        
        # If there are no non-nan samples for this tissue in GDDSC, then use the
        # alpha from training on all of GDSC.
        if X_test_tmp.shape[0] == 0:
            drug_alpha_dct[drug] = all_tissue_drug_alpha_dct[drug]
            continue
            
        # Initialize the alpha dictionary for the current drug.
        best_alpha_dct = {}
        pool = Pool(processes=24)
        alpha_space = np.logspace(-2, -1, 100)
        # Check which alpha yields the highest performance.
        rmse_lst = pool.map(fit_model, zip(itertools.repeat(X_train_tmp),
            itertools.repeat(y_train_tmp), itertools.repeat(X_test_tmp),
            itertools.repeat(y_test_tmp), alpha_space))
        for i, e in enumerate(alpha_space):
            best_alpha_dct[e] = rmse_lst[i]
        pool.close()
        pool.join()
        # Get the alpha corresponding to the lowest rmse.
        drug_alpha_dct[drug] = min(best_alpha_dct, key=best_alpha_dct.get)
    return drug_alpha_dct

def main():
           
    args = parse_args()
    
    train_gene_expression = os.path.join(args.input_directory,args.train_gene_expression)
    address_out_dir = args.output_directory
    if not os.path.exists(address_out_dir):
        os.makedirs(address_out_dir)
    test_gene_expression = os.path.join(args.input_directory,args.test_gene_expression)
    train_drug_response =  os.path.join(args.input_directory,args.train_drug_response)
    test_drug_response =  os.path.join(args.input_directory,args.test_drug_response)
    gdsc_tissue =  os.path.join(args.input_directory,args.gdsc_tissue)
    tcga_tissue =  os.path.join(args.input_directory,args.tcga_tissue)
    
    output_file = os.path.join(args.output_directory,args.output_file)
    print('Importing done successfully')

    # Read CSV files.
    train_expr = pd.read_csv(train_gene_expression, index_col=0)
    test_expr = pd.read_csv(test_gene_expression, index_col=0)
    train_resp = pd.read_csv(train_drug_response, index_col=0)
    test_resp = pd.read_csv(test_drug_response, index_col=0)
    
    # Get binary matrices indicating sample-tissue membership.
    gdsc_tissue = pd.read_csv(gdsc_tissue, index_col=0)
    tcga_tissue = pd.read_csv(tcga_tissue, index_col=0)

    all_tissue_drug_alpha_dct = get_drug_lasso_alpha(train_expr, train_resp)
    

    # This is the drug response dictionary. Keys are (drug, sample) pairs.
    predicted_dr_dct = {}
    
    
    # Loop through the TCGA tissue types. We predict on all of them.
    for tissue in tcga_tissue.index.values:
        # If tissue in GDSC, train on GDSC samples in tissue.
        if tissue in gdsc_tissue.index.values:
            gdsc_tissue_samples = gdsc_tissue.loc[tissue]
            gdsc_tissue_samples = gdsc_tissue_samples.iloc[gdsc_tissue_samples.to_numpy().nonzero()].index.values
            # Get the samples corresponding to the tissue.
            train_expr_tissue = train_expr[gdsc_tissue_samples]
            train_resp_tissue = train_resp[gdsc_tissue_samples]
            # Get all samples excluding current tissue.
            train_expr_non_tissue = train_expr.drop(gdsc_tissue_samples, axis=1)
            train_resp_non_tissue = train_resp.drop(gdsc_tissue_samples, axis=1)
            # Find the best alpha with the tissue samples as the validation set.
            drug_alpha_dct = tune_alpha(train_expr_non_tissue, train_resp_non_tissue,
                train_expr_tissue, train_resp_tissue, all_tissue_drug_alpha_dct)
        # Otherwise, train on all GDSC samples.
        else:
            # This dictionary is trained from lasso_tissue.py
            drug_alpha_dct = all_tissue_drug_alpha_dct
        # Train on all GDSC samples rather than just the tissue.
        train_expr_tissue = train_expr
        train_resp_tissue = train_resp
        
        # Do the same for TCGA.
        tcga_tissue_samples = tcga_tissue.loc[tissue]
        tcga_tissue_samples = tcga_tissue_samples.iloc[tcga_tissue_samples.to_numpy().nonzero()].index.values
       
        test_expr_tissue = test_expr[tcga_tissue_samples]
        test_resp_tissue = test_resp[tcga_tissue_samples]
        
        # Go through the drug names.
        for drug in list(train_resp.index):
            print("Predicting the response of '%s' tissue for '%s' drug"%(tissue, drug))
            y_train_tmp = train_resp_tissue.loc[drug].values
            not_nan_ind = ~np.isnan(y_train_tmp)
            y_train_tmp = y_train_tmp[not_nan_ind]
            X_train_tmp = train_expr_tissue.values.T[not_nan_ind,:]
            
            # Remake the training set to the entire set if the tissue has no non-nan values.
            if X_train_tmp.shape[0] == 0:
                y_train_tmp = train_resp.loc[drug].values
                not_nan_ind = ~np.isnan(y_train_tmp)
                y_train_tmp = y_train_tmp[not_nan_ind]
                X_train_tmp = train_expr.values.T[not_nan_ind,:]
                
             
            # Use the alpha learned from training on all samples.
            clf = Lasso(alpha=drug_alpha_dct[drug], random_state=0)
            clf.fit(X_train_tmp, y_train_tmp)
            
            # Write out the tissue lasso weights.
            with open('%s_%s_lasso_weights' % (tissue, drug), 'wb') as fp:
                pickle.dump(clf.coef_, fp)
            
            # Predict on the test values.    
            y_test_hat_tmp = clf.predict(test_expr_tissue.values.T)
            for i, ic50 in enumerate(y_test_hat_tmp):
                predicted_dr_dct[(drug, tcga_tissue_samples[i])] = ic50
            

    # Unstack the dictionary into a dataframe.
    dr_matrix = pd.Series(predicted_dr_dct).unstack()
    dr_matrix = dr_matrix[test_resp.columns]
    # Write out the results.
    dr_matrix.to_csv(output_file) 

if __name__ == '__main__':
    main()