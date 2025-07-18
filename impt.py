## Log-Transformed + Imputation + Standardization##
import numpy as np
import pandas as pd
def impt(dd):
    #dd = pd.read_csv("./Demo_data_for_Shiny-App_test_Howells.csv")
    NA_count = len(dd.isna().sum())
    imp_main = pd.read_csv("./Howells.imp.org.main.csv")
    main_mean = np.log(imp_main).mean()
    main_std = np.log(imp_main).std()
    if NA_count > 0:
        from fancyimpute import IterativeImputer
        dt_ms = [dd.iloc[:,1:],imp_main]
        msdt = pd.concat(dt_ms)
        mice_imputer = IterativeImputer(max_iter = 10000, random_state = 33617)
    # imputing the missing value with mice imputer
        df = mice_imputer.fit_transform(msdt)
        dt_imp = df[range(dd.shape[0]),]
        dt = pd.DataFrame(dt_imp, columns = imp_main.columns.tolist())
    else:
        dt = dd.iloc[:,1:]
    dat = (np.log(dt)-main_mean)/main_std
    dat.insert(0, 'ID', dd['ID'])
    return dat