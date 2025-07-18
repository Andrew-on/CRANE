### plotCM.py ###
def plotCM(dd):
    import numpy as np
    import pandas as pd
    #dd=pd.read_csv("./Demo_test_Shiny_Howells.csv")
    main_dt = pd.read_csv("./HowellsMain_26Groups_raw_mice.pmm.imp_82FTs(umap).csv")
    main_mean = np.log(main_dt.iloc[:,4:]).mean()
    main_std = np.log(main_dt.iloc[:,4:]).std()
    missing_pattern=dd.isna()

    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=2932513759)
    #k=3
    for fold, (train_index, test_index) in enumerate(kfold.split(main_dt.iloc[:,4:], main_dt.iloc[:,2:3]),1):
        if fold==3:
            test_idx = test_index

    i=0
    pattern = missing_pattern.iloc[i].values
    simu_miss_dt = main_dt.copy()
    for col_idx, is_missing in enumerate(pattern):
        if is_missing:
            simu_miss_dt.iloc[:, col_idx] = np.nan

    from fancyimpute import IterativeImputer
    from joblib import Parallel, delayed
    import pandas as pd
    # tqdm patch for joblib
    from joblib import parallel_backend
    imp_dt=simu_miss_dt.copy()
# 封装单个插补操作
    def impute_one(index):
        complete_msmain_dt = main_dt.copy()
        complete_msmain_dt.iloc[index, 4:] = simu_miss_dt.iloc[index, 4:]
        mice_imputer = IterativeImputer(max_iter=100, random_state=33612, tol=0.1)
        imp_miss_dt = mice_imputer.fit_transform(complete_msmain_dt.iloc[:, 4:])
        return pd.Series(imp_miss_dt[index, :], index=simu_miss_dt.columns[4:])
# 使用 tqdm 包装迭代器
    imp_dt = simu_miss_dt.copy()
    results = Parallel(n_jobs=-1)(delayed(impute_one)(idx) for idx in test_idx)
# 更新结果
    for i, idx in enumerate(test_idx):
        imp_dt.iloc[idx, 4:] = results[i]
   
    imp_dt.iloc[:,4:] = (np.log(imp_dt.iloc[:,4:])-main_mean)/main_std    
    imp_dt.iloc[:,1:3]=main_dt.iloc[:,1:3]
    
### part4 #
    distance = pd.read_csv("DIST.org.csv")
    group_mapping ={label:idx for idx, label in enumerate(distance['DISTANCE'])}

    data=imp_dt.copy()
    data['Population']=data['Population'].map(group_mapping)
    data_x=data.iloc[:,4:]
    data_y=data.loc[:,['Population']]
    x=data_x.values
    y_array=data_y.values
    y_size = y_array.size
    data_y_array=y_array.reshape(y_size)

    from tensorflow.keras.utils import to_categorical
    y = to_categorical(data_y_array)

    ### build DNN (dense neural network) based on hyperparameters search of kerastuner
    from tensorflow import keras
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict
    kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=2932513759)
    #k=3
    for fold, (train_index, test_index) in enumerate(kfold.split(x, data_y_array),1):
        if fold==3:
         train_x,test_x = x[train_index], x[test_index]
         train_y,test_y = y[train_index], y[test_index]
         from tensorflow.keras.models import load_model
         saved_model = load_model('./M12-94040_2.h5')
         y_pred = saved_model.predict(test_x)
         from sklearn.metrics import confusion_matrix
         cm=confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))
    import seaborn as sn
    import matplotlib.pyplot as plt
    df_cm0 = pd.DataFrame(cm, index = [*group_mapping],columns =[*group_mapping])
    cm0=pd.DataFrame(df_cm0).apply(lambda x:x/x.sum(), axis=1)
    plt.figure(figsize = (45,30))
    #plt.title("(Probs)Confusion Matrix: Howells' Skull DMLPNN Classification Model (31 Populations)", size=40)
    plt.title("X:Predictive Labels(Horizontal) Y:Actual Labels(Vertical)", size=15)
    plt.xlabel("Actual Labels")
    plt.ylabel("Predicted Labels")
    plt.yticks(size=13)
    sn.set(font_scale=0.9)
    cmplot=sn.heatmap(cm0, annot=True,linewidths=0.01,linecolor='grey',cbar=False)
    cmplot.set_xticklabels(cmplot.get_xticklabels(), rotation=27, horizontalalignment='right',size=13)
    return cmplot
