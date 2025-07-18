## Fun::plotDT ### Data Visualization ##
def plotDT(dat):
    import os
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
### 31_Grps_plus Classifier Visualization
    import numpy as np
    import pandas as pd
    distance = pd.read_csv("DIST.org.csv")
    group_mapping ={label:idx for idx, label in enumerate(distance['DISTANCE'])}
    data = pd.read_csv("HowellsMain_26Groups_LogTransf_mice.pmm.imp_82FTs.csv")
#group_mapping ={label:idx for idx, label in enumerate(set(data['Population']))}
    data['Population']=data['Population'].map(group_mapping)
    data_x=data.iloc[:,4:]
#data_y=data.loc[:,['Population']]
    data_x_array=data_x.values

#import random
#randsd=random.randint(0,2**32)
    from sklearn import preprocessing
#data_x_array_nor_standardized = preprocessing.scale(data_x_array)
    main_x = preprocessing.scale(data_x_array)
    dat_new = dat.iloc[:,1:]
    data_x_array_nor_standardized = np.vstack((main_x, dat_new.to_numpy()))

    from keras import layers
    from tensorflow.keras.models import load_model
    saved_model = load_model('M12-94040_2.h5')
##############
#y_pred = saved_model.predict(data_x_array_nor_standardized)
#prediction=pd.DataFrame(y_pred,columns=[*group_mapping])

    from keras import backend as K
    layer_output=K.function([saved_model.layers[0].input],[saved_model.layers[3].output])
    output=layer_output([data_x_array_nor_standardized])
    None
##array = np.array(layer_output([data_x_array_nor_standardized]))
##array.shape

    out=np.array(output)
    layer_3=pd.DataFrame(out[0,:,:])
    return layer_3
