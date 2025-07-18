## Fun::CNNpredict ## Employ DNN model to operate prediction ##
import numpy as np
import pandas as pd
def CNNpredict(dat):
    distance = pd.read_csv("DIST.org.csv")
    group_mapping ={label:idx for idx, label in enumerate(distance['DISTANCE'])}

    data = pd.read_csv("HowellsMain_26Groups_LogTransf_mice.pmm.imp_82FTs.csv")
    #group_mapping ={label:idx for idx, label in enumerate(set(data['Population']))}
    data['Population']=data['Population'].map(group_mapping)

    #pred = pd.read_csv("Howells.log-scaled-mice.pmm.test.all.csv")
    pred=dat.iloc[:,1:]
    x_pred=pred.values
########

### build DNN (Deep neural network) based on hyperparameters search of kerastuner
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, InputLayer
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop
    from tensorflow.keras.regularizers import l2

# load the saved model
    from tensorflow.keras.models import load_model
    saved_model = load_model('./M12-94040_2.h5')
##############
    y_pred = saved_model.predict(x_pred)
    prediction=pd.DataFrame(y_pred,columns=[*group_mapping])
    prediction=prediction.round(3)
    def highlight_row(s):

        normed = (s - s.min()) / (s.max() - s.min())
    
        colors = ['rgba(250, 250, 240, 1)'.format(1 - n) if n < 0.5 
                  else 'rgba(106, 90, 205, 0.35)'.format(2 * (n - 0.5)) 
                  for n in normed]
    
        styles = []
        for i, color in enumerate(colors):
            if s[i] == s.max():
                styles.append('font-weight: bold; background-color: {}'.format(color))
            else:
                styles.append('background-color: {}'.format(color))
    
        return styles
    prediction.index = dat['ID'].to_list()
    tbl = prediction.style.apply(highlight_row, axis=1)
    tbl = tbl.set_table_styles([
        {'selector': 'td', 'props': [('border', '1px solid black')]},
        {'selector': 'th', 'props': [('border', '1px solid black')]},])
    tbl = tbl.format("{:.3f}")
    return tbl
