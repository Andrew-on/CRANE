## Fun::plot2D##
def plot2D(layer_3):
    import umap.umap_ as umap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.io as pio
##### 2D UMAP Transformed data
    dt = pd.read_csv("HowellsMain_26Groups_raw_mice.pmm.imp_82FTs(umap).csv")
    x_2d_dnn = umap.UMAP(n_components=2, n_neighbors=8,random_state=1247828099).fit_transform(layer_3)
    ump_2d_dnn = pd.DataFrame(x_2d_dnn,columns=["UMAP_X","UMAP_Y"])
    ump_2d_dnn['26_Subpops']=dt.loc[:,['Population']]
    ump_2d_dnn['Pop']=ump_2d_dnn['26_Subpops']
    ump_2d_dnn['6_Ancestral_Groups']=dt.iloc[:,3]
    ump_2d_dnn.fillna('Your_Case', inplace=True)
    ump_2d_dnn['Marker_Size'] = ump_2d_dnn['26_Subpops'].apply(lambda x: 200 if x == 'Your_Case' else 18)
    ump_2d_dnn['Group']=ump_2d_dnn['6_Ancestral_Groups'].replace({
    'East Asian':' ', 'Native American':'  ', 'Australo-Melanesian':'   ', 'European':'    ',
    'African':'     ', 'Polynesian-Micronesian':'      ', 'Your_Case':'       '})
    fig2d1 = px.scatter(ump_2d_dnn, x="UMAP_X", y="UMAP_Y", size='Marker_Size',
                 symbol_sequence=['circle','cross','diamond','star','x','star-triangle-down','square'],
                 symbol="Group",color="Pop",width=1100, height=760,#title= '2D UMAP Plot(DNN layer_3 data 26 Groups)',
                 color_discrete_sequence=px.colors.qualitative.Light24)#Alphabet)
    fig2d1.update_traces(marker=dict(line=dict(width=0.6,color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig2d1.update_layout(legend= {'itemsizing': 'constant'},font=dict(size= 14), legend_title = "26_Populations", legend_traceorder="reversed")
    #fig2d1.show()
    return fig2d1
