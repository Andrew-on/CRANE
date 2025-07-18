## Fun::plot3D ## 
def plot3D(layer_3):
    import umap.umap_ as umap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.io as pio

##### 3D UMAP Transformed data
    dt = pd.read_csv("HowellsMain_26Groups_raw_mice.pmm.imp_82FTs(umap).csv")
    x_3d_dnn = umap.UMAP(n_components=3, n_neighbors=8,random_state=124782800).fit_transform(layer_3)
    ump_3d_dnn = pd.DataFrame(x_3d_dnn,columns=["UMAP_X","UMAP_Y","UMAP_Z"])
    ump_3d_dnn['26_Subpops']=dt.loc[:,['Population']]
    ump_3d_dnn['Pop']=ump_3d_dnn['26_Subpops']
    ump_3d_dnn['6_Groups']=dt.iloc[:,3]
    ump_3d_dnn.fillna('Your_Case', inplace=True)
    ump_3d_dnn['Marker_Size'] = ump_3d_dnn['26_Subpops'].apply(lambda x: 200 if x == 'Your_Case' else 16)
    ump_3d_dnn['Group']=ump_3d_dnn['6_Groups'].replace({
    'East Asian':' ', 'Native American':'  ', 'Australo-Melanesian':'   ', 'European':'    ',
    'African':'     ', 'Polynesian-Micronesian':'      ', 'Your_Case':'       '})
    fig3d2 = px.scatter_3d(ump_3d_dnn, x='UMAP_X', y='UMAP_Y', z='UMAP_Z',symbol="Group",color="Pop",
                     symbol_sequence=['x','circle','square-open','diamond','circle-open','cross','square'],
                     opacity=1,width=1200, height=800,size='Marker_Size',#title= '3D UMAP plot (DNN layer_3 data 26 Groups)',
                     color_discrete_sequence=px.colors.qualitative.Light24)#px.colors.qualitative.Dark24)
    fig3d2.update_traces(marker=dict(line=dict(width=0.0,color='DarkSlateGrey')))
    fig3d2.update_layout(legend= {'itemsizing': 'constant'}, legend_title = "26_Populations", legend_traceorder="reversed")
    #fig3d2.show()
    return fig3d2
