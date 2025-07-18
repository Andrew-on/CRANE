## Customized Functions ##
from impt import impt
from plotCM import plotCM
from CNNpredict import CNNpredict
from plotDT import plotDT
from plot2D import plot2D
from plot3D import plot3D

from shiny import App, ui, render, reactive
from shinywidgets import render_plotly, output_widget, render_widget 
import pandas as pd
import time
#import asyncio

ui.tags.style(
        '''
        .navbar {z-index: 1050; min-height: 10px;}
        .navset-pills {margin-top: 10px; }
        .nav-panel {z-index: 1;}
        '''
   )

app_ui = ui.page_navbar(

    #ui.nav_spacer(), 
    # Panel_1 App
    ui.nav_panel("CRANE", 
    ui.tags.style(
    """
      /* 1.最小宽度保证文字能放下 */
      .shiny-file-input-progress {
        display: inline-block !important;
        min-width: 1000px !important;  /* 根据“Upload complete”长度调 */
        height: auto !important;
        padding: 1px !important;      /* 给点内边距 */
      }
      /* 2.进度条本身也拉高 */
      .shiny-file-input-progress .progress {
        width: 100% !important;
        height: 20px !important;      /* 进度条条身高度 */
      }
      /* 3.文字强制单行不折、全显 */
      .shiny-file-input-progress .progress-text {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
      }
        /* Remove borders and shadows from accordion panels */
        .accordion-item {
            border: none !important;
            box-shadow: none !important;
        }

        .accordion-button {
            border: none !important;
            box-shadow: none !important;
        }

        .accordion-body {
            border-top: none !important;
        }
    """
    ),
        ui.accordion(
            ui.accordion_panel("Data Uploading", 
                ui.busy_indicators.options(spinner_type="bars3", spinner_size='500%'), ui.busy_indicators.use(spinners=True, pulse=False),
                ui.download_button("download_example", "Download Example Data File"),
                ui.input_file("file1", "Please upload your *.csv data file", accept=[".csv"], width=['1000px'],multiple=False), 
                ui.input_action_button("btn", "Run CRANE!", disabled=True), value='s0'
                #ui.output_ui("step0_progress"),
                ),                 
            ui.accordion_panel("Step 1: Population Affinity Estimation", ui.output_table("pred_table"),value='s1'),
            ui.accordion_panel("Step 2: 2D UMAP Plot", output_widget("plot_2d"),value='s2'),
            ui.accordion_panel("Step 3: 3D UMAP Plot", output_widget("plot_3d"),value='s3'),
            ui.accordion_panel("Step 4: Confusion Matrix of Missingness Simulations",ui.output_plot("plot_confusion_matrix",width='1340px',height='680px'),value='s4'),
            ui.accordion_panel("Note: To initiate a new session, please refresh the page (Press 'Ctrl' + 'F5' on your keyboard)",value='s5'),
            id="tab", multiple=True, open="s0",
        )
    ),
    # Panel_2 Intro
    ui.nav_panel("Introduction", 
    ui.div(
        ui.br(),ui.br(),
        ui.div(""), 
        ui.markdown(
        """ 
        ### **CRANE** (***CR***aniometric ***AN***cestry ***E***stimator) 
        ##### A Deep Learning-Powered Estimator for Population Ancestry Affiliation based on Craniometric Features        
        ```python
        
        Welcome to CRANE (CRaniometric ANcestry Estimator), a deep learning model to provide fine-grained estimations of ancestry using craniometric data. 
        
        The goal of CRANE is to enhance accuracy (with a cross-validation accuracy of 94%) and utility of craniometric ancestry estimation. 
        
        By providing more precise and reliable information, CRANE seeks to be an invaluable asset in forensic practice, 
        as well as other anthropological research. 
        
        The model was trained using Howells' craniometric data and aims to estimate a user-provided cranium's affinity 
        to each of the 26 worldwide ancestry populations (see a summary of the populations below). 
        
        Designed as a user-friendly platform for researchers and forensic anthropologists, CRANE offers several key advantages: 
        it provides probabilistic estimates of affinities for each ancestry population, capably handles incomplete craniometric 
        data (missing measurements), and includes an estimation of the reliability or uncertainty associated with each prediction.

        ```  
        **Summary of Population**        
        |Population|Male&nbsp;|&nbsp;Female|
        |-----------|----------|----------|
        |Europe|||
        |&nbsp;&nbsp;Norse of medieval Oslo|&nbsp;&nbsp;55 &nbsp;|&nbsp; 55 &nbsp;|
        |&nbsp;&nbsp;Zalavár, Hungary, cemeteries of 9th to 11th centuries A.D.|&nbsp;&nbsp;53 &nbsp;|&nbsp; 45 &nbsp;|
        |&nbsp;&nbsp;Berg, mountain village in Carinthia, Austria|&nbsp;&nbsp;56 &nbsp;|&nbsp; 53 &nbsp;|
        |Sub-Saharan Africa|||
        |&nbsp;&nbsp;Taita, tribe of Kenya, East Africa|&nbsp;&nbsp;33 &nbsp;|&nbsp; 50 &nbsp;|
        |&nbsp;&nbsp;Dogon, tribe of Mali, West Africa|&nbsp;&nbsp;47 &nbsp;|&nbsp; 52 &nbsp;|
        |&nbsp;&nbsp;Zulu, South Africa|&nbsp;&nbsp;55 &nbsp;|&nbsp; 46 &nbsp;|
        |Australia, Melanesia|||
        |&nbsp;&nbsp;Australia, Lower Murray River|&nbsp;&nbsp;52 &nbsp;|&nbsp; 49 &nbsp;|
        |&nbsp;&nbsp;Tasmania, general|&nbsp;&nbsp;45 &nbsp;|&nbsp; 42 &nbsp;|
        |&nbsp;&nbsp;Tolai, tribe of north New Britain|&nbsp;&nbsp;56 &nbsp;|&nbsp; 54 &nbsp;|
        |Polynesia|||
        |&nbsp;&nbsp;Hawaii, Mokapu Peninsula, Oahu|&nbsp;&nbsp;51 &nbsp;|&nbsp; 49 &nbsp;|
        |&nbsp;&nbsp;Easter Island, general|&nbsp;&nbsp;49 &nbsp;|&nbsp; 37 &nbsp;|
        |&nbsp;&nbsp;Moriori, Chatham Islands|&nbsp;&nbsp;57 &nbsp;|&nbsp; 51 &nbsp;|
        |Americas|||
        |&nbsp;&nbsp;Arikara, Sully village site, South Dakota|&nbsp;&nbsp;42 &nbsp;|&nbsp; 27 &nbsp;|
        |&nbsp;&nbsp;Santa Cruz Island, California|&nbsp;&nbsp;51 &nbsp;|&nbsp; 51 &nbsp;|
        |&nbsp;&nbsp;Peru, Yauyos District|&nbsp;&nbsp;55 &nbsp;|&nbsp; 55 &nbsp;|
        |East Asia|||
        |&nbsp;&nbsp;North Japan, Hokkaido|&nbsp;&nbsp;55 &nbsp;|&nbsp; 32 &nbsp;|
        |&nbsp;&nbsp;South Japan, northern Kyushu|&nbsp;&nbsp;50 &nbsp;|&nbsp; 41 &nbsp;|
        |&nbsp;&nbsp;Hainan Island, South Chinese|&nbsp;&nbsp;45 &nbsp;|&nbsp; 38 &nbsp;|
        |Western Pacific (Austronesian-speaking)|||
        |&nbsp;&nbsp;Atayal, Taiwan aboriginals|&nbsp;&nbsp;29 &nbsp;|&nbsp; 18 &nbsp;|
        |&nbsp;&nbsp;Guam, Marianas, Latte period|&nbsp;&nbsp;30 &nbsp;|&nbsp; 27 &nbsp;|
        |Additional Local Series|||
        |&nbsp;&nbsp;Egypt, Gizeh, 26th-30th dynasties|&nbsp;&nbsp;58 &nbsp;|&nbsp; 53 &nbsp;|
        |&nbsp;&nbsp;San, South Africa, general|&nbsp;&nbsp;41 &nbsp;|&nbsp; 49 &nbsp;|
        |&nbsp;&nbsp;Andaman Islands, general|&nbsp;&nbsp;35 &nbsp;|&nbsp; 35 &nbsp;|
        |&nbsp;&nbsp;Ainu, south central Hokkaido|&nbsp;&nbsp;48 &nbsp;|&nbsp; 38 &nbsp;|
        |&nbsp;&nbsp;Buryats, Siberia|&nbsp;&nbsp;55 &nbsp;|&nbsp; 54 &nbsp;|
        |&nbsp;&nbsp;Inuit, Inugsuk culture, Greenland|&nbsp;&nbsp;53 &nbsp;|&nbsp; 55 &nbsp;|

        ---
        ***Acknowledgements***
        
        This project was supported by Award [No. 15PNIJ-22-GG-04431-RESS](https://nij.ojp.gov/funding/awards/15pnij-22-gg-04431-ress), awarded by the National Institute of Justice, Office of Justice Programs, U.S. Department of Justice. 
        The opinions, findings, and conclusions or recommendations expressed in this publication/program/exhibition are those of the author(s) and do not necessarily reflect those of the Department of Justice. 

        """
        ), 
    )),
    # Panel_3 
    ui.nav_panel("User manual", ui.div(
        ui.br(),ui.br(),
        ui.markdown(
            """
            
**CRANE User Manual**

Welcome to **CRANE (CRaniometric Ancestry Estimator)**, a UI-friendly application for estimating cranium ancestry using deep neural networks and craniometric measurements. 

This manual will guide you through data preparation, upload, analysis execution, and result interpretation.

---
## 0. CRANE Docker Usage Guide

```bash

## 1. Load a Docker image from a .tar file
sduo docker load -i crane-shiny.tar

## 2. Verify the image is loaded
sudo docker images

## 3. Run the Docker container in detached mode, mapping port 8123
sudo docker run -d -p 8123:8123 crane-shiny

## 4. Open CRANE in your web browser by navigating to:
http://localhost:8123

```

## 1. Data Format

1. The input file must be in **CSV** format with a `.csv` extension.  
2. Represent **missing values** using `NA`, sample `ID` is required。
3. The **column order** (features) must match the provided example data file (`CRANE_Example_data.csv`).  
4. Each run supports **only one** cranium sample . Do not include multiple cases in a single file.

You can download the example file from the “Data Uploading” panel and use it as a template.

---

## 2. Usage Steps

### 2.1 Download Example Data

1. In the **Data Uploading** panel, click **Download Example Data File**.  
2. Open `CRANE_Example_data.csv` to inspect column names and order.  

### 2.2 Prepare and Upload Your Data

1. Ensure your CSV file:  
   - Uses the **same column names** as the example.  
   - Maintains the **same column order**.  
   - Uses `NA` for any missing measurements.  
2. In the **Data Uploading** panel, click **Browse** or **Choose File** and select your CSV file.  
3. Once the file is detected, the **Run CRANE!** button will become enabled.  

### 2.3 Run the CRANE Analysis

1. Click **Run CRANE!**.  
2. The application will execute the following steps sequentially, each with a progress indicator:  
   1. **Imputation**: Preprocess data and fill missing values.  
   2. **Population Affinity Estimation**: Predict ancestry probabilities using the CNN model.  
   3. **2D UMAP Visualization**: Render a 2D UMAP plot of the sample.  
   4. **3D UMAP Visualization**: Render an interactive 3D UMAP plot.  
   5. **Confusion Matrix of Missingness Simulation**: Display a confusion matrix evaluating simulation performance with missingness.  
3. The relevant panel will automatically expand when each step starts.  
4. Refresh the page (press `Ctrl`+`F5` on your keyboard) to start a new session. 

### 2.4 View and Save Results

- **Step 1** panel shows a table of ancestry probability predictions.  
- **Step 2 / Step 3** panels display interactive 2D and 3D UMAP plots, respectively (zoom, pan, and rotate).  
- **Step 4** panel shows the confusion matrix for missingness simulations.  
- To save outputs, click the camera icon at the top of the legend on plots to download screenshots for external use.

---

## 3. Notes & Tips

- **Single-case execution**: Only one sample per run is supported. Batch analysis requires separate runs.  
- **Performance**: Step 1-3 may take 6-7 minutes, It might take about 5 more minutes to render the confusion matrix after Step 4 completes. please wait for the progress indicator to complete.  
- **Error handling**: If you encounter upload or analysis errors, verify your CSV format and column alignment. Contact support if issues persist.

---


## 4. Interpretation of CRANE Results

### Step 1: Population Affinity Estimation
This step presents the population affinity prediction results obtained using a deep neural network classifier trained on the main Howells craniometric dataset. For the uploaded cranium case, the model calculates the probabilities of belonging to each of 26 predefined population groups. The population with the highest predicted probability is highlighted, offering a data-driven inference about the most likely ancestry group of the sample.

### Step 2 & Step 3: UMAP-Based Data Visualization (2D and 3D)
These steps provide two- and three-dimensional visualizations of the population distribution using the UMAP (Uniform Manifold Approximation and Projection) technique. The uploaded cranium case is represented by a purple square marker, and its spatial relationship with the 26 reference populations is displayed. These plots enable users to visually explore the topological structure of population clusters and assess the proximity of the unknown case to known groups of population ancestry. The plots support interactive features such as zooming and selection, allowing for more detailed regional inspection and interpretation.

### Step 4: Missingness Simulation and Confusion Matrix
This step presents a confusion matrix summarizing the classification outcomes from simulations that replicate the missing data pattern observed in the uploaded case. By applying the same missingness structure to the training data, we evaluate the robustness of population classification under incomplete data conditions. The consistency of predicted affinity across simulations serves as supporting evidence to reinforce the credibility of the model's prediction for the uploaded case.

For more technical details and methodological background, please refer to our publication:  
DOI: [Insert DOI here](doi)


Thank you for using `CRANE`. We hope this powerful tool enhances your craniometric ancestry research!

            """
        ),
        ui.div(""),  
    )), 
    # Panel_4
    ui.nav_panel("Contact us",  ui.div(
        ui.br(),ui.br(),
        ui.markdown(
            """
            **Xiaoming Liu, Ph.D.**
            ```python
            
            Associate Professor
            Department of Global, Environmental, & Genomic Health Sciences
            College of Public Health
            University of South Florida
            Email: xiaomingliu@usf.edu
            ```
            **Jinyong Pang, M.S.**
            ```python
            
            Department of Biostatistics & Data Science
            College of Public Health
            University of South Florida
            Email: jpang@usf.edu
            ```
            """
        ),
        ui.div(""),
    )),      
    #ui.nav_panel("More", "Page 4 content"), 
    title = "Craniometric Ancestry Estimator",  
    id = "page",  position = "fixed-top", padding="50px",fillable = True,
) 

### Server
def server(input, output, session):
    @render.download(filename="./CRANE_Example_data.csv")
    def download_example():
        return open("./CRANE_Example_data.csv", "rb")
    #-----------------------Data Uploading--------------------------------#
    @reactive.calc
    def parsed_file():
        return pd.read_csv(input.file1()[0]["datapath"])
    
    @reactive.effect
    def _():
        ui.update_action_button("btn", disabled=not input.file1())
        ui.update_action_button("rst_btn", disabled=not input.file1())
    
    @reactive.effect
    @reactive.event(input.btn) 
    def _open_step1_window():
        ui.update_accordion("tab",show="s1")
       
    #-----------------------------0.Imputation--------------------------------#
    
    @reactive.calc    
    @reactive.event(input.btn, ignore_none=False)    
    def imputation(): ## dat
        #ui.update_accordion("tab",show="s1")
        with ui.Progress(min=0,max=1) as p:
            p.set(message="Data being Preprocessed",detail="")
            for _ in range(2):
                p.inc(0.75)
                #time.sleep(2)
                result0=impt(parsed_file())
                p.set(value=0.85,message="Data being Pre-processed",detail="")
                p.set(value=1,message="Data preprocessing, Done")
        return result0
        #return impt(parsed_file())
       
    @reactive.calc
    def plot_dat(): ##layer_3        
        with ui.Progress(min=0,max=1) as p:
            p.set(message="Data being Pre-processed for Step 2",detail="")
            for _ in range(2):
                p.inc(0.65)
                #time.sleep(2)
                result00=plotDT(imputation())
                p.set(value=1,message="Data preprocessing for Step 2, Done")
        return result00
        #return plotDT(imputation())
    
    #------------------------1.CRANE classification---------------------------#
    @render.table
    @reactive.event(input.btn, ignore_none=False) 
    def pred_table():
        with ui.Progress(min=0,max=1) as p:
            p.set(message="Step 1 being processed",detail="")
            for _ in range(2):
                p.inc(0.6)
                #time.sleep(2)
                result1=CNNpredict(imputation())
                p.set(value=0.8,message="Step 1 being processed",detail="")
                p.set(value=1,message="Step 1, Done")
        ui.update_accordion("tab",show=["s2","s1"])
        return result1
        #return CNNpredict(imputation())

    #------------------------------2.2D UMAP----------------------------------# 
    #@output
    #@render_plotly
    @render_widget
    @reactive.event(input.btn, ignore_none=False) 
    def plot_2d():
        with ui.Progress(min=0,max=1) as p:
            p.set(message="Step 2 being processed",detail="")
            for _ in range(2):
                p.inc(0.6)
                #time.sleep(2)
                result2=plot2D(plot_dat())
                p.set(value=0.85,message="Step 2 being processed",detail="")
                p.set(value=1,message="Step 2, Done")
        ui.update_accordion("tab",show=["s3","s2","s1"])
        return result2
        #return plot2D(plot_dat())

    #------------------------------3.3D UMAP----------------------------------# 
    #@output
    #@render_plotly
    @render_widget
    #@reactive.event(plot_dat) 
    def plot_3d():
        with ui.Progress(min=0,max=1) as p:
            p.set(message="Step 3 being processed",detail="")
            for _ in range(2):
                p.inc(0.75)
                #time.sleep(2)
                result3=plot3D(plot_dat())
                p.set(value=0.85,message="Step 3 being processed",detail="")
                p.set(value=1,message="Step 3, Done")
        ui.update_accordion("tab",show=["s4","s3","s2","s1"])
        return result3
        #return plot3D(plot_dat())

    #-------------------------4.Comfusion Matrix------------------------------# 
    @render.plot
    #@reactive.event(input.btn, ignore_none=False) 
    def plot_confusion_matrix():
        with ui.Progress(min=0,max=1) as p:
            p.set(message="Step 4 being processed",detail="Please allow 4–6 minutes for the confusion matrix to load after Step 4")
            for _ in range(2):
                p.inc(0.75)
                #time.sleep(1)
                result4=plotCM(parsed_file())
                p.set(value=0.90,message="Step 4 being processed",detail="Please allow 4–6 minutes for the confusion matrix to load after Step 4")
                p.set(value=1,message="Step 4, Done", detail="The confusion matrix may take about 4-6 minutes to load" )
                input.file1()[0]["datapath"]=''
        #ui.update_accordion("tab",show=['s1','s2','s3','s4'])
        return result4      
        #return plotCM(parsed_file())

app = App(app_ui, server)

###shiny run app_V.py --reload