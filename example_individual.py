# Copyright 2025, Carlos Emilio Angarita Trillos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%% Importing libraries
import pandas as pd
#If you are using the prediction functions locally
from predict import prediction_ML_walls_individual 
#If you install the open-source library (on your command prompt: "pip install wall_ml_prediction")
#from wall_ml_prediction.predict import prediction_ML_walls_individual 

#%%Importing the example data
Input_Data = pd.read_excel("Example dataset/Example.xlsx")
Target_Data = Input_Data[["PFA_max","rDR_max"]] #To save the target values
Input_Data.drop(columns=["PFA_max","rDR_max"], inplace=True) #To delete the columns "PFA_max" and "rDR_max" with the target values

#%%Taking only one sample to predict in the individual mode
index = 15 #Modify this number between 0 and 30 to change a sample
individual_sample = Input_Data.loc[index]

#%%Predict PFA_max and rDR_max with the function in individual mode
variables = ["PFA_max","rDR_max"]
models = ["RF", "ANN"]
df_prediction = prediction_ML_walls_individual(WI = individual_sample["WI"], 
                                               T1 = individual_sample["T1"], 
                                               H_Tcr = individual_sample["H_Tcr"], 
                                               Ar = individual_sample["Ar"], 
                                               ALR_G = individual_sample["ALR_G"], 
                                               AI = individual_sample["AI"], 
                                               Sa = individual_sample["Sa"], 
                                               Sv = individual_sample["Sv"],
                                               var_predict = variables,
                                               model_predict = models)

#%%Print the results
for var in variables:#For each output variable
    print(f"{var} target value: {Target_Data.loc[index, var]}")
    for model in models:#For each model
        print(f"     Prediction {model}: {df_prediction.loc[model, var]}")
    print("")
        
