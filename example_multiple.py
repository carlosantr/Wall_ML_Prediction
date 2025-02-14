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
import matplotlib.pyplot as plt
from predict import prediction_ML_walls_multiple

#%%Importing the example data
Input_Data = pd.read_excel("Example dataset/Example.xlsx")
Target_Data = Input_Data[["PFA_max","rDR_max"]] #To save the target values
Input_Data.drop(columns=["PFA_max","rDR_max"], inplace=True) #To delete the columns "PFA_max" and "rDR_max" with the target values

#%%Predict PFA_max and rDR_max with the function in multiple mode
variables = ["PFA_max","rDR_max"]
models = ["RF", "ANN"]
df_prediction = prediction_ML_walls_multiple(Input_Data,
                                             var_predict = variables,
                                             model_predict = models)

#%%Visualize the results in scatter plots for each model and output variable
for model in models:#For each model
    for var in variables:#For each output variable
        plt.title(f"Results {var} - {model}")
        plt.xlabel("Prediction")
        plt.ylabel("Target value")
        plt.scatter(df_prediction[f"{var} - {model}"],
                    Target_Data[f"{var}"],
                    c="black")
        maximum_value = max([max(df_prediction[f"{var} - {model}"]),max(Target_Data[f"{var}"])]) #Find the limit of the plot
        plt.plot([0,maximum_value],[0,maximum_value]) # Fitting line
        plt.show()