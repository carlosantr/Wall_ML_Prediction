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

import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

#%%Prediction Function - Individual
def prediction_ML_walls_individual(WI, T1, H_Tcr, Ar, ALR_G, AI, Sa, Sv,
                                   var_predict=["PFA_max","rDR_max"], model_predict=["RF", "ANN"]):
    """
    WI: Wall Index
    T1: Fundamental period 
    H_Tcr: Stiffness Index
    Ar: Mean wall aspect ratio
    ALR_G: Average axial load ratio for gravity loads 
    AI: Arias Intensity
    Sa: Spectral acceleration 
    Sv: Spectral velocity
    ___________________________________________________________________
    *The next variables are optional according to the user preference*
    var_predict: The variables to predict
        PFA_max: Peak floor acceleration
        rDR_max: Roof drift ratio
    model_predict: The Machine Learning models to do the predictions
        RF: Random Forest
        ANN: Artificial Neural Networks    
    """
    df_prediction = pd.DataFrame(index=model_predict, columns=var_predict)#Dataframe to save predictions
    for var in var_predict:
        for model in model_predict:
            #Import Scalers
            path_scalerX = f"Models/Scalers/ScalerX_{var}_{model}.pkl"
            path_scalerY = f"Models/Scalers/ScalerY_{var}_{model}.pkl"
            scalerX = joblib.load(path_scalerX)
            scalerY = joblib.load(path_scalerY)
            #Import models
            if model == "ANN":
                path_model = f"Models/{var}/{var}_{model}.h5"
                regressor = load_model(path_model, custom_objects={'mse': MeanSquaredError()})
            elif model == "RF":
                path_model = f"Models/{var}/{var}_{model}.pkl"
                regressor = joblib.load(path_model)
            #Creating dataframe of predictors (and scaling the data)
            X = pd.DataFrame({'IM-Arq':[WI],
                              'T-Arq':[T1],
                              'H/Tcr-Arq':[H_Tcr],
                              'Ar_Mean':[Ar],
                              'ALR-G (%)':[ALR_G],
                              'IA':[AI],
                              'Sa':[Sa],
                              'Sv':[Sv]})
            X_scaled = scalerX.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            #Prediction
            prediction_scaled = regressor.predict(X_scaled)
            prediction = scalerY.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
            #Saving the prediction
            df_prediction.loc[model, var] = prediction
            
    return df_prediction

#%%Prediction function - Dataframe
def prediction_ML_walls_multiple(X,
                                 var_predict=["PFA_max","rDR_max"], model_predict=["RF", "ANN"]):
    """
    X is a dataframe with the next columns (each row is a prediction):
        WI: Wall Index
        T1: Fundamental period 
        H_Tcr: Stiffness Index
        Ar: Mean wall aspect ratio
        ALR_G: Average axial load ratio for gravity loads 
        AI: Arias Intensity
        Sa: Spectral acceleration 
        Sv: Spectral velocity
        ***************************************
        The columns must have the same name
        i.e., X.columns = ['WI', 'T1', 'H_Tcr', 'Ar', 'ALR_G', 'AI', 'Sa', 'Sv']
        ***************************************
    ___________________________________________________________________
    *The next variables are optional according to the user preference*
    var_predict: The variables to predict
        PFA_max: Peak floor acceleration
        rDR_max: Roof drift ratio   
    model_predict: The Machine Learning models to do the predictions
        RF: Random Forest
        ANN: Artificial Neural Networks    
    """
    #Verifying if X have the necessary columns
    X_cols_df = ['WI', 'T1', 'H_Tcr', 'Ar', 'ALR_G', 'AI', 'Sa', 'Sv']
    if list(X.columns) != X_cols_df:
        ValueError(f"The dataframe X does not have the correct columns: {X_cols_df}")
    #Pre-procesing the input data
    X = X[X_cols_df]
    X.rename(columns={"ALR_G":"ALR-G (%)",
                      "WI":"IM-Arq",
                      "H_Tcr":"H/Tcr-Arq",
                      "Ar":"Ar_Mean",
                      "AI":"IA",
                      "T1":"T-Arq"},inplace=True)
    #Creating prediction columns
    col = [f"{var} - {model}" for var in var_predict for model in model_predict]
    df_prediction = pd.DataFrame(columns=col)
    for var in var_predict:
        for model in model_predict:
            #Import Scalers
            path_scalerX = f"Models/Scalers/ScalerX_{var}_{model}.pkl"
            path_scalerY = f"Models/Scalers/ScalerY_{var}_{model}.pkl"
            scalerX = joblib.load(path_scalerX)
            scalerY = joblib.load(path_scalerY)
            #Import models
            if model == "ANN":
                path_model = f"Models/{var}/{var}_{model}.h5"
                regressor = load_model(path_model, custom_objects={'mse': MeanSquaredError()})
            elif model == "RF":
                path_model = f"Models/{var}/{var}_{model}.pkl"
                regressor = joblib.load(path_model)
            #Scaling the data
            X_scaled = scalerX.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            #Prediction
            prediction_scaled = regressor.predict(X_scaled)
            prediction = scalerY.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
            #Saving the predictions
            df_prediction[f"{var} - {model}"] = prediction
    return df_prediction
