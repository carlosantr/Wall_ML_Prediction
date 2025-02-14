#ML prediction - RC Wall buildings

This repository presents the trained ML models of the paper "Machine Learning based evaluation of roof drifts and accelerations of RC wall buildings during ground motions". Artificial Neural Netwroks (ANN) and Random Forest (RF) are available to make predictions of the maximum roof acceleration (PFA_max) and the maximum roof drift ratio (rDR_max) in reinforced concrete wall buildings. This repository presents the trained models, prediction functions (please see the function documentation in the .py file) and examples for multiple and individual prediction modes. You can use the prediction functions in your own project with two different ways:

1. Install the open-source library distributed through Python Package index (PyPi) using the command "pip install wall_ml_prediction" in your local command prompt (Anaconda prompt is recommended). Once the installation is done, you can call the prediction functions in your projects with the command "from wall_ml_prediction.predict import prediction_ML_walls_individual" or ""from wall_ml_prediction.predict import prediction_ML_walls_multiple". Then, use freely the functions to make predictions of PFA_max or rDR_max using the trained ML models (i.e. RF and ANN).

2. Download this repository and locate the files (in some cases you will need to unzip the downloaded file from the repository) in the same path of your project. Then, call the functions from the predict.py file as follows "from predict import prediction_ML_walls_individual" or ".predict import prediction_ML_walls_multiple". Then, use freely the functions to make predictions of PFA_max or rDR_max using the trained ML models (i.e. RF and ANN).

For further information about the input data required by the functions, please see the documentation in the predict.py file in this repository. The abovementioned documentation is presented below:

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

### IMPORTANT NOTE: 
For the second case (downloading directly from the repository), you have to accomplish with the dependencies requirements in your python environment (Scikit-Learn: write in console: pip install --user scikit-learn==1.2.1, Joblib: write in console: pip install --user joblib==1.4.2, Keras: wite in console: pip install --user keras==3.5.0, and Tensorflow: write in console: pip install --user tensorflow==2.17.0.
