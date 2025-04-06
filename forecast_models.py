import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
from typing import List, Dict



class XGBoostEnergyForecastModel:
    """
    XGBoost-based energy consumption forecasting model with hyperparameter optimization using Optuna.

    Attributes:
        gpu_id (int): GPU ID to use for training.
        use_gpu (bool): Whether to use GPU for training.
        models (dict): Dictionary to store trained models for each customer.
        best_params (dict): Dictionary to store best hyperparameters for each customer.
    """
    def __init__(self):
        """Initialize the model configuration with default values."""
        self.gpu_id = 0 # Default CUDA device ID
        self.use_gpu = True # Use GPU for training
        self.models = {} # CustomerID: trained XGBOost model
        self.best_params = {} # CustomerID: best hyperparameters
    
    def _get_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance DataFrame with time-based features.
        
        Converts temporal components into cylical sine/cosine features to
        improve detection of periodic patterns in the data.

        Args:
            df (pd.DataFrame): DataFrame containing data
        
        Returns:
            pd.DataFrame: DataFrame with additional time-based features

        """
        # Convert monthly patterns using sine/cosine with 12-month periodicity
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        # Daily patterns with 31-day normalization
        df["day_sin"] = np.sin(2 * np.pi * df.index.day / 31)
        df["day_cos"] = np.cos(2 * np.pi * df.index.day / 31)

        # Hourly patterns with 24-hour periodicity
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

        # Weekly patterns with 7-day periodicity
        df["weekday_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)
        return df
    
    def _get_gpu_params(self) -> Dict[str, str]:
        """Return GPU parameters for XGBoost.
        
        Returns:
            dict: Dictionary containing GPU parameters for XGBoost.
        """
        return {
            'tree_method': 'hist',  # Use GPU for training
            'device': 'cuda',  # Use GPU for training
        } if self.use_gpu else {}


    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
      """Optuna optimization objective function for hyperparameter tuning.

      Args:
          trial (optuna.Trial): Optuna trial object.
          X (pd.DataFrame): Feature DataFrame.
          y (pd.Series): Target Series.
      Returns:
          float: Mean absolute error of the model on the validation set.
      """
      params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8), # Constrain model complexity
            'subsample': trial.suggest_float('subsample', 0.7, 1.0), # Stochastic sampling
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0), # L2 regularization
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            **self._get_gpu_params()
        }
      
      tscv = TimeSeriesSplit(n_splits=3) # Time-aware cross-validation
      scores = []
      
      for train_idx, val_idx in tscv.split(X):
          # Temporal split preserrving time order
          X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
          y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
          model = XGBRegressor(
              objective='reg:absoluteerror', # MAE optimization
              n_estimators=2000, # Generous upper bound
              early_stopping_rounds=50, # Prevent overfitting
              **params
          )
          
          model.fit(
              X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False # Suppress training output
          )
          
          # Use best iteration from early stopping
          best_iter = model.best_iteration
          scores.append(mean_absolute_error(y_val, model.predict(X_val, iteration_range=(0, best_iter))))
      
      return np.mean(scores) # Aggregate CV performance


    
    def train(self, df: pd.DataFrame, country_code: str, customer_ids: List[int]):
        """Train separate model for each customer with hyperparameter optimization.
        
        Args:
            df (pd.DataFrame): DataFrame with features and targets
            country_code (str): Country code for the columns.
            customer_ids (List[int]): List of 
        """
        df = self._get_time_features(df.copy())
        idx = customer_ids.index(2816)
        for cust_id in customer_ids[idx:]:
            print(f"Training model for customer {cust_id}...")

            # Feature matrix construction
            X = df[[
                f"INITIALROLLOUTVALUE_customer{country_code}_{cust_id}",  # Lag feature
                "is_holiday", "spv", "temp",
                "temperature_2m", "relative_humidity_2m",
                "cloudcover", "precipitation",
                "month_sin", "month_cos",
                "day_sin",
                "day_cos",
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos"
            ]]
            y = df[f"VALUEMWHMETERINGDATA_customer{country_code}_{cust_id}"]

            split_idx = int(len(X) * 0.9)  # Train/test split (last 20% for testing)


            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]




            # Hyperparameter optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=10, n_jobs=-1)

            # Final model training with best hyperparameters
            best_params = study.best_params
            self.best_params[cust_id] = best_params

            final_model = XGBRegressor(
                objective='reg:absoluteerror',
                n_estimators=2000,
                early_stopping_rounds=50,
                n_jobs=-1,
                **best_params,
                **self._get_gpu_params()
            )

            final_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=100
            )

            self.models[cust_id] = final_model

            final_model.save_model(os.path.join(r"models", f"model_{country_code}_{cust_id}.json"))
            print(f"Trained {cust_id} | Test MAE: {mean_absolute_error(y_test, final_model.predict(X_test)):.4f}")
      
    def predict(self, df: pd.DataFrame, country_code: str, customer_ids: List[int]) -> pd.DataFrame:
      """Generate predictions for all customers of a country.
      
      Args:
          df (pd.DataFrame): DataFrame with required features
          country_code (str): Country code for the columns.
          customer_ids (List[int]): List of customer IDs to predict for.
      
      Returns:
          pd.DataFrame: Predictions with DateTimeIndex and customer columns
      """
      df = self._get_time_features(df.copy())
      predictions = pd.DataFrame(index=df.index)

      if len(self.models) == 0:
          for cust_id in customer_ids:
              print("Loading model for customer", cust_id)
              # Load the model from the file
              self.models[cust_id] = XGBRegressor()
              self.models[cust_id].load_model(os.path.join(r"models", f"model_{country_code}_{cust_id}.json"))
      
      for cust_id in customer_ids:
          print("Predicting for customer", cust_id)
          X = df[[
              f"INITIALROLLOUTVALUE_customer{country_code}_{cust_id}",  # Lag feature
              "is_holiday", "spv", "temp",
              "temperature_2m", "relative_humidity_2m",
              "cloudcover", "precipitation",
              "month_sin", "month_cos",
              "day_sin",
              "day_cos",
              "hour_sin",
              "hour_cos",
              "weekday_sin",
              "weekday_cos"
          ]]

          predictions[f"VALUEMWHMETERINGDATA_customer{country_code}_{cust_id}"] = self.models[cust_id].predict(X)
      return predictions
      


        
