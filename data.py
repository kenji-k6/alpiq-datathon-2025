"""
Data Processing Pipeline for the Datathon 2025 Alpiq Challenge

This module handles data loading, merging, cleaning, and splitting as 
well as imputation of missing values.
"""


from asyncio import streams
from multiprocessing import process
import pandas as pd
import os
from typing import Tuple
from weather import calculate_country_average


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def impute_energy_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing consumption values (Y-values), as well as  missing
    rollout data (part of X-values) using seasonal patterns and fallback values.

    Strategy:
    1. Try seasonal imputation using the value at t+1 year.
    2. If seasonal imputation fails, use a fallback value from the
        post-rollout period (t+1 year) using round-robin sampling.
    
    Loop runs in reverse chronological order to allow chaining of
    seasonal imputation.
    3. Forward-fill any remaining missing values after processing.

    Args:
        data (pd.DataFrame): DataFrame containing the data to be processed.
    
    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    processed = data.copy()
    original_columns = processed.columns.tolist()

    # Create a mapping from datetime to index for efficient access
    datetime_index = {dt: idx for idx, dt in processed["DATETIME"].items()}

    for col in processed.columns.drop(["DATETIME"]):
        if not processed[col].isna().any():
            continue
        
        # Find first valid observations
        first_valid_idx = processed[col].first_valid_index()
        if first_valid_idx is None:
            continue
        
        first_valid_dt = processed.at[first_valid_idx, "DATETIME"]

        # Initialize imputation flags
        seasonal_flag = f"{col}_seasonal_imputed"
        fallback_flag = f"{col}_fallback_imputed"
        processed[seasonal_flag] = 0
        processed[fallback_flag] = 0

        # Identify missing values needing imputation
        missing_mask = (
            (processed["DATETIME"] < first_valid_dt)
            & processed[col].isna()
        )
        missing_indices = processed.index[missing_mask]

        # Get valid post-rollout values
        post_rollout_values = processed.loc[
            processed["DATETIME"] >= first_valid_dt, col
        ].dropna().values

        if len(post_rollout_values) == 0:
            continue
        
        fallback_counter = 0

        # Process missing indices in reverse chronological order
        for idx in reversed(missing_indices):
            current_dt = processed.at[idx, "DATETIME"]
            seasonal_candidate = current_dt + pd.DateOffset(years=1)
            candidate_idx = datetime_index.get(seasonal_candidate)

            # Attempt seasonal imputation
            if candidate_idx is not None:
                candidate_value = processed.at[candidate_idx, col]
                is_fallback = processed.at[candidate_idx, fallback_flag]

                if not pd.isna(candidate_value) and not is_fallback:
                    processed.at[idx, col] = candidate_value
                    processed.at[idx, seasonal_flag] = 1
                    continue
            
            # Fallback to roun-robin sampling from post-rollout values
            processed.at[idx, col] = post_rollout_values[
                fallback_counter % len(post_rollout_values)
            ]
            processed.at[idx, fallback_flag] = 1
            fallback_counter += 1

    # Remove temporary columns and restore original order
    processed = processed[original_columns]

    # Forward-fill any remaining missing values
    processed = processed.ffill()
    processed = processed.fillna(0)
    return processed


def merge_data_sources(
    consumption_data: pd.DataFrame,
    rollout_data: pd.DataFrame,
    holidays_data: pd.DataFrame,
    spv_data: pd.DataFrame,
    country_code: str
) -> pd.DataFrame:
    """
    This function loads the data from the csv files and merges them into a single dataframe.

    Args:
        consumption_df (pd.DataFrame): DataFrame containing consumption data.
        rollout_df (pd.DataFrame): DataFrame containing rollout data.
        holidays_df (pd.DataFrame): DataFrame containing holiday data.
        spv_df (pd.DataFrame): DataFrame containing PV data.
        country_code (str): country code (e.g. "IT" or "ES").

    Returns:
        pd.DataFrame: Merged DataFrame containing all data.
    """
    # Merge the consumption and rollout data
    merged = pd.merge(
        consumption_data, rollout_data,
        on="DATETIME",
        how="outer"
    )

    # Add holiday indicators using vectorized operations
    country_holidays = holidays_data[f"holiday_{country_code}"]
    merged["is_holiday"] = merged["DATETIME"].isin(country_holidays).astype(int)

    # Incorporate PV data
    return pd.merge(
        merged, spv_data,
        on="DATETIME",
        how="outer"
    )


def load_reference_solution(data_dir: str, country_code: str) -> pd.DataFrame:
    """Load the example solution dataset for assertions and slicing"""
    solution_path = os.path.join(data_dir, f"example_set_{country_code}.csv")
    return pd.read_csv(
        solution_path,
        index_col=0,
        parse_dates=True,
        date_format=DATE_FORMAT
    )


def create_dataset_splits(
    data_dir: str,
    processed_data_dir: str,
    country_code: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create training and forecast-ready datasets for the specified country.

    Args:
        data_dir (str): Directory containing the data files.
        country_code (str): Country code (e.g. "IT" or "ES").
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training and forecast datasets.
    """
    # Cache paths for training and forecast datasets
    train_path = os.path.join(processed_data_dir, f"training_set_{country_code}.csv")
    forecast_path = os.path.join(processed_data_dir, f"forecast_set_{country_code}.csv")

    # Return cached datasets if they exist
    if os.path.exists(train_path) and os.path.exists(forecast_path):
        return (
            pd.read_csv(train_path, index_col=0, parse_dates=True),
            pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        )

    # Load the raw data
    consumption = _load_and_preprocess(
        data_dir, processed_data_dir, country_code, "historical_metering_data", impute_energy_data
    )
    rollout = _load_and_preprocess(
        data_dir, processed_data_dir, country_code, "rollout_data", impute_energy_data
    )

    # Load supplementary data
    holidays = pd.read_excel(
        os.path.join(data_dir, f"holiday_{country_code}.xlsx"),
        parse_dates=[f"holiday_{country_code}"],
        date_format=DATE_FORMAT
    )
    spv = pd.read_excel(
        os.path.join(data_dir, "spv_ec00_forecasts_es_it.xlsx"),
        sheet_name=country_code
    ).rename(columns={"Unnamed: 0": "DATETIME"})

    # Create unified dataset
    merged_data = merge_data_sources(
        consumption, rollout, holidays, spv, country_code
    )
    weather_data = calculate_country_average(
        processed_data_dir="processed",
        country_code=country_code
    ).reset_index().drop(columns=["DATETIME"])

    merged_data = pd.concat([
        merged_data,
        weather_data], axis=1)

    # Compute datetime ranges
    example_solution = load_reference_solution(data_dir, country_code)
    full_train_range = pd.date_range(
        start=consumption["DATETIME"].min(),
        end=consumption["DATETIME"].max(),
        freq="1h"
    )
    forecast_range = pd.date_range(
        start=example_solution.index[0],
        end=example_solution.index[-1],
        freq="1h"
    )

    # Generate splits with complete datetime indices
    training_data = _create_temportal_split(merged_data, full_train_range)
    forecast_data = _create_temportal_split(merged_data, forecast_range)

    # Ensure complete training timeline (daylight saving gaps), forward-fill missing values
    training_data = training_data.reindex(full_train_range).ffill()

    # Save the datasets to CSV files
    training_data.to_csv(train_path, index=True)
    forecast_data.to_csv(forecast_path, index=True)

    return training_data, forecast_data
        

def _load_and_preprocess(
    data_dir: str,
    processed_data_dir: str,
    country_code: str,
    base_filename: str,
    imputation_func: callable
) -> pd.DataFrame:
    """
    Helper function to load and preprocess data"""
    file_path = os.path.join(data_dir, f"{base_filename}_{country_code}.csv")
    cached_path = os.path.join(processed_data_dir, f"imputed_{base_filename}_{country_code}.csv")

    if os.path.exists(cached_path):
        return pd.read_csv(cached_path, parse_dates=["DATETIME"], date_format=DATE_FORMAT)
    
    data = pd.read_csv(file_path, parse_dates=["DATETIME"], date_format=DATE_FORMAT)
    processed = imputation_func(data)
    processed.to_csv(cached_path, index=False)
    return processed


def _create_temportal_split(
    data: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create time-based data split with proper index"""
    return (
        data[data["DATETIME"].isin(date_range)]
          .set_index("DATETIME")
          .sort_index()
    )


def _get_weather_feature(country_code: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch and process weather data for the specified country.

    Args:
        country_code (str, "IT" or "ES"): Country code.
        data (pd.DataFrame): DataFrame containing the data to be processed.
    """

    weather_df = calculate_country_average(
        processed_data_dir="processed",
        country_code=country_code
    )

    result = pd.merge(
        data,
        weather_df,
        how="left"
    )

    return result
