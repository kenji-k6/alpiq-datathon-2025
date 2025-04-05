import pandas as pd
import os

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def impute_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function backfills the consumption data for the metering dataframe.
    """
    orig_cols = df.columns.tolist()
    df["hour"] = df["DATETIME"].dt.hour
    df["year"] = df["DATETIME"].dt.month

    customer_cols = df.columns.drop(["DATETIME", "hour", "year"])
    for c in customer_cols:
      if df[c].isna().sum() == 0:
          continue

      first_valid_idx = df[c].first_valid_index()
      if first_valid_idx is None:
          continue

      first_valid_time = df.loc[first_valid_idx, 'DATETIME']

      seasonal_flag = f'{c}_is_seasonal'
      fallback_flag = f'{c}_is_fallback'
      df[seasonal_flag] = 0
      df[fallback_flag] = 0

      missing_indices = df[(df['DATETIME'] < first_valid_time) & (df[c].isna())].index
      post_rollout_values = df.loc[df['DATETIME'] >= first_valid_time, c].dropna().values
      fallback_ptr = 0

      for idx in missing_indices:
          target_time = df.loc[idx, 'DATETIME']
          future_time = target_time + pd.DateOffset(years=1)

          match_row = df[df['DATETIME'] == future_time]
          if not match_row.empty and not pd.isna(match_row[c].values[0]):
              df.at[idx, c] = match_row[c].values[0]
              df.at[idx, seasonal_flag] = 1
          else:
              df.at[idx, c] = post_rollout_values[fallback_ptr % len(post_rollout_values)]
              df.at[idx, fallback_flag] = 1
              fallback_ptr += 1

    df = df[orig_cols]
    return df


def merge_dfs(
        consumption_df: pd.DataFrame,
        rollout_df: pd.DataFrame,
        holidays_df: pd.DataFrame,spv_df: pd.DataFrame, 
        country: str) -> pd.DataFrame:
    """
    This function loads the data from the csv files and merges them into a single dataframe.
    :param path: path to the folder containing the csv files
    :param country: country code (e.g. "IT" or "ES")
    :return: merged dataframe
    """

    # Set the holidays properly
    df = pd.merge(
        consumption_df, rollout_df,
        on="DATETIME",
        how="inner"
    )
    df["is_holiday"] = 0

    for holiday in holidays_df[f"holiday_{country}"]:
        df.loc[df["DATETIME"] == holiday, "is_holiday"] = 1

    df = pd.merge(
        df, spv_df,
        on="DATETIME",
        how="inner"
    )

    return df


def get_example_solution(path: str, country: str) -> pd.DataFrame:
    example_solution_path = os.path.join(path, "example_set_" + country + ".csv")
    return pd.read_csv(
        example_solution_path,
        index_col=0,
        parse_dates=True,
        date_format=DATE_FORMAT
    )


def get_train_forecast_split(path: str, country) -> tuple[pd.DataFrame, pd.DataFrame]:
    consumption_path = os.path.join(path, "historical_metering_data_" + country + ".csv")
    rollout_path = os.path.join(path, "rollout_data_" + country + ".csv")
    holidays_path = os.path.join(path, "holiday_" + country + ".xlsx")
    spv_path = os.path.join(path, "spv_ec00_forecasts_es_it.xlsx")

    # Load the data
    consumption_df = pd.read_csv(consumption_path, parse_dates=["DATETIME"], date_format=DATE_FORMAT)
    rollout_df = pd.read_csv(rollout_path, parse_dates=["DATETIME"], date_format=DATE_FORMAT)
    holidays_df = pd.read_excel(holidays_path, parse_dates=[f"holiday_{country}"], date_format=DATE_FORMAT)
    spv_df = pd.read_excel(spv_path, sheet_name=country)

    # Rename the columns to match the format of the other dataframes, datatype is datetime already
    spv_df.rename(columns={"Unnamed: 0": "DATETIME"}, inplace=True)


    # clean the consumption data
    if os.path.exists(os.path.join(path, f"imputed_consumption_{country}.csv")):
        consumption_df = pd.read_csv(os.path.join(path, f"imputed_consumption_{country}.csv"), parse_dates=["DATETIME"])
    else:
      consumption_df = impute_consumption(consumption_df)
      consumption_df.to_csv(os.path.join(path, f"imputed_consumption_{country}.csv"), index=False)

    merged_df = merge_dfs(consumption_df, rollout_df, holidays_df, spv_df, country)
    consumption_df.set_index("DATETIME", inplace=True)
    merged_df.set_index("DATETIME", inplace=True)

    example_sol = get_example_solution(path, country)

    start_training, end_training = consumption_df.index.min(), consumption_df.index.max()
    start_forecast, end_forecast = example_sol.index[0], example_sol.index[-1]

    range_training = pd.date_range(start=start_training, end=end_training, freq="1h")
    range_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq="1h")

    print(range_training)
    print(range_forecast)

    training_df = pd.DataFrame(columns=merged_df.columns, index=range_training)
    forecast_df = pd.DataFrame(columns=merged_df.columns, index=range_forecast)

    training_df.to_csv(os.path.join(path, f"training_set_{country}.csv"), index=True)
    forecast_df.to_csv(os.path.join(path, f"forecast_set_{country}.csv"), index=True)

    return training_df, forecast_df

    



    




class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

        consumptions_path = os.path.join(self.path, "historical_metering_data_" + country + ".csv")
        features_path = os.path.join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = os.path.join(self.path, "example_set_" + country + ".csv")

        consumptions = pd.read_csv(
            consumptions_path, index_col=0, parse_dates=True, date_format=DATE_FORMAT
        )
        features = pd.read_excel(
            features_path,
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=DATE_FORMAT,
        )
        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=DATE_FORMAT,
        )

        return consumptions, features, example_solution


