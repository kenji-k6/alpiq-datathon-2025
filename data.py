import re
import pandas as pd
from os.path import join

def get_cleaned_df(path: str, country: str) -> pd.DataFrame:
    """
    This function loads the data from the csv files and merges them into a single dataframe.
    :param path: path to the folder containing the csv files
    :param country: country code (e.g. "IT" or "ES")
    :return: merged dataframe
    """

    date_format = "%Y-%m-%d %H:%M:%S"
    holiday_col = f"holiday_{country}"

    consumption_path = join(path, "historical_metering_data_" + country + ".csv")
    rollout_path = join(path, "rollout_data_" + country + ".csv")
    holidays_path = join(path, "holiday_" + country + ".xlsx")
    spv_path = join(path, "spv_ec00_forecasts_es_it.xlsx")

    # Load the data
    consumption_df = pd.read_csv(consumption_path)
    rollout_df = pd.read_csv(rollout_path)
    holidays_df = pd.read_excel(holidays_path)
    spv_df = pd.read_excel(spv_path, sheet_name=country)

    # Rename the columns to match the format of the other dataframes, datatype is datetime already
    spv_df.rename(columns={"Unnamed: 0": "DATETIME"}, inplace=True)
  
    # Convert the timestamp columns to datetime objects
    consumption_df["DATETIME"] = pd.to_datetime(consumption_df["DATETIME"],
                                                 format=date_format
                                                 )

    rollout_df["DATETIME"] = pd.to_datetime(rollout_df["DATETIME"],
                                             format=date_format
                                             )


    holidays_df[holiday_col] = pd.to_datetime(
        holidays_df[holiday_col], format=date_format
    )

    # clean the consumption data
    consumption_df = backfill_consumption(consumption_df)

    # Set the holidays properly
    df = pd.merge(
        consumption_df, rollout_df,
        on="DATETIME",
        how="inner"
    )
    df["is_holiday"] = 0

    for holiday in holidays_df[holiday_col]:
        df.loc[df["DATETIME"] == holiday, "is_holiday"] = 1

    df = pd.merge(
        df, spv_df,
        on="DATETIME",
        how="inner"
    )

    return df


def backfill_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function backfills the consumption data for the metering dataframe.
    """

    df = df.loc[:,df.std() != 0]
    df = df.ffill()

    while df.isna().sum().sum() > 0:
      latest_valid_idx = df.apply(pd.Series.first_valid_index).max()
      relevant_cols = df.columns[
          df.apply(pd.Series.first_valid_index) == latest_valid_idx
      ]

      next_valid_idx = df.drop(columns=relevant_cols).apply(pd.Series.first_valid_index).max()

      corr_matrix = df.loc[latest_valid_idx:].corr()

      for col in relevant_cols:
          threshold = 0.9
          while df.loc[next_valid_idx:latest_valid_idx, col].isna().sum().sum() > 0:
              threshold -= 0.1
              high_corr_cols = corr_matrix[col].drop(col)[corr_matrix[col].drop(col) > threshold].index
              if len(high_corr_cols) > 1:
                  # Replace values in the target column with the mean of highly correlated columns
                  df.loc[next_valid_idx:latest_valid_idx, col] = (
                      df.loc[next_valid_idx:latest_valid_idx, high_corr_cols].mean(axis=1)
                      )
    return df
    




class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")
        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

        consumptions = pd.read_csv(
            consumptions_path, index_col=0, parse_dates=True, date_format=date_format
        )
        features = pd.read_excel(
            features_path,
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        return consumptions, features, example_solution


# Encoding Part


class SimpleEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        end_training,
        start_forecast,
        end_forecast,
    ):
        self.consumption_mask = ~consumption.isna()
        self.consumption = consumption[self.consumption_mask]
        self.features = features
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast

    def meta_encoding(self):
        """
        This function returns the feature, split between past (for training) and future (for forecasting)),
        as well as the consumption, without missing values.
        :return: three numpy arrays

        """
        features_past = self.features[: self.end_training].values.reshape(-1, 1)
        features_future = self.features[
            self.start_forecast : self.end_forecast
        ].values.reshape(-1, 1)

        features_past = features_past[self.consumption_mask]

        return features_past, features_future, self.consumption
