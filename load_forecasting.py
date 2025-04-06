import os
import pandas as pd
from typing import List
import numpy as np

# depending on your IDE, you might need to add datathon_eth. in front of data
from data import create_dataset_splits, load_reference_solution

# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import XGBoostEnergyForecastModel

from weather import calculate_country_average

def get_customer_ids(df: pd.DataFrame) -> List[int]:
    customer_cols = [col for col in df.columns if col.startswith("VALUEMWHMETERINGDATA")]
    customer_ids = [int(col.split("_")[-1]) for col in customer_cols]
    return customer_ids


def main(zone: str) -> None:
    """

    Train and evaluate the models for IT and ES

    """

    # Inputs
    data_dir = r"datasets2025"
    processed_data_dir = r"processed"
    output_dir = r"outputs"

    train_df, forecast_df = create_dataset_splits(
        data_dir=data_dir,
        processed_data_dir=processed_data_dir,
        country_code=country
    )

    example_results = load_reference_solution(data_dir, zone)

    """
    EVERYTHING STARTING FROM HERE CAN BE MODIFIED.
    """
    team_name = "Segfault"
    # Data Manipulation and Training
    model = XGBoostEnergyForecastModel()
    # model.train(train_df, zone, get_customer_ids(train_df))
    forecast = model.predict(forecast_df, zone, get_customer_ids(forecast_df))

    # """
    # END OF THE MODIFIABLE PART.
    # """

    # test to make sure that the output has the expected shape.
    dummy_error = np.abs(forecast - example_results).sum().sum()
    assert np.all(forecast.columns == example_results.columns), (
        "Wrong header or header order."
    )
    assert np.all(forecast.index == example_results.index), (
        "Wrong index or index order."
    )
    assert isinstance(dummy_error, np.float64), "Wrong dummy_error type."
    assert forecast.isna().sum().sum() == 0, "NaN in forecast."
    # Your solution will be evaluated using
    # forecast_error = np.abs(forecast - testing_set).sum().sum(),
    # and then doing a weighted sum the two portfolios:
    # score = forecast_error_IT + 5 * forecast_error_ES

    forecast.to_csv(
        os.path.join(output_dir, "students_results_" + team_name + "_" + country + ".csv")
    )
    pass


if __name__ == "__main__":
    country = "IT"
    main(country)

    # Average out the results of the different models used