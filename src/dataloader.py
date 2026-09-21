# data_loader.py

from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd


class LoadForecastDataLoader:

    def __init__(
        self,
        csv_path,
        datetime_col,
        features,
        target,
        lookback,
        horizon,
        datetime_format=None,          # NEW
        expected_frequency=None,
        sort_by_datetime=True,
        check_numeric=True,
    ):

        self.csv_path = Path(csv_path)

        self.datetime_col = datetime_col

        self.features = features

        if isinstance(target, str):
            self.target = [target]
        else:
            self.target = target

        self.lookback = lookback
        self.horizon = horizon

        # Optional datetime format
        self.datetime_format = datetime_format

        self.expected_frequency = expected_frequency

        self.sort_by_datetime = sort_by_datetime

        self.check_numeric = check_numeric

        self.df = None

        self.abnormalities = {}

        # Load and validate
        self._load_csv()
        self._validate_structure()
        self._validate_columns()
        self._validate_datetime()
        self._validate_data_types()
        self._check_missing_values()
        self._check_duplicate_timestamps()
        self._check_time_order()
        self._check_time_gaps()
        self._check_numeric_abnormalities()

    # =========================================================
    # LOAD CSV
    # =========================================================

    def _load_csv(self):

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}"
            )

        if self.csv_path.suffix.lower() != ".csv":
            raise ValueError(
                "Input file must be a CSV file."
            )

        try:
            self.df = pd.read_csv(
                self.csv_path
            )
        except Exception as e:
            raise ValueError(
                f"Could not read CSV file: {e}"
            )

        if self.df.empty:
            raise ValueError(
                "CSV file contains no data."
            )

    # =========================================================
    # STRUCTURE VALIDATION
    # =========================================================

    def _validate_structure(self):

        # Check column names
        if any(
            not isinstance(col, str)
            for col in self.df.columns
        ):
            raise ValueError(
                "All CSV headers must be strings."
            )

        # Remove accidental whitespace from headers
        cleaned_columns = [
            col.strip()
            for col in self.df.columns
        ]

        if len(cleaned_columns) != len(
            set(cleaned_columns)
        ):
            raise ValueError(
                "CSV contains duplicate column headers."
            )

        self.df.columns = cleaned_columns

        # Check for unnamed columns
        unnamed_columns = [
            col for col in self.df.columns
            if col.lower().startswith("unnamed:")
        ]

        if unnamed_columns:

            raise ValueError(
                "CSV contains unnamed columns: "
                f"{unnamed_columns}"
            )

    # =========================================================
    # COLUMN VALIDATION
    # =========================================================

    def _validate_columns(self):

        if self.datetime_col not in self.df.columns:

            raise ValueError(
                f"Datetime column '{self.datetime_col}' "
                f"not found.\n"
                f"Available columns: "
                f"{list(self.df.columns)}"
            )

        # Features
        missing_features = [
            col
            for col in self.features
            if col not in self.df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Feature columns not found: "
                f"{missing_features}"
            )

        # Target
        missing_targets = [
            col
            for col in self.target
            if col not in self.df.columns
        ]

        if missing_targets:

            raise ValueError(
                f"Target columns not found: "
                f"{missing_targets}"
            )

        # Datetime cannot be a feature
        if self.datetime_col in self.features:

            raise ValueError(
                "Datetime column cannot be used "
                "directly as a numeric feature."
            )

        # Target cannot be duplicated
        duplicate_targets = set(
            self.features
        ).intersection(self.target)

        if duplicate_targets:

            raise ValueError(
                "The following columns are both "
                f"features and targets: "
                f"{duplicate_targets}"
            )

    # =========================================================
    # DATETIME VALIDATION
    # =========================================================

    def _validate_datetime(self):

        if self.datetime_col not in self.df.columns:
            raise ValueError(
                f"Datetime column '{self.datetime_col}' "
                f"not found."
            )

        # -----------------------------------------------------
        # If format is supplied, enforce it
        # -----------------------------------------------------

        if self.datetime_format is not None:

            parsed_datetime = pd.to_datetime(
                self.df[self.datetime_col],
                format=self.datetime_format,
                errors="coerce"
            )

        # -----------------------------------------------------
        # Otherwise let pandas infer the format
        # -----------------------------------------------------

        else:

            parsed_datetime = pd.to_datetime(
                self.df[self.datetime_col],
                errors="coerce",
                utc=True
            )

        # -----------------------------------------------------
        # Check invalid datetime values
        # -----------------------------------------------------

        invalid_datetime = (
            parsed_datetime.isna()
        )

        if invalid_datetime.any():

            invalid_rows = (
                self.df.loc[
                    invalid_datetime,
                    self.datetime_col
                ]
                .head(10)
                .tolist()
            )

            count = invalid_datetime.sum()

            if self.datetime_format is not None:

                raise ValueError(
                    f"{count} invalid datetime values "
                    f"found in '{self.datetime_col}'.\n"
                    f"Expected format: "
                    f"'{self.datetime_format}'\n"
                    f"Examples of invalid values: "
                    f"{invalid_rows}"
                )

            else:

                raise ValueError(
                    f"{count} invalid datetime values "
                    f"found in '{self.datetime_col}'.\n"
                    f"Examples: {invalid_rows}"
                )

        # Replace original column with datetime objects
        self.df[self.datetime_col] = parsed_datetime

        # -----------------------------------------------------
        # Check duplicate timestamps
        # -----------------------------------------------------

        if self.df[
            self.datetime_col
        ].duplicated().any():

            count = self.df[
                self.datetime_col
            ].duplicated().sum()

            self.abnormalities[
                "duplicate_timestamps"
            ] = int(count)

        # -----------------------------------------------------
        # Sort chronologically
        # -----------------------------------------------------

        if self.sort_by_datetime:

            self.df = self.df.sort_values(
                self.datetime_col
            ).reset_index(drop=True)

    # =========================================================
    # DATA TYPE VALIDATION
    # =========================================================

    def _validate_data_types(self):

        columns_to_check = (
            self.features + self.target
        )

        if not self.check_numeric:
            return

        for col in columns_to_check:

            numeric_values = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

            invalid = (
                numeric_values.isna()
                & self.df[col].notna()
            )

            if invalid.any():

                count = invalid.sum()

                raise ValueError(
                    f"Column '{col}' contains "
                    f"{count} non-numeric values."
                )

            self.df[col] = numeric_values

    # =========================================================
    # MISSING VALUES
    # =========================================================

    def _check_missing_values(self):

        relevant_columns = (
            [self.datetime_col]
            + self.features
            + self.target
        )

        missing = (
            self.df[
                relevant_columns
            ]
            .isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

        if not missing.empty:

            self.abnormalities[
                "missing_values"
            ] = missing.to_dict()

    # =========================================================
    # DUPLICATE TIMESTAMPS
    # =========================================================

    def _check_duplicate_timestamps(self):

        duplicates = self.df[
            self.datetime_col
        ].duplicated(
            keep=False
        )

        if duplicates.any():

            duplicate_times = (
                self.df.loc[
                    duplicates,
                    self.datetime_col
                ]
                .tolist()
            )

            self.abnormalities[
                "duplicate_timestamp_values"
            ] = duplicate_times

    # =========================================================
    # TIME ORDER
    # =========================================================

    def _check_time_order(self):

        datetime_series = self.df[
            self.datetime_col
        ]

        if not datetime_series.is_monotonic_increasing:

            self.abnormalities[
                "datetime_not_sorted"
            ] = True

    # =========================================================
    # TIME GAPS
    # =========================================================

    def _check_time_gaps(self):

        if self.expected_frequency is None:
            return

        datetime_series = self.df[
            self.datetime_col
        ]

        differences = datetime_series.diff()

        expected_delta = pd.Timedelta(
            self.expected_frequency
        )

        abnormal_gaps = differences[
            differences != expected_delta
        ]

        # Ignore first row
        abnormal_gaps = abnormal_gaps.iloc[1:]

        if not abnormal_gaps.empty:

            self.abnormalities[
                "time_gaps"
            ] = {
                "count": len(abnormal_gaps),
                "examples": abnormal_gaps.head(
                    10
                ).to_dict()
            }

    # =========================================================
    # NUMERIC ABNORMALITIES
    # =========================================================

    def _check_numeric_abnormalities(self):

        columns = (
            self.features + self.target
        )

        numeric_report = {}

        for col in columns:

            series = self.df[col]

            report = {}

            # Infinite values
            infinite_count = np.isinf(
                series
            ).sum()

            if infinite_count > 0:

                report[
                    "infinite_values"
                ] = int(infinite_count)

            # Constant column
            if series.nunique(
                dropna=True
            ) <= 1:

                report[
                    "constant_column"
                ] = True

            # Negative values
            negative_count = (
                series < 0
            ).sum()

            if negative_count > 0:

                report[
                    "negative_values"
                ] = int(negative_count)

            if report:

                numeric_report[col] = report

        if numeric_report:

            self.abnormalities[
                "numeric_abnormalities"
            ] = numeric_report

    # =========================================================
    # SELECTED DATA
    # =========================================================

    def get_selected_dataframe(self):

        columns = (
            [self.datetime_col]
            + self.features
            + self.target
        )

        # Remove duplicates while preserving order
        columns = list(
            dict.fromkeys(columns)
        )

        return self.df[
            columns
        ].copy()

    # =========================================================
    # CREATE SEQUENCES
    # =========================================================

    def create_sequences(self):

        """
        Creates LSTM sequences.

        X shape:
            [samples, lookback, features]

        y shape:
            [samples, horizon, targets]
        """

        selected_df = (
            self.get_selected_dataframe()
        )

        X_data = selected_df[
            self.features
        ].values

        y_data = selected_df[
            self.target
        ].values

        X = []
        y = []

        total_length = (
            len(selected_df)
        )

        for i in range(
            total_length
            - self.lookback
            - self.horizon
            + 1
        ):

            X.append(
                X_data[
                    i :
                    i + self.lookback
                ]
            )

            y.append(
                y_data[
                    i + self.lookback :
                    i + self.lookback
                    + self.horizon
                ]
            )

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y

    # =========================================================
    # DATASET INFORMATION
    # =========================================================

    def summary(self):

        print("\n==============================")
        print("DATASET SUMMARY")
        print("==============================")

        print(
            f"File:       {self.csv_path}"
        )

        print(
            f"Rows:       {len(self.df)}"
        )

        print(
            f"Columns:    {len(self.df.columns)}"
        )

        print(
            f"Datetime:   {self.datetime_col}"
        )

        print(
            f"Features:   {self.features}"
        )

        print(
            f"Target:     {self.target}"
        )

        print(
            f"Lookback:   {self.lookback}"
        )

        print(
            f"Horizon:    {self.horizon}"
        )

        print(
            f"Frequency:  {self.expected_frequency}"
        )

        print(
            f"Start:      "
            f"{self.df[self.datetime_col].min()}"
        )

        print(
            f"End:        "
            f"{self.df[self.datetime_col].max()}"
        )

        print(
            f"\nAbnormalities:"
        )

        if not self.abnormalities:

            print("None detected.")

        else:

            for key, value in (
                self.abnormalities.items()
            ):

                print(
                    f"- {key}: {value}"
                )

        print("==============================\n")

    # =========================================================
    # RETURN DATAFRAME
    # =========================================================

    def get_dataframe(self):

        return self.df.copy()