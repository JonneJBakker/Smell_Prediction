import numpy as np


def inverse_transform_target(y, pipeline, target_column):
    """
    Inverts the scaling (and if applied, the log transform) for the target column.
    y: numpy array of predictions or true targets (in transformed space).
    pipeline: dictionary containing:
        - "scaler": fitted StandardScaler
        - "numeric_cols": list of numeric columns (order as used during fitting)
        - "log_transform": dict with keys "columns" (list) and "min_values" (dict)
    target_column: name of the target column.
    Returns y in the original (interpretable) scale.
    """
    try:
        # Invert scaling:
        scaler = pipeline["scaler"]
        if scaler is None:
            print("There is no scaler in the pipeline")
            return y

        numeric_cols = pipeline["numeric_cols"]
        # Check if target is in numeric columns
        if target_column not in numeric_cols:
            print(
                f"WARNING: Target column '{target_column}' not found in numeric_cols. Available columns: {numeric_cols}"
            )
            return y

        target_idx = numeric_cols.index(target_column)

        # StandardScaler stores parameters for each column:
        y_inversed = (
            y * scaler.scale_[target_idx] + scaler.mean_[target_idx]
        )  # Could use inverse_transform here as it's more elegant

        # If the target was log-transformed, invert that as well.
        log_transform_info = pipeline.get("log_transform", {})
        log_columns = log_transform_info.get("columns", [])

        if target_column in log_columns:
            min_values = log_transform_info.get("min_values", {})
            if target_column not in min_values:
                print(
                    f"WARNING: Target column '{target_column}' marked for log transform but no min_value found."
                )
                print(f"Available min_values: {min_values}")
                return y_inversed

            min_val = min_values[target_column]
            y_inversed = (
                np.expm1(y_inversed) + min_val
            )  # expm1 because we're using log1p

        return y_inversed

    except Exception as e:
        print(f"ERROR during inverse transform: {e}")
        print(f"Target column: {target_column}")
        print(f"Pipeline keys: {list(pipeline.keys())}")
        if "log_transform" in pipeline:
            print(
                f"Log transform columns: {pipeline['log_transform'].get('columns', [])}"
            )
            print(
                f"Min values keys: {list(pipeline['log_transform'].get('min_values', {}).keys())}"
            )
        # Return original array in case of error
        return y
