import numpy as np
import pandas as pd


def normalize_submission(
    df,
    data_specs,
    *,
    require_identifier=True,
):
    """
    Normalize a complete synthetic table to the shared schema.

    Uses:
        input_vars
        numerical_vars
        boolean_vars
        categorical_vars
        target_var
        identifier_var

    Optional schema settings:
        nullable_vars:
            Columns allowed to contain missing values.
            Defaults to allowing missing feature values, but not
            missing targets or identifiers.

        categorical_values:
            Mapping from categorical feature names to allowed values.
            If omitted for a feature, its categories are unrestricted.

    Policy:
        - Reject missing, extra, or duplicate columns.
        - Convert numerical features to float64.
        - Convert booleans to pandas nullable boolean.
        - Accept boolean values, 0/1, and their string representations.
        - Convert categorical features to pandas string.
        - Require nonmissing target labels 0/1.
        - Convert identifiers to strings and require uniqueness.
        - Reject entirely missing feature columns.
        - Return a copy; do not modify the input.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Submission must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Submission must contain rows")

    if not df.columns.is_unique:
        raise ValueError("Submission has duplicate column names")

    target = data_specs["target_var"]
    identifier = data_specs.get("identifier_var")
    features = list(data_specs["input_vars"])

    if not features or len(features) != len(set(features)):
        raise ValueError(
            "input_vars must contain unique feature names"
        )

    metadata = {target, identifier} - {None}
    if len(metadata) != (2 if identifier else 1):
        raise ValueError("Target and identifier must differ")

    if set(features) & metadata:
        raise ValueError(
            "Target and identifier cannot be input features"
        )

    type_lists = {
        key: [
            column
            for column in data_specs.get(key, [])
            if column != target
        ]
        for key in (
            "numerical_vars",
            "boolean_vars",
            "categorical_vars",
        )
    }

    declared = [
        column
        for columns in type_lists.values()
        for column in columns
    ]

    if (
        len(declared) != len(set(declared))
        or set(declared) != set(features)
    ):
        raise ValueError(
            "Each input feature must have exactly one declared type"
        )

    if require_identifier and not identifier:
        raise ValueError(
            "identifier_var is required by the shared schema"
        )

    required = features + [target]
    if identifier and require_identifier:
        required.append(identifier)

    allowed = set(features + [target])
    if identifier:
        allowed.add(identifier)

    missing = [
        column for column in required
        if column not in df.columns
    ]
    extra = [
        column for column in df.columns
        if column not in allowed
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")

    nullable = set(
        data_specs.get("nullable_vars", features)
    )
    if not nullable <= set(features):
        raise ValueError(
            "nullable_vars may contain only input features"
        )

    category_values = data_specs.get(
        "categorical_values", {}
    )
    if not set(category_values) <= set(
        type_lists["categorical_vars"]
    ):
        raise ValueError(
            "categorical_values may describe only categorical features"
        )

    result = df.copy()

    def numeric_values(column):
        source = result[column]

        # Empty CSV fields normally arrive as missing already.
        # Nonmissing strings that cannot be parsed remain errors.
        try:
            values = pd.to_numeric(source, errors="raise")
            array = values.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{column}: expected numerical values"
            ) from exc

        if np.isinf(array).any():
            raise ValueError(
                f"{column}: infinite values are not allowed"
            )

        return pd.Series(array, index=result.index)

    for column in type_lists["numerical_vars"]:
        result[column] = numeric_values(column)

    boolean_strings = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    def boolean_value(value):
        if pd.isna(value):
            return pd.NA

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in boolean_strings:
                return boolean_strings[normalized]

        elif isinstance(
            value,
            (bool, np.bool_, int, float, np.integer, np.floating),
        ):
            if value == 0:
                return False
            if value == 1:
                return True

        raise ValueError("expected True/False or 0/1")

    for column in type_lists["boolean_vars"]:
        try:
            result[column] = pd.Series(
                [
                    boolean_value(value)
                    for value in result[column]
                ],
                index=result.index,
                dtype="boolean",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{column}: expected True/False, 0/1, "
                "or missing values"
            ) from exc

    for column in type_lists["categorical_vars"]:
        result[column] = result[column].astype("string")

        if column in category_values:
            allowed_values = {
                str(value)
                for value in category_values[column]
            }
            invalid = (
                result[column].notna()
                & ~result[column].isin(allowed_values)
            )

            if invalid.any():
                raise ValueError(
                    f"{column}: contains categories "
                    "outside the shared schema"
                )

    # No target mapping: the task uses 0 = survival, 1 = death.
    target_values = numeric_values(target)

    if (
        target_values.isna().any()
        or not target_values.isin([0, 1]).all()
    ):
        raise ValueError(
            f"{target}: expected nonmissing binary labels 0/1"
        )

    result[target] = target_values.astype("int64")

    if identifier and identifier in result.columns:
        result[identifier] = result[identifier].astype("string")

        if (
            result[identifier].isna().any()
            or result[identifier].str.strip().eq("").any()
        ):
            raise ValueError(
                f"{identifier}: identifiers must not be missing or empty"
            )

        if result[identifier].duplicated().any():
            raise ValueError(
                f"{identifier}: identifiers must be unique"
            )

    for column in features:
        if result[column].isna().all():
            raise ValueError(
                f"{column}: entirely missing columns are not allowed"
            )

        if column not in nullable and result[column].isna().any():
            raise ValueError(
                f"{column}: missing values are not allowed"
            )

    column_order = (
        ([identifier] if identifier in result.columns else [])
        + features
        + [target]
    )

    return result.loc[:, column_order]