import pandas as pd


def unpivot_by_two_columns(df: pd.DataFrame, value_columns: list[str],
                           value_name: str, variable_name: str,
                           variable_name_replace: dict[str, str]):

    # select only the columns we are interested in
    df = df[value_columns]

    # unpivot
    df = pd.melt(df,
                 ignore_index=False,
                 var_name=variable_name,
                 value_name=value_name,
                 value_vars=value_columns)
    df[variable_name] = df[variable_name].replace(variable_name_replace)

    return df
