import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Loads dataset and cleans column names.
    """

    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    # Fix Total Charges column
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    return df