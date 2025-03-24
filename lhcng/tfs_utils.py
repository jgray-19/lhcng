"""
tfs_utils.py
============

Utility functions for processing TFS files to be used with MAD-NG.
Includes functions to:
- Convert TFS data (headers and columns) to a MAD-X–friendly format.
- Filter the TFS DataFrame to include only BPM data.
- Read, convert, and export a TFS file.
"""

from pathlib import Path
import tfs

def convert_tfs_to_madx(tfs_df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Convert the TFS DataFrame to a format compatible with MAD-X.
    
    This function performs the following steps:
      - Converts all column names and header keys to uppercase.
      - Renames the columns 'MU1' and 'MU2' to 'MUX' and 'MUY', respectively.
      - Renames drift element names to consecutive values starting at 'DRIFT_0'.
      - Removes rows containing 'vkicker' or 'hkicker' in the 'KIND' column.
      - Sets the 'NAME' column as the index and removes rows with '$start' or '$end'.
    
    Parameters
    ----------
    tfs_df : tfs.TfsDataFrame
        The input TFS DataFrame.
    
    Returns
    -------
    tfs.TfsDataFrame
        The converted TFS DataFrame.
    """
    # Convert all headers and column names to uppercase.
    tfs_df.columns = tfs_df.columns.str.upper()
    tfs_df.headers = {key.upper(): value for key, value in tfs_df.headers.items()}

    # Rename columns 'MU1' and 'MU2' to 'MUX' and 'MUY'
    tfs_df = tfs_df.rename(columns={"MU1": "MUX", "MU2": "MUY"})

    # Replace drift names so that they are consecutive (starting from DRIFT_0)
    drifts = tfs_df[tfs_df["KIND"] == "drift"]
    replace_names = [f"DRIFT_{i}" for i in range(len(drifts))]
    tfs_df["NAME"] = tfs_df["NAME"].replace(drifts["NAME"].to_list(), replace_names)

    # Remove rows containing 'vkicker' or 'hkicker' in the 'KIND' column.
    tfs_df = tfs_df[~tfs_df["KIND"].str.contains("vkicker|hkicker")]

    # Set the NAME column as index and remove unwanted rows
    tfs_df = tfs_df.set_index("NAME")
    tfs_df = tfs_df.filter(regex=r"^(?!\$start|\$end).*$", axis="index")

    return tfs_df


def filter_out_BPM_near_IPs(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Filter the TFS DataFrame to include only BPM rows.
    
    Uses a regex filter to select only those rows whose index (NAME)
    starts with 'BPM.' followed by a number.
    
    Parameters
    ----------
    df : tfs.TfsDataFrame
        The input TFS DataFrame.
    
    Returns
    -------
    tfs.TfsDataFrame
        The filtered TFS DataFrame containing only BPMs.
    """
    return df.filter(regex=r"^BPM\.[1-9][0-9].", axis="index")


def export_tfs_to_madx(tfs_file: Path) -> None:
    """
    Read a TFS file, convert its contents to MAD-X format, and write it back.
    
    This function uses the `convert_tfs_to_madx` function to adjust the TFS file
    for compatibility with MAD-X and then writes the converted DataFrame back to disk.
    
    Parameters
    ----------
    tfs_file : Path
        Path to the TFS file.
    """
    tfs_df = tfs.read(tfs_file)
    tfs_df = convert_tfs_to_madx(tfs_df)
    tfs.write(tfs_file, tfs_df, save_index="NAME")
