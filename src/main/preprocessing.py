import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project root to sys.path

from utils.common_imports import *

def preprocess_data():

    spx_df = pd.read_csv("data/raw/SPX_16Yrs_EOD.csv")
    spx_df['date_column'] = pd.to_datetime(spx_df['time'], unit='s').dt.date
    spx_df.drop(columns=['time', 'Volume'], inplace=True)

    vix_df = pd.read_csv("data/raw/VIX_EOD_2010.csv")
    vix_df['date_column'] = pd.to_datetime(vix_df['time'], unit='s').dt.date
    # vix_df = vix_df[['open', 'date_column']]
    vix_df.drop(columns=['time', 'high', 'low', 'close', 'Volume'], inplace=True)
    vix_df.rename(columns={"open":"open_vix"}, inplace=True)

    # Merge DataFrames based on the 'date_column' column
    merged_df = pd.merge(spx_df, vix_df, on='date_column', how='inner')

    merged_df['year'] = pd.to_datetime(merged_df['date_column']).dt.year
    merged_df['month'] = pd.to_datetime(merged_df['date_column']).dt.month
    merged_df['day'] = pd.to_datetime(merged_df['date_column']).dt.day
    merged_df.drop(columns=['date_column'], inplace=True)
    merged_df.to_csv("data/processed/spx_vix_data.csv")

    return None