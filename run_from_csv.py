import pandas as pd
from performance_monitor import run_monitor

CSV_PATH = "csv-file.csv"

df = pd.read_csv(CSV_PATH)

print(run_monitor(df=df))