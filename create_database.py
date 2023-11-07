import os
import sqlite3
from tqdm import tqdm

# Step 1: Extract attributes from file names
def extract_attributes(file_name):
    split_name = file_name.split("_")
    n_days = int(split_name[0])
    stock_name = split_name[2]
    name = split_name[4]
    trend = split_name[-1].split(".")[0]
    dict = {"name": name, "n_days": n_days, "stock_symbol": stock_name, "trend": trend}
    return dict

# Step 2: Create an SQLite database and a table
conn = sqlite3.connect('file_attributes.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS file_attributes (
        file_name TEXT,
        name TEXT,
        n_days INTEGER,
        stock_symbol TEXT,
        trend TEXT,
        PRIMARY KEY (file_name)
    )
''')

# Step 3: Parse through the files and insert data into the database
cur_path = os.getcwd()
folder_path = os.path.join(cur_path, "data")
for file_name in tqdm(os.listdir(folder_path)):
    attributes = extract_attributes(file_name)
    if attributes:
        cursor.execute('''
            INSERT OR IGNORE INTO file_attributes (file_name, name, n_days, stock_symbol, trend)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_name[:-4], attributes['name'], attributes['n_days'], attributes['stock_symbol'], attributes['trend']))
conn.commit()
conn.close()
