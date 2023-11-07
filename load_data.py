import yfinance as yf
from generate_training_data import write_n_day_images
import os
import pandas as pd
from tqdm import tqdm

stock_list = pd.read_csv('stocks_list.csv')
stock_list = stock_list.Symbol.tolist()

current_path = os.getcwd()
new_path = os.path.join(current_path, "data")

if not os.path.exists(new_path):
    os.makedirs(new_path)

for acronym in tqdm(stock_list):
    try:
        hist = yf.Ticker(acronym).history(period = "max")
    except:
        continue
    write_n_day_images(hist, 20, 64, new_path, acronym)

print("Done")



































                 
        
               
