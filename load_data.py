import yfinance as yf
from generate_training_data import write_n_day_images
import os
from tqdm import tqdm

nasdaq = ["ADBE", 
          "ADI",
          "ADP",
          "ADSK",
          "AEP", 
          "ALGN",
          "AMAT",
          "AMGN",
          "ANSS",
          "AZN",
          "BIIB",
          "BKR",
          "CEG",
          "CHTR",
          "CMCSA",
          "CPRT",
          "CSCO",
          "CSGP",
          "CSX",
          "CTAS",
          "CTSH",
          "DLTR",
          "EA",
          "EBAY",
          "ENPH",
          "FANG",
          "FAST",
          "FTNT",
          "GEHC",
          "GFS",
          "GILD",
          "GOOG",
          "HON",
          "ILMN",
          "INTU",
          "ISRG",
          "JD",
          "KDP",
          "KHC",
          "KLAC",
          "LCID",
          "LRCX",
          "LULU",
          "MAR",
          "MDLZ",
          "ΜΕΤΑ",
          "MNST",
          "MRNA",
          "MRVL",
          "MU",
          "NVDA",
          "ODFL",
          "ON",
          "ORLY",
          "PAYX",
          "PCAR",
          "PDD",
          "PEP",
          "QCOM",
          "REGN",
          "ROST",
          "SBUX",
          "SGEN",
          "SIRI",
          "SNPS",
          "TEAM",
          "TMUS",
          "TXN",
          "VRTX",
          "WDAY",
          "XEL",
          "ZM",
          "ZS"]



current_path = os.getcwd()
new_path = os.path.join(current_path, "data")

if not os.path.exists(new_path):
    os.makedirs(new_path)

for acronym in tqdm(nasdaq):
    hist = yf.Ticker(acronym).history(period = "max")
    write_n_day_images(hist, 20, 64, new_path, acronym)

msft = yf.Ticker("AAPL")

# get historical market data
hist = msft.history(period= "max")

print("type(hist)", type(hist))

print("hist", hist)



































                 
        
               
