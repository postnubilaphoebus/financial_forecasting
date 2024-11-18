# financial_forecasting
Trying out some methods for financial forecasting <br> 
Implementation of the paper: Jiang, J., Kelly, B. T., & Xiu, D. (2020). (Re-)Imag(in)ing price trends. Chicago Booth Research Paper, (21-01) 
<br> https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13268 <br>
<br> <br>

# Installation guide: <br> <br> 

cd financial_forecasting <br>
chmod +x install.sh <br>
./install.sh <br> <br>

# Data Creation <br>

source .venv/bin/activate <br>
python load_data.py <br>
python create_database.py <br> <br>

# Running training <br>

python train.py


