import sys
sys.path.append(r"C:\Users\BUSGR025\Desktop\local_code\DataLoader")
from loader import load_data

_data = load_data(category="stock", universe="B500")

prices         = _data["prices"]          # DataFrame (date x ticker)    prezzi giornalieri
bench_weights  = _data["weights"]         # DataFrame (date x ticker)    pesi nel costituents, normalizzati
gics           = _data["gics"]            # Series    (ticker)            settore GICS short code
common_columns = _data["common_columns"]  # list                          ticker in comune prezzi/weights
common_dates   = _data["common_dates"]    # DatetimeIndex                 date in comune (solo weekday)
