# Data

`raw/<TICKER>.csv` holds daily prices for ten VN30 stocks: ACB, ACV, BID, CTG, FPT, GAS, HPG,
VCB, VHM and VIC. Each file merges the yearly Investing.com "historical data" exports used in the
original project, keeping the latest export when dates overlap.

Columns: `date` (ISO), `close`, `open`, `high`, `low` (VND per share) and `volume` (shares,
empty when the export had "-").

| Ticker | First date | Last date | Rows |
|---|---|---|---|
| ACB | 2014-01-02 | 2024-03-15 | 2540 |
| ACV | 2016-11-22 | 2024-03-15 | 1822 |
| BID | 2014-01-27 | 2024-03-15 | 2526 |
| CTG | 2014-01-02 | 2024-03-15 | 2543 |
| FPT | 2014-01-02 | 2024-03-15 | 2543 |
| GAS | 2014-01-02 | 2024-03-15 | 2543 |
| HPG | 2014-01-02 | 2024-03-15 | 2543 |
| VCB | 2014-01-02 | 2024-03-15 | 2543 |
| VHM | 2018-05-18 | 2024-03-15 | 1458 |
| VIC | 2014-01-02 | 2024-03-15 | 2543 |

Known quirks in the source data: VHM jumps 38% on 2018-05-22, a few days after listing, and
VIC jumps 27% on 2014-08-20. Both are kept as exported.
