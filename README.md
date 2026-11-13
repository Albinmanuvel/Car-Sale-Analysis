# Car price prediction

Predicting car prices (regression) from engine and body specs. This was one of my early machine learning projects, kept here as a learning record.

## Data
`CarPrice.csv` - 205 cars x 26 columns (make, body type, engine type, engine size, horsepower, mpg, price, etc). There is no year or mileage column.

## What the script does (`cars.py`)
1. Basic cleaning and a look at unique values per column.
2. Feature selection with Welch ANOVA (categorical vs price) and Pearson correlation (numeric vs price). Features with no relationship to price (doornumber, fueltype, peakrpm, compressionratio, stroke, carheight) are dropped. A few categories have only one car, so Welch ANOVA is skipped for those.
3. One-hot encoding, 70/30 train/test split (`random_state=42`).
4. Four regressors compared by R2 on the test set.

## Results (test R2)
| Model | R2 |
|---|---|
| Linear Regression | 0.849 |
| Decision Tree | 0.872 |
| XGBoost RF | 0.919 |
| Random Forest | 0.941 |

Small dataset (205 rows) and a single split, so these numbers are optimistic. Cross-validation would be the next step.

## Run it
```
pip install -r requirements.txt
python cars.py
```
