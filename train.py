import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("cumulative.csv")

df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]

features = [
    "koi_period", # Umlaufzeit des Planenten (in Tagen)
    "koi_duration", # Dauer des Transits (in Stunden)
    "koi_depth", # Helligkeitsabfall des Sterns 
    "koi_prad", # Geschätzter Planetenradius (in Erdradien)
    "koi_teq", # Gleichgewichtstemperatur des Planeten(in K)
    "koi_steff", # Oberflächentemperatur Stern (in K)
    "koi_srad", # Radius vom Stern (in Sonnenradien)
]

df_clean = df[features + ["koi_disposition"]].dropna() #keine NaN Werte
