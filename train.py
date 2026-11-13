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
X = df_clean[features] # die bekannten daten zum lernen
y = df_clean["koi_disposition"] # das was man vorhersagen möchte, Einordnung in Confirmed oder False positive

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # training

predictions = model.predict(X_test) # Übergabe der Testdate, ohne Lösung
print(f"Acurracy: {accuracy_score(y_test, predictions):.2%}\n") # Vergleich der Testdaten mit den Vorhersagen des modells
print(classification_report(y_test, predictions)) # die Leistung für beide Klassen wird aufgeteilt in Genauigkeit und Trefferquote

joblib.dump(model, "exoplanet_model.pkl")
print("Model saved.")
