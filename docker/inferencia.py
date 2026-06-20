"""
Script de inferencia: predice si lloverá mañana (RainTomorrow) dado un CSV de entrada.

Uso:
    python inferencia.py --input datos.csv [--output predicciones.csv]

El CSV de entrada debe tener las columnas originales del dataset weatherAUS:
    Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
    WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
    WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
    Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent

DIRECCION_GRADOS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def load_artifacts():
    return {
        "mapeo":          joblib.load(ARTIFACTS / "mapeo_ciudad_region.joblib"),
        "ohe":            joblib.load(ARTIFACTS / "ohe_region.joblib"),
        "imp_mediana":    joblib.load(ARTIFACTS / "imputer_mediana.joblib"),
        "imp_knn":        joblib.load(ARTIFACTS / "imputer_knn.joblib"),
        "imp_iter":       joblib.load(ARTIFACTS / "imputer_iterative.joblib"),
        "scaler":         joblib.load(ARTIFACTS / "scaler.joblib"),
        "cols_escalar":   joblib.load(ARTIFACTS / "cols_a_escalar.joblib"),
        "feature_names":  joblib.load(ARTIFACTS / "feature_names.joblib"),
        "model":          joblib.load(ARTIFACTS / "modelo.joblib"),
        "threshold":      json.loads((ARTIFACTS / "threshold.json").read_text())["umbral"],
    }


def preprocess(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    df = df.copy()

    # Columnas a ignorar si aparecen
    df.drop(columns=["Unnamed: 0", "RainfallTomorrow", "RainTomorrow"], errors="ignore", inplace=True)

    # Codificación binaria de RainToday
    df["RainToday"] = df["RainToday"].astype(str).str.strip().str.title().replace({"Yes": 1, "No": 0})
    df["RainToday"] = pd.to_numeric(df["RainToday"], errors="coerce")

    # Features de fecha
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Month"]     = df["Date"].dt.month
    df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
    df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)

    # Region_BOM
    df["Region_BOM"] = df["Location"].map(art["mapeo"])

    # Drop columnas no necesarias
    df.drop(columns=["Date", "Location", "DayOfYear", "Month", "Region"], errors="ignore", inplace=True)

    # Codificación cíclica del viento
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        grados = df[col].map(DIRECCION_GRADOS)
        rad    = np.deg2rad(grados)
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
        df.drop(columns=[col], inplace=True)

    # OneHotEncoder Region_BOM
    col_names_ohe = [f"Region_{cat}" for cat in art["ohe"].categories_[0][1:]]
    enc = art["ohe"].transform(df[["Region_BOM"]])
    df_enc = pd.DataFrame(enc, columns=col_names_ohe, index=df.index)
    df = pd.concat([df.drop(columns=["Region_BOM"]), df_enc], axis=1)

    # Imputación
    cols_mediana  = ["MinTemp", "MaxTemp", "Rainfall", "WindSpeed9am", "WindSpeed3pm",
                     "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm", "RainToday",
                     "WindGustDir_sin", "WindGustDir_cos",
                     "WindDir9am_sin",  "WindDir9am_cos",
                     "WindDir3pm_sin",  "WindDir3pm_cos"]
    cols_knn      = ["Pressure9am", "Pressure3pm", "WindGustSpeed"]
    cols_iter     = ["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"]

    df[cols_mediana] = art["imp_mediana"].transform(df[cols_mediana])
    df[cols_knn]     = art["imp_knn"].transform(df[cols_knn])
    df[cols_iter]    = art["imp_iter"].transform(df[cols_iter])

    # Escalado
    df[art["cols_escalar"]] = art["scaler"].transform(df[art["cols_escalar"]])

    # Reordenar columnas exactamente como en entrenamiento
    df = df[art["feature_names"]]

    return df


def predict(df_raw: pd.DataFrame, art: dict) -> pd.DataFrame:
    X = preprocess(df_raw, art)
    proba = art["model"].predict_proba(X)[:, 1]
    pred  = (proba >= art["threshold"]).astype(int)
    return pd.DataFrame({
        "RainTomorrow_pred": pred,
        "probabilidad":      proba.round(4),
    })


def main():
    parser = argparse.ArgumentParser(description="Inferencia: predicción de lluvia mañana")
    parser.add_argument("--input",  required=True, help="CSV de entrada")
    parser.add_argument("--output", default="predicciones.csv", help="CSV de salida (default: predicciones.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: no se encontró el archivo '{input_path}'", file=sys.stderr)
        sys.exit(1)

    print("Cargando artefactos...")
    art = load_artifacts()

    print(f"Leyendo {input_path}...")
    df_raw = pd.read_csv(input_path)
    print(f"  {len(df_raw)} filas recibidas")

    print("Preprocesando y prediciendo...")
    resultados = predict(df_raw, art)

    output_path = Path(args.output)
    resultados.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en '{output_path}'")
    print(resultados.head())


if __name__ == "__main__":
    main()
