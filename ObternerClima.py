import requests
import pandas as pd

def obtener_clima_df(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "America/New_York"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "rain_bool": [r > 0 for r in data["hourly"]["precipitation"]]
    })
    return df

anios = [2015, 2009, 2014, 2011, 2012, 2010, 2013]

lat, lon = 40.7128, -74.0060

dfs = []
for anio in anios:
    start_date = f"{anio}-01-01"
    end_date = f"{anio}-12-31"
    print(f"Descargando {anio}...")
    df_anio = obtener_clima_df(lat, lon, start_date, end_date)
    df_anio["year"] = anio
    dfs.append(df_anio)

df_ny = pd.concat(dfs, ignore_index=True)

# Mostrar un resumen
print(df_ny.head())
df_ny.info()

# Exportar a CSV
df_ny.to_csv("clima_nyc.csv", index=False, encoding="utf-8")
print("Archivo clima_nyc.csv generado con éxito.")