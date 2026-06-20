# Inferencia - Predicción de Lluvia (weatherAUS)

Modelo: Regresión Logística (optimizada con Optuna)  
Target: `RainTomorrow` (0 = No llueve, 1 = Llueve)


## Construir la imagen

Desde la carpeta `docker/`:

```bash
docker build -t rain-predictor .
```

## Ejecutar el contenedor

El script espera un CSV de entrada con las columnas originales del dataset weatherAUS.  
El archivo CSV debe estar accesible desde el contenedor (se monta como volumen).

```bash
docker run --rm \
  -v "/proyecto:/data" \
  rain-predictor \
  --input /data/datos.csv \
  --output /data/predicciones.csv
```

```powershell
docker run --rm `
  -v "proyecto:/data" `
  rain-predictor `
  --input /data/datos.csv `
  --output /data/predicciones.csv
```
