# INE Obstruction Detector

Este script detecta obstrucciones en fotografías de credenciales INE, por ejemplo papel, notas adhesivas, dedos y brillos, y genera un reporte por región relevante de la credencial. 

## Qué hace

El flujo completo toma una imagen o una carpeta de imágenes, localiza el contorno de la credencial, la endereza a un tamaño canónico, determina si es anverso o reverso, y luego evalúa regiones de interés específicas para decidir si están obstruidas y con qué confianza. En anverso evalúa foto principal, holograma, bloque central de texto, bloque inferior de texto y zona de firma. En reverso evalúa los dos QR grandes, el QR pequeño y la MRZ. 

Para separar contenido real de “zonas tapadas” combina señales de textura, frecuencia, bordes y color. En particular usa entropía de textura con LBP para identificar regiones demasiado uniformes, análisis de frecuencia con DCT para medir qué tanta energía se concentra en el componente DC, una clasificación por gradientes para distinguir brillo contra papel según la nitidez del borde, y una fusión multi señal para decidir obstrucción a partir de varios puntajes. Para detectar el lado de la credencial usa detección de rostro con cascadas Haar como indicador fuerte de anverso, firma de textura de QR como indicador fuerte de reverso, conteo de líneas tipo MRZ y señales de layout.  

## Requisitos

Se requiere Python con `opencv-python` y `numpy`. El script usa las cascadas Haar incluidas en OpenCV. 

Instalación típica:

```bash
pip install opencv-python numpy
```

## Uso

El script acepta como entrada una imagen o una carpeta. 

Ejemplo con una sola imagen:

```bash
python ine_obstrucciones_detector_v8.py "ruta/a/ine.jpg" --json-out "salida/reporte.json" --debug-dir "salida/debug"
```

Ejemplo con una carpeta:

```bash
python ine_obstrucciones_detector_v8.py "ruta/a/carpeta_con_ines" --json-out "salida/reporte.json" --debug-dir "salida/debug"
```

Si necesitas forzar el lado cuando el encuadre es extremo o está muy tapada:

```bash
python ine_obstrucciones_detector_v8.py "ruta/a/ine.jpg" --force-side front
```

Parámetros disponibles: 

| Parámetro                   | Qué hace                                 |
| --------------------------- | ---------------------------------------- |
| `input`                     | Ruta a imagen o carpeta                  |
| `--debug-dir`               | Guarda overlays de depuración por imagen |
| `--json-out`                | Escribe el reporte completo en JSON      |
| `--force-side {front,back}` | Fuerza la clasificación de lado          |
| `--log-metrics`             | Loguea métricas detalladas para tuning   |
| `-v, --verbose`             | Modo verboso                             |


## Salidas

### Overlays de depuración

Si usas `--debug-dir`, por cada imagen se guarda un PNG con rectángulos por ROI, verde si está libre y rojo si se detectó obstrucción, además de la caja del “blob” cuando aplica.  

### Reporte JSON

Si usas `--json-out`, el archivo contiene una lista de resultados, uno por imagen, incluyendo el lado detectado, un booleano global `obstructed`, el detalle por región, y la ruta del overlay si se generó.   

Estructura esperada, simplificada:

```json
[
  {
    "path": "...",
    "ok": true,
    "side": "front",
    "obstructed": true,
    "regions": {
      "text_center": {
        "roi_px": [0, 0, 0, 0],
        "obstructed": true,
        "confidence": 0.72,
        "reason": "..."
      }
    },
    "debug_image": "salida/debug/ine__debug.png",
    "side_detection": {
      "confidence": 0.90,
      "face_detected": true,
      "qr_signature_score": 0.12,
      "mrz_lines": 0
    }
  }
]
```

## Resultados de la ultima prueba

| Img | Side             | Obstruido JSON | Regiones marcadas                               |
| --: | :--------------- | :------------- | :---------------------------------------------- |
|   1 | back, conf 0.85  | sí             | qr_mid:0.66; mrz:0.66                           |
|   2 | front, conf 0.86 | sí             | text_center:0.51; text_bottom:0.67              |
|   3 | front, conf 0.86 | sí             | mrz:0.68                                        |
|   4 | front, conf 0.77 | sí             | text_center:0.60; hologram_photo:0.58; mrz:0.52 |
|   5 | front, conf 0.75 | sí             | text_bottom:0.56; hologram_photo:0.79           |
|   6 | front, conf 0.76 | sí             | text_bottom:0.67; hologram_photo:0.66           |
|   7 | front, conf 0.87 | sí             | hologram_photo:0.54                             |
|   8 | front, conf 0.90 | sí             | text_center:0.54; text_bottom:0.83              |
|   9 | back, conf 0.87  | sí             | mrz:0.59                                        |
|  10 | back, conf 0.86  | sí             | qr_left:0.74; qr_mid:0.56; mrz:0.64             |
|  11 | back, conf 0.87  | sí             | qr_left:0.77; qr_mid:0.64; mrz:0.73             |
|  12 | back, conf 0.86  | sí             | mrz:0.60                                        |
|  13 | back, conf 0.86  | sí             | qr_small:0.62; mrz:0.57                         |
|  14 | back, conf 0.87  | sí             | mrz:0.60                                        |
|  15 | back, conf 0.87  | sí             | mrz:0.59                                        |
|  16 | back, conf 0.87  | sí             | qr_left:0.53; mrz:0.64                          |
|  17 | back, conf 0.87  | sí             | qr_left:0.71; qr_mid:0.61; mrz:0.51             |
|  18 | back, conf 0.87  | sí             | qr_left:0.67; qr_mid:0.64                       |
|  19 | back, conf 0.87  | sí             | qr_left:0.52; mrz:0.56                          |
|  20 | back, conf 0.87  | sí             | mrz:0.52                                        |
|  21 | back, conf 0.87  | sí             | mrz:0.58                                        |
|  22 | back, conf 0.87  | sí             | qr_small:0.62; mrz:0.56                         |
|  23 | back, conf 0.87  | no             | -                                               |
|  24 | back, conf 0.87  | sí             | qr_mid:0.63; mrz:0.61                           |
|  25 | front, conf 0.80 | no             | -                                               |
|  26 | back, conf 0.87  | sí             | qr_mid:0.63; mrz:0.69                           |
|  27 | back, conf 0.87  | sí             | qr_mid:0.50                                     |
|  28 | back, conf 0.74  | no             | -                                               |
|  29 | back, conf 0.86  | sí             | qr_left:0.70                                    |
|  30 | back, conf 0.83  | sí             | mrz:0.61                                        |
|  31 | front, conf 0.57 | sí             | hologram_photo:0.66                             |
|  32 | front, conf 0.95 | no             | -                                               |

### Falsos Positivos

| Img | Side             | Side correcto | Obstruido JSON | Regiones marcadas   | Validación                                            | Notas                                                     |
| --- | ---------------- | ------------- | -------------- | ------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| 12  | back, conf 0.86  | ✓             | sí             | mrz:0.60            | mrz ✗, ROI toma fondo bajo la tarjeta                 | Falso positivo en MRZ por caja demasiado baja             |
| 14  | back, conf 0.87  | ✓             | sí             | mrz:0.60            | mrz ✗, ROI se sale a la derecha                       | Falso positivo en MRZ por caja corrida                    |
| 23  | back, conf 0.87  | ✓             | no             | -                   | qr_mid FN, hay papel                                  | Falso negativo en QR central                              |
| 25  | front, conf 0.80 | ✓             | no             | -                   | text_center ⚠ FN, SIM cubre texto                     | La SIM no se ha podido detectar                           |
| 28  | back, conf 0.74  | ✓             | no             | -                   | qr_left FN, hay papel rosa                            | Falso negativo en QR izquierdo                            |
| 31  | front, conf 0.57 | ✗             | sí             | hologram_photo:0.66 | hologram_photo ✗, caja fuera de la INE                | Side parece back, y el ROI de holograma está en el fondo  |
| 32  | front, conf 0.95 | ✓             | no             | -                   | text_bottom FN y hologram_photo FN, hay cinta o papel | Falso negativo fuerte en zonas tapadas                    |
