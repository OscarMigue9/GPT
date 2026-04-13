## Estructura general

El dataset esta dividido en dos formatos:

- `data/npy`: arreglos NumPy para calculo numerico rapido.
- `data/csv`: tablas para inspeccion tabular y exportacion.

## Contenido en `data/npy`

- `pressures.npy`:
  - Niveles verticales de presion atmosferica.
  - Shape: `(43,)`.
  - Unidad esperada: `mb`.

- `temperature.npy`:
  - Temperatura por perfil y nivel.
  - Shape: `(2311, 43)`.
  - Unidad: `K`.

- `humidity.npy`:
  - Humedad (fraccion de masa `kg/kg`, equivalente numericamente a `g/g`).
  - Shape: `(2311, 43)`.

- `ozone.npy`:
  - Ozono en fraccion de masa (`g/g`).
  - Shape: `(2311, 43)`.

- `ts.npy`:
  - Temperatura superficial por perfil.
  - Shape: `(2311,)`.

- `ps.npy`:
  - Presion superficial por perfil.
  - Shape: `(2311,)`.

## Contenido en `data/csv`

- `metadata.csv`:
  - Metadatos por perfil (por ejemplo: latitud, longitud, fecha, clase/indice atmosferico).

- `tigr_wide.csv`:
  - Representacion ancha de perfiles y variables en una tabla.

- `tigr_wide.txt`:
  - Version texto del formato ancho.

## Dataset filtrado cloud-free (`RHmax < 96%`)

El notebook calcula humedad relativa por nivel y genera una mascara por perfil:

- Criterio: perfil valido si `RHmax < 96%`.
- Resultado: subconjunto cloud-free guardado en `data/npy`.

Archivos generados:

- `temperature_cf96.npy`
- `humidity_cf96.npy`
- `ozone_cf96.npy`
- `rh_cf96.npy`
- `profile_index_cf96.npy`

## Conversiones usadas en las graficas finales

- H2O:
  - De `kg/kg` a `g/kg`: multiplicar por `1000`.

- O3:
  - De fraccion de masa a `ppmv`:
  - `x_ppmv = w * (M_air / M_o3) * 1e6`
  - con `M_air = 28.9647` y `M_o3 = 48.0`.
