## Carpeta npy

Ruta: `data/npy`

Contiene arreglos NumPy listos para analisis numerico:

- `pressures.npy`: niveles de presion, shape `(43,)`
- `temperature.npy`: temperatura por perfil y nivel, shape `(2311, 43)`
- `humidity.npy`: H2O por perfil y nivel, shape `(2311, 43)`
- `ozone.npy`: O3 por perfil y nivel, shape `(2311, 43)`
- `ts.npy`: temperatura superficial, shape `(2311,)`
- `ps.npy`: presion superficial, shape `(2311,)`

## Carpeta csv

Ruta: `data/csv`

Contiene tablas para inspeccion y uso en Excel/R/BI:

- `metadata.csv`: metadatos por perfil (`latitude`, `longitude`, `ific`, `date`)
- `tigr_wide.csv`: tabla completa en formato ancho (una fila por perfil)
- `tigr_wide.txt`: mismo contenido en texto con columnas alineadas