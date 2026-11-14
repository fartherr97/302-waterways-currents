# 302 Waterways currents

Tidal and ocean surface currents for the 302 Waterways Currents layer, served by GitHub Pages
at `https://fartherr97.github.io/302-waterways-currents/`.

- Source: NOAA's Operational Forecast Systems (National Ocean Service), on AWS Open Data
  (bucket `noaa-nos-ofs-pds`), public domain. The surface current, hourly from the newest run
  to about 48 hours ahead, from:
  - Chesapeake Bay (CBOFS) and Delaware Bay (DBOFS), with the coastal ocean off them
  - New York Harbor (NYOFS and its fine grid), with the Hudson, the Harlem and the East River
  - the Gulf of Maine (GoMOFS)
  - Tampa Bay (TBOFS)
  - the northern Gulf of America from Texas to the Florida panhandle (NGOFS2)
  - the St. Johns River (SJROFS)
  - Lakes Erie, Michigan and Huron, Superior, and Ontario (LEOFS, LMHOFS, LSOFS, LOOFS)
- Station index: NOAA CO-OPS tidal current prediction stations from Maine to Texas
  (`stations.json`), public domain. The app asks CO-OPS itself for a station's predictions.
- Refreshed four times a day (02:40, 08:40, 14:40 and 20:40 UTC), a little after each model run
  is out. Each run replaces the whole site, so only the current run's hours are here.

## Files

- `manifest.json`: the hours (`{ t, key, run, tiles, models }`), the tile levels, the scale and
  offset, and which model runs went in. The app reads it first.
- `<YYYYMMDDHH>/<tile>.png`: one folder per hour. Tiles in three levels:
  - `f<x>_<y>`: 0.005° cells (about 500 m), 1° tiles, where some model is finer than about 1.6 km
  - `m<x>_<y>`: 0.02° cells (about 2 km), 4° tiles, wherever a model covers
  - `o`: one 0.05° overview of the whole area (98°W–66°W, 24°N–50°N)
- `stations.json`: `{ fields: [id, name, lat, lng, bin, type, depth], rows: [...] }`, one depth
  bin per station (the one nearest the surface).

## Format

Each tile is an RGBA PNG, rows north → south, the same as the app's wind tiles:

- R, G: the current toward the east and the north, as byte = m/s × 40 + 128 (±3.2 m/s, 0.05 kn steps)
- B: how far inside a model the cell is, 0 at its open boundary at sea → 255, so the app fades
  the field out where a model stops in open water
- A: 255 where a model has water, 0 on land and outside every model

## How it's made

`scripts/build.py`, run by `.github/workflows/currents.yml`. Only the surface layer of u and v is
read from each model file, a few byte ranges at a time; nothing is downloaded whole. Curvilinear
grids (ROMS) are turned from grid-relative to east/north with their `angle`; every model
(curvilinear or the unstructured FVCOM meshes) is laid on a plain latitude/longitude grid through
a triangle mesh, with land masked. Where models overlap, the more local one is laid over the other.

    pip install numpy scipy h5py
    python3 -m unittest discover -s scripts -p 'test_*.py'
    python3 scripts/build.py --out site --workers 4

The same scripts are in the app repository under `scripts/currents`. The app uses this site as
its default currents address (`VITE_CURRENTS_URL` overrides it).
