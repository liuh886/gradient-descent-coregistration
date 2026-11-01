# Gradient Descent Coregistration (GDC)

**Status: extraction in progress — no released API yet (v0.1 in preparation).**

GDC is a point-to-surface coregistration method for DEM quality control with
sparse trusted points. It finds the translation (dx, dy, dz) between a DEM and
a set of reference points by shifting the points — not resampling the DEM on
every iteration — sampling the DEM at the shifted points, and minimising a
robust statistic of the elevation differences (dh).

## Where it is aimed

Sparse-point DEM quality control first: a few hundred to a few thousand
points (e.g. ICESat-2) validating a DEM, especially at fine resolutions where
Nuth and Kääb (2011) coregistration becomes slow or fails at small sample
sizes. The primary workflow is GeoDataFrame points + a rasterio DEM; xarray
is supported second.

## What GDC is — and is not

- GDC solves only the last step of a coregistration chain: the convergence of
  the final 2–3 parameters (dx, dy, dz) and their uncertainty. Search,
  matching and coarse alignment are upstream problems and are not included.
- The core evidence is the residual distribution after convergence: the
  median of dh is the systematic bias, the NMAD is the random-noise level,
  and the spatial structure of the residuals shows what a 2–3 parameter model
  cannot explain. If the residuals do not converge, the problem is upstream.
- "Gradient descent" is the established name of the method (see the blog
  posts below). The optimiser actually used is recorded in every result
  rather than hidden behind the name — the original code contains both a
  compass-search variant and a SciPy L-BFGS-B variant.

## Planned form

A dependency-light core (NumPy/SciPy) with adapters, not core dependencies,
for xarray, rasterio and geopandas. Python import name: `gdc`.
DEM-to-DEM coregistration reduces to the same point routine by sampling
points from the reference DEM.

## Background

- Method comparison vs xDEM's NuthKaab, point interpolation vs footprint
  zonal statistics: <https://zhihaol.eu.org/blog/2023/gdc/>
- Applying GDC to compare DTM1, DTM10, Copernicus GLO-30 and FABDEM against
  ICESat-2: <https://zhihaol.eu.org/blog/2023/dataset/>
- Paper: Liu, Filhol & Treichler, *Retrieving snow depth distribution by
  downscaling ERA5 Reanalysis with ICESat-2 laser altimetry*, Cold Regions
  Science and Technology, 2025 — <https://doi.org/10.1016/j.coldregions.2025.104580>
- Open dataset (ICESat-2 ATL08 vs the four DEMs above):
  <https://doi.org/10.5281/zenodo.10048875>
- Comparator: xDEM (NuthKaab / ICP) — <https://github.com/GlacioHack/xdem>

## Provenance

The method was originally developed inside the `xsnow` toolbox of
[liuh886/subgrid_snow](https://github.com/liuh886/subgrid_snow), the research
code for the snow-depth paper above. That repository stays as the paper's
research code; this repository is the extracted, standalone home of GDC,
created on 2026-11-01 by renaming the former empty `liuh886/xsnow`
placeholder repository.

## License

MIT — see [LICENSE](LICENSE).
