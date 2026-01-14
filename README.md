# METISMap (temporary SunPy Map subclass)

> **Disclaimer (temporary / evolving code)**  
> This repository provides a **custom** `METISMap` class (a subclass of `sunpy.map.GenericMap`) for **Solar Orbiter / Metis** FITS products.  
> It is currently a **temporary, notebook-oriented** implementation used for internal development and tutorials.  
> The plan is to **clean it up and integrate it into the SunPy ecosystem** as soon as possible (better API stability, packaging, tests, and colormap registration).  
> Until then, expect **API changes** and **refactoring**.

---

## What this is

`METISMap` extends SunPy’s `GenericMap` to make Metis L2 images easier to work with in Python notebooks.

Compared to a plain `GenericMap`, it adds:

- **Metis-aware product typing** (e.g. VL-TB, VL-PB, UV, Stokes, Pixel Quality, Absolute Error) based on FITS metadata.
- **Convenient default contrast handling** for plotting (percentile clipping tuned per product type).
- **Metis-specific masking utilities**, e.g. masking the internal/external occulter regions and masking bad pixels using the Pixel Quality matrix.
- **Helper functions for Cartesian ↔ polar transformations**, useful for radial profiles, azimuthal plots, and polar representations of the corona.
- **Temporary colormap support** for Metis products (until these colormaps are properly registered upstream).

---

## Intended usage

This code is primarily intended to support the workflow shown in the companion notebooks:

- load Metis FITS with `sunpy.map.Map(...)`
- get `METISMap` objects automatically (when metadata matches Metis)
- apply optional masking (`mask_occs`, `mask_bad_pix`)
- plot with sensible defaults
- optionally convert to/from polar representation for analysis

---

## Quick start (notebook-style)

```python
import sunpy.map
from metismap import METISMap  # or: from metismap.map import METISMap

# Load a Metis FITS file (often multi-HDU: image + quality + error)
maps = sunpy.map.Map("solo_L2_metis-vl-image_YYYYMMDDThhmmss_V01.fits")

# Typical pattern: science image + Pixel Quality (PQ) + Absolute Error (AE)
metis_img = maps[0]
metis_pq  = maps[-2]
metis_ae  = maps[-1]

# Mask occulter regions (optional)
metis_img.mask_occs()
metis_pq.mask_occs()
metis_ae.mask_occs()

# Mask bad pixels using PQ matrix (optional)
metis_img.mask_bad_pix(metis_pq.data)

# Plot with Metis defaults (temporary colormaps + contrast handling)
metis_img.peek()
```

# Polar conversion helpers

This repository also provides utilities to work in polar coordinates (Rsun, position angle):

- get_img_arr_polar(...): create a polar image from a METISMap

- cart_to_polar(...), polar_to_cart(...): lower-level transforms

- set_rsun_pa_axes(...): convenience helper for plotting polar axes

# License and credits

Authors / Maintainers: Aleksandr Burtovoi (UNIFI) & Giovanna Jerse (INAF-OATs)

License: GPL (see LICENSE)

Additional credits for colormap prototyping and discussion: V. Andretta, A. Liberatore, and collaborators.
