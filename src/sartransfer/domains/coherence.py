"""Interferometric coherence for the forest chips (notebooks 04a and 04b).

The global seasonal Sentinel-1 interferometric coherence and backscatter data set (Kellndorfer et al.,
Scientific Data 9, 73, 2022, doi 10.1038/s41597-022-01189-6, CC BY 4.0), public on AWS, no account.
Read on 2026-10-04 from the files themselves -
- one-degree tiles of 1200 x 1200 px at 3 arcsec in EPSG:4326, named by their upper left corner
  (N48W001 covers 47 to 48 N and 1 W to 0, S25E113 covers 26 to 25 S and 113 to 114 E)
- <tile>_<season>_vv_COH12.tif, the 12-day VV coherence x 100 as uint8, 0 = no data, LZW strips of
  6 rows; <tile>_summer_vv_AMP.tif, VV backscatter as uint16, 0 = no data, strips of 3 rows
- the seasons winter, spring, summer and fall of December 2019 to November 2020
- tiles over open water are not in the bucket and count as no data here.
A chip reads only the windows it covers, through GDAL's HTTP reader (range requests), never whole
tiles - the 5,670 forest chips touch 4,007 tiles, whole tiles would be about 24 GB.

A chip is 128 x 128 px of 10 m, north up. ForTy gives one latitude and longitude per chip; whether it
is the centre or the upper left corner is decided on the data by `placement_check` (section
15). The value at a chip pixel is interpolated bilinearly from the 3 arcsec grid at the pixel's
latitude and longitude, the weights of no-data neighbours dropped.
"""

from __future__ import annotations

import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BUCKET = "https://sentinel-1-global-coherence-earthbigdata.s3.us-west-2.amazonaws.com/data/tiles"
SOURCE = {"name": "Global Seasonal Sentinel-1 Interferometric Coherence and Backscatter Data Set",
          "authors": "Kellndorfer et al.", "paper": "Scientific Data 9, 73 (2022), doi 10.1038/s41597-022-01189-6",
          "licence": "CC BY 4.0", "bucket": "s3://sentinel-1-global-coherence-earthbigdata (us-west-2)"}
TILE_PX = 1200                       # pixels per degree, 3 arcsec
SEASONS = ("winter", "spring", "summer", "fall")
COHERENCE, BACKSCATTER = "vv_COH12", "vv_AMP"
M_PER_DEG = 111_320.0                # metres per degree of latitude, enough at 90 m pixels
CHIP_PX, PIXEL_M = 128, 10.0
ANCHORS = ("centre", "corner")
WORKERS = 32                         # chips at once. The wall is latency (about 10 small range requests per
                                     # chip, each a round trip to us-west-2), not bandwidth or CPU; S3 serves
                                     # thousands of GET per second per prefix, so 32 threads stay far below that
                                     # and keep the CPU runtime's cores busy decoding the small LZW strips
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2", "GDAL_HTTP_MULTIRANGE": "YES",
            "VSI_CACHE": "TRUE"}


def tile_of(lat: float, lon: float) -> tuple[int, int]:
    """The (top, left) degree of the tile that holds this point."""
    return math.floor(lat) + 1, math.floor(lon)


def tile_name(top: int, left: int) -> str:
    """N48W001 for top 48, left -1; S25E113 for top -25, left 113."""
    return f"{'N' if top >= 0 else 'S'}{abs(top):02d}{'E' if left >= 0 else 'W'}{abs(left):03d}"


def chip_lat_lon(lat0: float, lon0: float, anchor: str = "centre", size: int = CHIP_PX,
                 pixel_m: float = PIXEL_M, shift_m: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of every pixel centre of a north-up chip, each (size, size).

    anchor "centre" - (lat0, lon0) is the middle of the chip; "corner" - its upper left corner.
    shift_m (north, east) moves the whole chip, for the floor of `placement_check`.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    idx = np.arange(size, dtype=np.float64)
    north = ((size - 1) / 2 - idx) * pixel_m if anchor == "centre" else -(idx + 0.5) * pixel_m
    east = (idx - (size - 1) / 2) * pixel_m if anchor == "centre" else (idx + 0.5) * pixel_m
    north, east = north + shift_m[0], east + shift_m[1]
    lat = np.broadcast_to(lat0 + north[:, None] / M_PER_DEG, (size, size))
    lon = np.broadcast_to(lon0 + east[None, :] / (M_PER_DEG * max(math.cos(math.radians(lat0)), 1e-6)), (size, size))
    return lat.copy(), lon.copy()


class Tiles:
    """Windows of the bucket's tiles. `root` is the bucket URL (read through /vsicurl/) or a local folder
    with the same layout (tests). A tile that is not there gives None."""

    def __init__(self, root: str | Path = BUCKET):
        self.root = str(root)
        self.remote = self.root.startswith("http")
        self._exists: dict[str, bool] = {}
        self._lock = threading.Lock()
        self.reads = 0

    def path(self, name: str, kind: str) -> str:
        return f"{self.root}/{name}/{name}_{kind}.tif"

    def exists(self, name: str, kind: str) -> bool:
        key = f"{name}_{kind}"
        with self._lock:
            if key in self._exists:
                return self._exists[key]
        if self.remote:
            import requests

            for attempt in range(5):
                try:
                    code = requests.head(self.path(name, kind), timeout=30).status_code
                except requests.RequestException:
                    code = None
                if code in (200, 403, 404):          # S3 answers 403 for a missing key without list rights
                    break
                time.sleep(2 * (attempt + 1))
            else:
                raise OSError(f"no answer for {self.path(name, kind)} after 5 tries")
            there = code == 200
        else:
            there = Path(self.path(name, kind)).is_file()
        with self._lock:
            self._exists[key] = there
        return there

    def window(self, name: str, kind: str, r0: int, r1: int, c0: int, c1: int) -> np.ndarray | None:
        """Rows r0..r1-1 and columns c0..c1-1 of one tile, as stored; None when the tile is not there."""
        import rasterio
        from rasterio.windows import Window

        if not self.exists(name, kind):
            return None
        path = ("/vsicurl/" + self.path(name, kind)) if self.remote else self.path(name, kind)
        with rasterio.Env(**GDAL_ENV), rasterio.open(path) as src:
            out = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
        with self._lock:
            self.reads += 1
        return out


def sample(tiles: Tiles, kind: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """The values of one layer at these points, bilinear on the 3 arcsec grid, float32, nan for no data.

    The window covering the points is put together from every tile it touches (a chip near a tile
    border reads two or four), then each point takes the four grid cells around it with the weights
    of no-data cells (0) dropped; nan where all four are no data or the tile is not there."""
    fr = (90.0 - lat) * TILE_PX - 0.5                 # global row and column of the points, cell centres at .0
    fc = (lon + 180.0) * TILE_PX - 0.5
    r0, c0 = int(math.floor(fr.min())), int(math.floor(fc.min()))
    r1, c1 = int(math.floor(fr.max())) + 2, int(math.floor(fc.max())) + 2
    grid = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
    for tr in range(r0 // TILE_PX, (r1 - 1) // TILE_PX + 1):
        for tc in range(c0 // TILE_PX, (c1 - 1) // TILE_PX + 1):
            gr0, gr1 = max(r0, tr * TILE_PX), min(r1, (tr + 1) * TILE_PX)
            gc0, gc1 = max(c0, tc * TILE_PX), min(c1, (tc + 1) * TILE_PX)
            got = tiles.window(tile_name(90 - tr, tc - 180), kind, gr0 - tr * TILE_PX, gr1 - tr * TILE_PX,
                               gc0 - tc * TILE_PX, gc1 - tc * TILE_PX)
            if got is not None:
                block = got.astype(np.float32)
                block[got == 0] = np.nan
                grid[gr0 - r0:gr1 - r0, gc0 - c0:gc1 - c0] = block
    i, j = np.floor(fr).astype(int) - r0, np.floor(fc).astype(int) - c0
    di, dj = (fr - np.floor(fr)).astype(np.float32), (fc - np.floor(fc)).astype(np.float32)
    total, weight = np.zeros(lat.shape, np.float32), np.zeros(lat.shape, np.float32)
    for oi, oj, w in ((0, 0, (1 - di) * (1 - dj)), (0, 1, (1 - di) * dj), (1, 0, di * (1 - dj)), (1, 1, di * dj)):
        v = grid[i + oi, j + oj]
        ok = np.isfinite(v)
        total += np.where(ok, v * w, 0.0)
        weight += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / weight, np.nan).astype(np.float32)


def chip_coherence(tiles: Tiles, lat0: float, lon0: float, anchor: str) -> np.ndarray:
    """(128, 128) float32, the mean 12-day VV coherence of the four seasons, 0 to 1, nan for no data."""
    lat, lon = chip_lat_lon(lat0, lon0, anchor)
    seasons = np.stack([sample(tiles, f"{s}_{COHERENCE}", lat, lon) for s in SEASONS])
    with warnings.catch_warnings():                   # a pixel without any season is nan, as meant
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmean(seasons, axis=0) / 100.0).astype(np.float32)


def to_u8(coherence: np.ndarray) -> np.ndarray:
    """0 to 1 as uint8 0 to 255 (the decoder's image skip is divided by 255), no data as 0."""
    return np.round(np.clip(np.nan_to_num(coherence, nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)


def _smooth(x: np.ndarray, k: int = 9) -> np.ndarray:
    """Box mean over k x k pixels (90 m for 10 m pixels), nan where the box holds no finite value."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0).astype(np.float64)
    pad = k // 2

    def box(a):
        a = np.pad(a, pad, mode="edge").cumsum(0).cumsum(1)
        a = np.pad(a, ((1, 0), (1, 0)))
        return a[k:, k:] - a[:-k, k:] - a[k:, :-k] + a[:-k, :-k]

    n = box(ok.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, box(v) / n, np.nan)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def placement_check(tiles: Tiles, vv_db: np.ndarray, lats, lons, n: int = 300, seed: int = 0,
                    workers: int = WORKERS) -> dict:
    """Which reading of the chip's latitude and longitude fits the data (no labels).

    For `n` chips drawn with `seed` - the chip's own VV in dB, smoothed to 90 m, against the data set's
    summer VV backscatter (log10 of the stored value) read under each anchor, the Pearson correlation
    per chip. Next to them the centre reading shifted 1.3 km north and east, as the floor. Returns
    {"medians" - {reading - median}, "chips" - n, "anchor" - "centre" or "corner", the higher median}.
    """
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(lats), min(n, len(lats)), replace=False))
    readings = {"centre": ("centre", (0.0, 0.0)), "corner": ("corner", (0.0, 0.0)),
                "centre shifted 1.3 km": ("centre", (1300.0, 1300.0))}

    def one(k):
        vv = _smooth(np.asarray(vv_db[k], np.float64))
        out = {}
        for name, (anchor, shift) in readings.items():
            lat, lon = chip_lat_lon(float(lats[k]), float(lons[k]), anchor, shift_m=shift)
            amp = sample(tiles, f"summer_{BACKSCATTER}", lat, lon)
            out[name] = _corr(vv, np.log10(amp))
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_chip = list(pool.map(one, pick))
    medians = {name: float(np.nanmedian([c[name] for c in per_chip])) for name in readings}
    anchor = max(ANCHORS, key=lambda a: medians[a])
    return {"medians": medians, "chips": len(pick), "anchor": anchor}


def build(tiles: Tiles, lats, lons, anchor: str, workers: int = WORKERS, every: int = 500) -> tuple[np.ndarray, dict]:
    """The coherence of every chip as uint8 (N, 128, 128) (`to_u8`) and a record - chips, chips without
    any coherence, the share of pixels without, seconds, chips per second, tile windows read."""
    n = len(lats)
    out = np.zeros((n, CHIP_PX, CHIP_PX), np.uint8)
    empty, missing_px = [0], [0]
    t0, reads0 = time.perf_counter(), tiles.reads
    lock = threading.Lock()

    def one(k):
        coh = chip_coherence(tiles, float(lats[k]), float(lons[k]), anchor)
        out[k] = to_u8(coh)
        bad = int((~np.isfinite(coh)).sum())
        with lock:
            missing_px[0] += bad
            empty[0] += bad == coh.size

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one, range(n)):
            done += 1
            if every and done % every == 0:
                rate = done / max(time.perf_counter() - t0, 1e-9)
                print(f"  {done:,} of {n:,} chips, {rate:.1f} chips per second", flush=True)
    seconds = time.perf_counter() - t0
    return out, {"chips": n, "chips_without_coherence": empty[0],
                 "pixels_without_coherence_share": missing_px[0] / max(n * CHIP_PX * CHIP_PX, 1),
                 "seconds": seconds, "chips_per_second": n / max(seconds, 1e-9),
                 "tile_windows_read": tiles.reads - reads0}
