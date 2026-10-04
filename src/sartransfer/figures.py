"""Figures of the results - prediction GIFs, a feature-map GIF, the results chart and the architecture picture.

Notebook 05_figures makes them on a GPU from what the scoring saved - the decoder heads of seed 0 and the
settings of every run (its __info.json, the channel statistics of the training split and of the encoder
input); nothing is trained or tuned here. The chips are picked by a rule on the labels alone
(`pick_chips`), never by how well a model does on them. Matplotlib figures are drawn on the Agg canvas
directly, no window and no notebook backend. No colon in any title or caption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IGNORE = 255
DARK = "#3a3a3a"
PALETTES = {
    "glacial_lakes": {0: DARK, 1: "#2f7fd8"},
    "snow": {0: DARK, 1: "#eef4ff"},
    "glavitu_alp": {0: DARK, 1: "#59c9e0"},
    "forty_v1": {0: "#1b7837", 1: "#7fbc41", 2: "#c2a54a", 3: "#d9ef8b", 4: "#d73027", 5: "#4575b4",
                 6: "#e0f3f8", 7: "#a6761d"},
}
TITLES = {"glacial_lakes": "Glacial lakes", "snow": "Snow", "forty_v1": "Forest types", "glavitu_alp": "Alpine glaciers"}
SCORE_NAMES = {"glacial_lakes": "lake IoU", "snow": "snow IoU", "forty_v1": "mIoU", "glavitu_alp": "glacier IoU"}
STEP_SECONDS, LAST_SECONDS = 1.8, 4.5
FRAME_INCHES, FRAME_DPI = (12.0, 6.75), 100       # 1200 x 675 px, the same for every frame of a GIF


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------ chips and their pictures

def class_shares(label: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """The share of each class among the labelled pixels of one chip, and the labelled share of the chip."""
    lab = np.asarray(label)
    ok = lab != IGNORE
    n = int(ok.sum())
    counts = np.bincount(lab[ok].ravel().astype(np.int64), minlength=k)[:k].astype(np.float64)
    return (counts / n if n else counts), n / lab.size


def pick_chips(labels: np.ndarray, regions, k: int, n: int = 2, min_labelled: float = 0.9) -> list[int]:
    """`n` chips by a rule on the labels alone - a labelled share of at least `min_labelled`, or of
    `min_labelled` times the best chip's share where every chip carries unlabelled padding (the alpine tiles
    are at most 88 percent labelled), then the most even class mix (the highest entropy of the class shares),
    each next chip from a region not taken yet when one is left. Ties go to the lower index. Returns the
    indices in the order picked."""
    regions = np.asarray(regions).astype(str)
    per_chip = [class_shares(labels[i], k) for i in range(len(labels))]
    best = max((labelled for _, labelled in per_chip), default=0.0)
    threshold = min(min_labelled, min_labelled * best)
    entropy = np.full(len(labels), -1.0)
    for i, (shares, labelled) in enumerate(per_chip):
        if labelled >= threshold:
            p = shares[shares > 0]
            entropy[i] = float(-(p * np.log(p)).sum())
    order = sorted((i for i in range(len(labels)) if entropy[i] >= 0), key=lambda i: (-entropy[i], i))
    picked: list[int] = []
    for i in order:
        if len(picked) == n:
            break
        if regions[i] in {regions[j] for j in picked} and any(regions[j] not in {regions[q] for q in picked}
                                                               for j in order if j not in picked):
            continue
        picked.append(i)
    return picked


def radar_rgb(image: np.ndarray) -> np.ndarray:
    """(C, H, W) radar in dB -> (H, W, 3) uint8 - red VV, green VH, blue VV - VH, each stretched between its
    2nd and 98th percentile inside the chip; no data black."""
    x = np.asarray(image, np.float32)
    channels = [x[0], x[1], x[0] - x[1]]
    out = np.zeros((*x.shape[1:], 3), np.uint8)
    for c, v in enumerate(channels):
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        lo, hi = np.percentile(v[ok], [2, 98])
        s = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        out[..., c] = np.where(ok, np.round(s * 255), 0).astype(np.uint8)
    return out


def label_rgb(label: np.ndarray, palette: dict) -> np.ndarray:
    """(H, W) class ids -> (H, W, 3) uint8 in the dataset's colours, unlabelled black."""
    lab = np.asarray(label)
    out = np.zeros((*lab.shape, 3), np.uint8)
    for cls, colour in palette.items():
        out[lab == cls] = _rgb(colour)
    return out


def chip_score(pred: np.ndarray, label: np.ndarray, k: int) -> float:
    """The score of one chip on its labelled pixels - the IoU of class 1 for two classes, else the mean IoU over
    the classes in the truth or the prediction. nan when there is nothing to score."""
    lab, pr = np.asarray(label), np.asarray(pred)
    ok = lab != IGNORE
    t, p = lab[ok].astype(np.int64), pr[ok].astype(np.int64)
    classes = [1] if k == 2 else sorted(set(np.unique(t)) | set(np.unique(p)))
    ious = []
    for c in classes:
        inter, union = int(((t == c) & (p == c)).sum()), int(((t == c) | (p == c)).sum())
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")


def predict(head, feats: np.ndarray, skip_u8: np.ndarray, device: str = "cuda", batch: int = 4) -> np.ndarray:
    """The decoder's class map of each chip, (N, H, W) uint8 - as the scoring computes it (features (N, g, g, D),
    the image skip (N, H, W) or (N, 2, H, W) uint8, bfloat16 autocast on CUDA, argmax)."""
    import torch

    head = head.to(device).eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        for i in range(0, len(feats), batch):
            f = torch.from_numpy(np.asarray(feats[i:i + batch])).to(device).float().permute(0, 3, 1, 2)
            img = torch.from_numpy(np.asarray(skip_u8[i:i + batch]))
            img = (img[:, None] if img.ndim == 3 else img).to(device).float() / 255.0
            out.append(head(f, img).float().argmax(1).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ frames and GIFs

def _figure():
    from matplotlib.figure import Figure

    fig = Figure(figsize=FRAME_INCHES, dpi=FRAME_DPI, facecolor="white")
    return fig


def figure_array(fig) -> np.ndarray:
    """A matplotlib figure as (H, W, 3) uint8, drawn on the Agg canvas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _legend(fig, palette: dict, class_names: list[str]) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=palette[c], edgecolor="#777777", label=class_names[c].replace("_", " "))
               for c in range(len(class_names)) if c in palette]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), frameon=False, fontsize=10)


def _show(ax, picture, title):
    ax.imshow(picture, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def prediction_frames(title: str, chips: list[dict], models: list[str], palette: dict, class_names: list[str],
                      score_name: str) -> tuple[list[np.ndarray], list[float]]:
    """The frames of one prediction GIF and their seconds.

    chips  dicts with "radar" and "truth" (H, W, 3) uint8, "pred" {model - (H, W, 3) uint8}, "score" {model -
           float} and "label" (the caption of the chip)
    The GIF steps through the radar, the truth and each model's prediction with its score, the chips side by
    side, and ends on all panels together."""
    steps = [("radar", "Sentinel-1 radar (VV, VH, VV-VH)"), ("truth", "Truth"), *[(m, m) for m in models]]
    frames, seconds = [], []
    for key, name in steps:
        fig = _figure()
        fig.suptitle(f"{title} - {name}", fontsize=15, y=0.97)
        for j, chip in enumerate(chips):
            ax = fig.add_axes([0.03 + j * 0.49, 0.12, 0.45, 0.76])
            picture = chip[key] if key in ("radar", "truth") else chip["pred"][key]
            caption = chip["label"] if key in ("radar", "truth") else f"{score_name} {chip['score'][key]:.2f}"
            _show(ax, picture, caption)
        _legend(fig, palette, class_names)
        frames.append(figure_array(fig))
        seconds.append(STEP_SECONDS)
    fig = _figure()
    fig.suptitle(f"{title} - all together", fontsize=15, y=0.98)
    cols = 2 + len(models)
    for r, chip in enumerate(chips):
        for c, (key, name) in enumerate(steps):
            ax = fig.add_axes([0.01 + c * (0.98 / cols), 0.53 - r * 0.42, 0.98 / cols - 0.01, 0.36])
            picture = chip[key] if key in ("radar", "truth") else chip["pred"][key]
            short = name.split(" (")[0]
            label = short if key in ("radar", "truth") else f"{short}, {chip['score'][key]:.2f}"
            ax.imshow(picture, interpolation="nearest")
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    _legend(fig, palette, class_names)
    frames.append(figure_array(fig))
    seconds.append(LAST_SECONDS)
    return frames, seconds


def feature_frames(rows: list[dict], models: list[str]) -> tuple[list[np.ndarray], list[float]]:
    """The frames of the feature-map GIF - one row per dataset ("title", "radar", "truth", "features" {model -
    (h, w, 3) uint8, one PCA per image}), the steps radar, truth and each model, then all together."""
    steps = [("radar", "Sentinel-1 radar"), ("truth", "Truth"), *[(m, m) for m in models]]
    frames, seconds = [], []
    n = len(rows)
    for key, name in steps:
        fig = _figure()
        fig.suptitle(f"What each model sees - {name}" if key not in ("radar", "truth") else name, fontsize=15, y=0.97)
        for j, row in enumerate(rows):
            ax = fig.add_axes([0.01 + j * (0.98 / n), 0.08, 0.98 / n - 0.01, 0.78])
            picture = row[key] if key in ("radar", "truth") else row["features"][key]
            _show(ax, picture, row["title"])
        frames.append(figure_array(fig))
        seconds.append(STEP_SECONDS)
    fig = _figure()
    fig.suptitle("What each model sees - feature colours, one PCA per image", fontsize=15, y=0.98)
    cols = len(steps)
    for r, row in enumerate(rows):
        for c, (key, name) in enumerate(steps):
            ax = fig.add_axes([0.08 + c * (0.91 / cols), 0.9 - (r + 1) * (0.86 / n), 0.91 / cols - 0.008, 0.86 / n - 0.03])
            picture = row[key] if key in ("radar", "truth") else row["features"][key]
            ax.imshow(picture, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name.split(" (")[0], fontsize=10)
            if c == 0:
                ax.set_ylabel(row["title"], fontsize=10)
    frames.append(figure_array(fig))
    seconds.append(LAST_SECONDS)
    return frames, seconds


def save_gif(frames: list[np.ndarray], path: str | Path, seconds: list[float]) -> Path:
    """The frames as one looping GIF, each with its seconds, colours reduced per frame (adaptive palette)."""
    from PIL import Image

    if not frames or len(frames) != len(seconds):
        raise ValueError(f"{len(frames)} frames and {len(seconds)} durations")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"the frames differ in size, {shapes}")
    images = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE, colors=255) for f in frames]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=[int(s * 1000) for s in seconds],
                   loop=0, optimize=True, disposal=1)
    return path


# ------------------------------------------------------------------ static pictures

def results_chart(table, path: str | Path, models: list[str], datasets: list[str], colours: dict) -> Path:
    """Grouped bars of the decoder scores on the validation splits, the seed SD as error bars.

    table  rows with dataset, model, mean, sd (the stage A table of the scoring); `models` in bar order,
           `datasets` in group order, `colours` {model - colour}."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.2), dpi=150, facecolor="white")
    ax = fig.add_axes([0.06, 0.2, 0.92, 0.7])
    width = 0.8 / len(models)
    for m, model in enumerate(models):
        means, sds = [], []
        for d in datasets:
            row = table[(table["dataset"] == d) & (table["model"] == model)]
            means.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
            sds.append(float(row["sd"].iloc[0]) if len(row) else np.nan)
        x = np.arange(len(datasets)) - 0.4 + width * (m + 0.5)
        ax.bar(x, means, width * 0.92, yerr=sds, capsize=2, color=colours[model], label=model,
               edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([f"{TITLES[d]}\n({SCORE_NAMES[d]})" for d in datasets], fontsize=11)
    ax.set_ylabel("validation score, decoder, mean of 3 seeds", fontsize=11)
    ax.set_title("Radar segmentation with frozen features - students against the references", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(models), frameon=False, fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def architecture_picture(path: str | Path) -> Path:
    """The training set-up as one picture - the inputs, the student with its heads, the four teacher slots,
    and which teachers each of the students F, N and C copies."""
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = Figure(figsize=(12, 6.6), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, w, h, text, colour, size=10, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=colour, edgecolor="#444444", linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, wrap=True)

    def arrow(x0, y0, x1, y1, colour="#555555", style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, color=colour,
                                     linewidth=1.1))

    ax.text(6, 6.3, "Distillation - one radar and optical student copies four frozen teachers", ha="center",
            fontsize=14, weight="bold")
    inputs = [("Sentinel-1\nVV, VH in dB", 4.4), ("Sentinel-2\n12 bands", 3.1), ("Aerial\nRGB + NIR, 0.4 m", 1.8)]
    for text, y in inputs:
        box(0.3, y, 1.9, 0.95, text, "#e8eef7")
        arrow(2.2, y + 0.47, 3.0, 3.6)
    box(3.0, 2.6, 2.4, 2.0, "Student\nViT-B/16, 88.8 M\nthree input layers\n256 px crops", "#fde9c9", 11, "bold")
    box(5.9, 3.4, 1.7, 0.75, "summary heads\ncosine loss", "#f3f3f3", 9)
    box(5.9, 2.5, 1.7, 0.75, "spatial heads\nPHI-S MSE", "#f3f3f3", 9)
    arrow(5.4, 3.75, 5.9, 3.75)
    arrow(5.4, 2.9, 5.9, 2.9)
    teachers = [("1 SigLIP2 so400m\nFixRes or NaFlex, summary only", "#dcefdc", 5.0),
                ("2 DINOv3 ViT-L SAT-493M\nsatellite, dense", "#dcefdc", 3.9),
                ("3 PE-Spatial-L14\nread at 224 px", "#dcefdc", 2.8),
                ("4 TerraFM-B, Sentinel-1 layer\nreads the radar itself", "#d6e6fb", 1.7)]
    for text, colour, y in teachers:
        box(8.7, y, 3.0, 0.85, text, colour, 9)
        arrow(8.7, y + 0.42, 7.6, 3.3, colour="#888888")              # the teacher gives the target
    ax.text(10.2, 0.95, "slots 1 to 3 read the optical image of the same place\nslot 4 only rows that hold radar",
            ha="center", fontsize=9, color="#444444")
    ax.text(4.2, 1.6, "students\nF  slots 1 (FixRes), 2, 3, 4\nN  slots 1 (NaFlex), 2, 3, 4\nC  slots 1 (FixRes), 2, 3",
            ha="center", va="top", fontsize=10, family="monospace")
    ax.text(6, 0.2, "Scored afterwards with frozen features - a linear probe and a small decoder on four Sentinel-1 "
                    "segmentation tasks", ha="center", fontsize=10, color="#333333")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
