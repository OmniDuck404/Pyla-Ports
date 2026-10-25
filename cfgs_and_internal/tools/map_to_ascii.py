#!/usr/bin/env python3
"""Convertit une capture de carte Brawl Stars (vue du dessus, sans HUD) en
matrice ASCII utilisable par le bot.

Usage :
    python tools/map_to_ascii.py IMAGE.png [--id bb_today] [--out cfg/maps]

Sorties (dans --out) :
    <id>.json  : metadata + grille ASCII
    <id>.npy   : matrice uint8 (0=sol, 1=mur, 2=anneau rouge, 3=anneau bleu)

Legende ASCII :
    '#' mur plein     '+' mur partiel
    'R' anneau rouge  'B' anneau bleu  '.' sol libre

La calibration (pas de tuile + offset) est auto-detectee sur l'image ; elle
suppose une capture de la carte entiere en colonnes de tuiles alignees.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Valeurs de la matrice numerique .npy :
#   0 = sol libre (praticable)
#   1 = mur (bloquant)  -- '#' ET '+' sont bloquants : les hitboxes Brawl Stars
#       sont carrees et strictes, raser un mur partiel coince le bot.
#   2/3 = anneaux rouge/bleu : SOL PRATICABLE, marqueurs de zone de but.
#   4 = buisson (praticable mais masque la vue)
#   5 = cage / but (structure, bloquant)
CHAR_TO_VALUE = {"#": 1, "+": 1, "R": 0, "B": 0, ".": 0, "b": 4, "G": 5}
# Marquage des anneaux conserve a part pour l'affichage et les metas.
MARKER_CHARS = {"R", "B", "G"}

# LUT palette "obstacle" par style de map : on ne code pas la couleur en dur, on
# l'apprend depuis le fond de l'image (voir `_background_color`).
OBSTACLE_KEYS = ("block", "orange", "bush", "goal")


def _background_color(image):
    """Couleur de fond dominante = mediane des bords de l'image (jamais un mur)."""
    border = np.concatenate([
        image[0:8, :].reshape(-1, 3),
        image[-8:, :].reshape(-1, 3),
        image[:, 0:8].reshape(-1, 3),
        image[:, -8:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def build_masks(image):
    """Construit les masques d'obstacles, robustes au style de map (sable/herbe/neige).

    Les blocs de jeu partagent un code couleur stable en HSV (orange/beige a
    haute valeur), les buissons sont le vert vegetal le plus sombre de la palette,
    et les cages sont rouges. Le sol (sable violet ou herbe claire) est toujours
    plus lumineux que les buissons et sans teinte orange saturée.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Blocs / murs : orange a beige (teinte 0..32) et haute valeur.
    warm = cv2.inRange(hsv, np.array((0, 60, 150)), np.array((32, 255, 255)))
    # Teinte orange franche (blocs satures des cartes recentes).
    orange = cv2.inRange(hsv, np.array((0, 170, 170)), np.array((18, 255, 255)))
    # Buissons : vert vegetal sombre (teinte 45..75, valeur basse a moyenne).
    # L'herbe de fond est plus lumineuse (V ~ 146) : le seuil V <= 130 la distingue.
    bush = cv2.inRange(hsv, np.array((45, 110, 40)), np.array((75, 255, 132)))
    # Cages / buts : structure rouge sombre a rose.
    goal = cv2.inRange(hsv, np.array((0, 120, 110)), np.array((12, 255, 230)))
    goal |= cv2.inRange(hsv, np.array((168, 120, 110)), np.array((179, 255, 230)))

    return {"block": warm, "orange": orange, "bush": bush, "goal": goal}



def _grid_axis_profile(edges, axis, step_min=42.0, step_max=48.0):
    """Trouve (pas, offset) maximisant l'energie de Canny sur les frontieres."""
    profile = edges.sum(axis=axis) / 255.0
    span = profile.shape[0]
    best = None
    for step in np.arange(step_min, step_max, 0.02):
        for offset in np.arange(-step / 2.0, step / 2.0, 0.5):
            idx = np.arange(offset, span, step)
            idx = np.clip(idx.astype(int), 0, span - 1)
            score = profile[idx].sum() / max(1, len(idx))
            if best is None or score > best[0]:
                best = (score, float(step), float(offset))
    return best[1], best[2]


def _fraction(mask, x1, y1, x2, y2):
    h, w = mask.shape[:2]
    sub = mask[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))]
    return float(sub.mean() / 255) if sub.size else 0.0


def detect_circles(image, tile_x, tile_y, origin_x, origin_y):
    """Detecte les anneaux de but et les classe rouge/bleu via la teinte mediane."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=55,
        param1=120, param2=45, minRadius=25, maxRadius=40,
    )
    circles = []
    if found is None:
        return circles
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for x, y, radius in np.round(found[0]).astype(int):
        px = np.clip((x + radius * np.cos(angles)).astype(int), 0, w - 1)
        py = np.clip((y + radius * np.sin(angles)).astype(int), 0, h - 1)
        hsv = cv2.cvtColor(image[py, px].reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hue = float(np.median(hsv[:, 0]))
        sat = float(np.median(hsv[:, 1]))
        if sat < 80:
            continue
        if hue > 150 or hue < 10:
            kind = "R"
        elif 85 < hue < 115:
            kind = "B"
        else:
            continue
        circles.append((int((x - origin_x) // tile_x), int((y - origin_y) // tile_y), kind))
    return circles


def image_to_grid(image, tile_px=None, origin=None, bounds=None):
    h, w = image.shape[:2]
    edges = cv2.Canny(image, 60, 160)

    if tile_px is None or origin is None:
        res_scale = w / 1920.0
        s_min = max(20.0, 42.0 * res_scale)
        s_max = max(26.0, 48.0 * res_scale)
        step_x, off_x = _grid_axis_profile(edges, axis=0, step_min=s_min, step_max=s_max)
        step_y, off_y = _grid_axis_profile(edges, axis=1, step_min=s_min, step_max=s_max)
        tile_px = tile_px or round((step_x + step_y) / 2.0, 1)
        origin = origin or [round(off_x, 1), round(off_y, 1)]

    # Pas de tuile : scalaire (tuile carree) ou (x, y) si la perspective
    # compresse un axe (cas des captures de carte Brawl Ball).
    if isinstance(tile_px, (tuple, list)):
        tile_x, tile_y = float(tile_px[0]), float(tile_px[1])
    else:
        tile_x = tile_y = float(tile_px)

    origin_x, origin_y = origin
    masks = build_masks(image)

    # Par defaut : derniere tuile entierement contenue dans l'image.
    # `bounds` (x1, y1, x2, y2) permet de restreindre a la zone utile de la carte.
    bx1, by1, bx2, by2 = bounds if bounds else (0.0, 0.0, float(w), float(h))
    cols = int((bx2 - origin_x) // tile_x)
    rows = int((by2 - origin_y) // tile_y)

    circles = detect_circles(image, tile_x, tile_y, origin_x, origin_y)

    grid = []
    for row in range(rows):
        line = ""
        for col in range(cols):
            x1 = origin_x + col * tile_x
            y1 = origin_y + row * tile_y
            x2, y2 = x1 + tile_x, y1 + tile_y
            solid = max(
                _fraction(masks["block"], x1, y1, x2, y2),
                _fraction(masks["orange"], x1, y1, x2, y2),
            )
            bush = _fraction(masks["bush"], x1, y1, x2, y2)
            goal = _fraction(masks["goal"], x1, y1, x2, y2)
            circle = next((k for c, r, k in circles if c == col and r == row), None)
            # Priorite : un obstacle plein l'emporte toujours sur un marqueur
            # de structure, sinon la cage "mange" les blocs poses devant elle.
            if solid > 0.50:
                line += "#"
            elif circle and solid <= 0.18:
                line += circle
            elif goal > 0.40:
                line += "G"
            elif bush > 0.40 and bush > solid:
                # Les buissons sont traversables : ils ne bloquent pas, mais
                # masquent la vue (utile pour le pathfinding et la LOS).
                line += "b"
            elif solid > 0.18:
                line += "+"
            else:
                line += "."
        grid.append(line)

    meta = {
        "rows": rows,
        "cols": cols,
        "tile_px": [tile_x, tile_y] if tile_x != tile_y else tile_x,
        "origin_px": [float(origin_x), float(origin_y)],
        "circles": [[c, r, k] for c, r, k in circles],
    }
    return grid, meta


def main():
    parser = argparse.ArgumentParser(description="Carte Brawl Stars -> matrice ASCII")
    parser.add_argument("image", type=Path, help="capture de la carte (vue du dessus)")
    parser.add_argument("--id", default=None, help="identifiant de la carte (defaut: nom du fichier)")
    parser.add_argument("--out", type=Path, default=Path("cfg/maps"), help="dossier de sortie")
    parser.add_argument("--tile-px", type=float, nargs="+", default=None,
                        help="pas de tuile : une valeur (carre) ou deux (x y)")
    parser.add_argument("--origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="offset pixel de la premiere tuile (auto si omis)")
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                        help="zone utile de la carte en pixels (defaut: image entiere)")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Image illisible : {args.image}")

    tile_px = args.tile_px
    if isinstance(tile_px, list):
        tile_px = tile_px[0] if len(tile_px) == 1 else tuple(tile_px[:2])

    grid, meta = image_to_grid(image, tile_px, args.origin, args.bounds)
    map_id = args.id or args.image.stem

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "id": map_id,
            "mode": "unknown",
            "rows": meta["rows"],
            "cols": meta["cols"],
            "tile_px": meta["tile_px"],
            "origin_px": meta["origin_px"],
            "src": str(args.image),
            "legend": {
                "#": "mur plein",
                "+": "mur partiel",
                "R": "anneau rouge",
                "B": "anneau bleu",
                "G": "cage / but",
                "b": "buisson (traversable, masque la vue)",
                ".": "sol libre",
            },
        },
        "grid": grid,
    }
    json_path = args.out / f"{map_id}.json"
    npy_path = args.out / f"{map_id}.npy"
    json_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    matrix = np.array([[CHAR_TO_VALUE[ch] for ch in row] for row in grid], dtype=np.uint8)
    np.save(npy_path, matrix)

    for index, row in enumerate(grid):
        print(f"{index:2d} |{row}|")
    print()
    print(f"grille {meta['rows']}x{meta['cols']}  tuile={meta['tile_px']}px  origine={meta['origin_px']}")
    print(f"cercles detectes : {meta['circles']}")
    print(f"ecrit : {json_path}  et  {npy_path}")


if __name__ == "__main__":
    main()

