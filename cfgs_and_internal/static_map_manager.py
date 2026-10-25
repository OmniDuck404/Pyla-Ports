#!/usr/bin/env python3
"""
Gestionnaire de carte statique (Blueprint discret) pour PylaAI.
Permet d'éliminer l'inférence YOLO des murs en match lorsque la carte
du jour est connue, réduisant l'utilisation CPU/GPU à < 0.5 ms par cycle.
"""
import json
import os
from pathlib import Path
import numpy as np


class StaticMapManager:
    def __init__(self, map_name="bb_today", maps_dir=None, scale_factor=1.0):
        if maps_dir is None:
            # Emplacement par défaut dans Pyla-Ports/cfgs_and_internal/cfg/maps
            base_path = Path(__file__).parent.resolve()
            maps_dir = base_path / "cfg" / "maps"
        else:
            maps_dir = Path(maps_dir)

        self.maps_dir = maps_dir
        self.map_name = map_name
        self.scale_factor = scale_factor or 1.0

        self.enabled = False
        self.meta = {}
        self.grid = []
        self.matrix = None  # uint8: 1 = mur, 0 = libre
        self.bushes_matrix = None  # uint8: 1 = buisson, 0 = autre

        self.rows = 0
        self.cols = 0
        self.tile_w = 54.0
        self.tile_h = 54.0

        # Position absolue estimée du joueur dans la grille globale (row, col)
        # Par défaut, spawn typique au bas du terrain (ex: row=25, col=10)
        self.player_grid_pos = None

        self.load_map(map_name)

    def load_map(self, map_name):
        json_path = self.maps_dir / f"{map_name}.json"
        npy_path = self.maps_dir / f"{map_name}.npy"

        if not json_path.exists():
            print(f"[StaticMapManager] Fichier map introuvable : {json_path}. Désactivé.")
            self.enabled = False
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.meta = data.get("meta", {})
            self.grid = data.get("grid", [])
            self.rows = self.meta.get("rows", len(self.grid))
            self.cols = self.meta.get("cols", len(self.grid[0]) if self.grid else 0)

            # Matrice binaire des murs
            if npy_path.exists():
                self.matrix = np.load(npy_path)
            else:
                char_map = {"#": 1, "+": 1, "G": 1}
                self.matrix = np.array(
                    [[char_map.get(ch, 0) for ch in row] for row in self.grid],
                    dtype=np.uint8
                )

            # Matrice binaire des buissons
            self.bushes_matrix = np.array(
                [[1 if ch in ("b", "B") else 0 for ch in row] for row in self.grid],
                dtype=np.uint8
            )

            # Dimensions visuelles d'une tuile projetée à l'écran
            tile_px = self.meta.get("tile_px", 54.0)
            if isinstance(tile_px, (list, tuple)):
                self.tile_w = float(tile_px[0])
                self.tile_h = float(tile_px[1])
            else:
                self.tile_w = float(tile_px)
                self.tile_h = float(tile_px)

            # Si le pas de tuile de la capture est petit (ex: 32px), ajuster à l'échelle écran
            # Le jeu à l'écran utilise généralement perceived_tile_size ~ 54px
            if self.tile_w < 40:
                scale_ratio = 54.0 / max(1.0, self.tile_w)
                self.tile_w *= scale_ratio
                self.tile_h *= scale_ratio

            self.enabled = True
            print(f"[StaticMapManager] Carte '{map_name}' chargée avec succès : {self.rows}x{self.cols} tuiles.")
            return True
        except Exception as e:
            print(f"[StaticMapManager] Erreur chargement carte {map_name} : {e}")
            self.enabled = False
            return False

    def reset_match_position(self, is_red_side=False):
        """Réinitialise la position estimée au spawn au début d'un match."""
        # En Brawl Ball / standard, spawn bleu vers le bas, rouge vers le haut
        if is_red_side:
            self.player_grid_pos = [max(0, int(self.rows * 0.2)), int(self.cols / 2)]
        else:
            self.player_grid_pos = [min(self.rows - 1, int(self.rows * 0.78)), int(self.cols / 2)]

    def update_player_movement(self, move_vector, dt):
        """Met à jour l'estimation de position sur la grille à partir de la vitesse."""
        if not self.enabled or self.player_grid_pos is None or not move_vector:
            return

        vx, vy = move_vector
        # Vitesse typique d'un brawler ~ 2.5 à 3 tuiles par seconde à pleine vitesse
        # dx, dy normalisés
        speed_tiles_per_sec = 2.8
        mag = (vx ** 2 + vy ** 2) ** 0.5
        if mag > 10:
            norm_x = vx / mag
            norm_y = vy / mag
            d_col = (norm_x * speed_tiles_per_sec * dt)
            d_row = (norm_y * speed_tiles_per_sec * dt)

            self.player_grid_pos[1] = max(0.0, min(self.cols - 1.0, self.player_grid_pos[1] + d_col))
            self.player_grid_pos[0] = max(0.0, min(self.rows - 1.0, self.player_grid_pos[0] + d_row))

    def get_visible_obstacles(self, player_screen_pos, screen_w=1024, screen_h=768):
        """
        Génère les bounding boxes d'obstacles visibles [x1, y1, x2, y2]
        relatives à la caméra actuelle, centrée sur le joueur à l'écran.
        """
        if not self.enabled or self.matrix is None:
            return [], []

        px, py = player_screen_pos
        tw = self.tile_w * self.scale_factor
        th = self.tile_h * self.scale_factor

        # Si aucune position estimée, on prend le centre de la map ou spawn bas
        if self.player_grid_pos is None:
            self.reset_match_position(is_red_side=False)

        p_row, p_col = self.player_grid_pos

        # Rayon de tuiles visibles autour du joueur
        half_cols = int(np.ceil((screen_w / 2.0) / max(1.0, tw))) + 1
        half_rows = int(np.ceil((screen_h / 2.0) / max(1.0, th))) + 1

        r_min = max(0, int(p_row - half_rows))
        r_max = min(self.rows, int(p_row + half_rows + 1))
        c_min = max(0, int(p_col - half_cols))
        c_max = min(self.cols, int(p_col + half_cols + 1))

        walls = []
        bushes = []

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                is_wall = self.matrix[r, c] == 1
                is_bush = self.bushes_matrix[r, c] == 1

                if not is_wall and not is_bush:
                    continue

                # Position écran relative à la caméra centrée sur player_screen_pos
                rel_col = c - p_col
                rel_row = r - p_row

                cx = px + rel_col * tw
                cy = py + rel_row * th

                x1 = cx - tw / 2.0
                y1 = cy - th / 2.0
                x2 = cx + tw / 2.0
                y2 = cy + th / 2.0

                # Filtrage des éléments hors écran
                if x2 < 0 or x1 > screen_w or y2 < 0 or y1 > screen_h:
                    continue

                if is_wall:
                    # Règle d'ancrage 2.5D : les murs utilisent la base au sol
                    # pour les tests de collision / visée
                    y_ground = y2 - (y2 - y1) * 0.40
                    walls.append([x1, y_ground, x2, y2])
                elif is_bush:
                    bushes.append([x1, y1, x2, y2])

        return walls, bushes
