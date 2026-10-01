"""Reject a prop falling during settling even if it later rests on the floor."""
import numpy as np


def initial_placement_integrity(positions, tabletop_z=.75):
    positions = np.asarray(positions)
    excursion = np.max(np.linalg.norm(positions - positions[0], axis=2), axis=0)
    lowest = np.min(positions[:, :, 2], axis=0)
    return {
        'pass': bool(np.max(excursion) < .025 and np.min(lowest) > tabletop_z - .03),
        'max_initial_excursion_m': excursion.tolist(),
        'lowest_body_origins_m': lowest.tolist(),
    }
