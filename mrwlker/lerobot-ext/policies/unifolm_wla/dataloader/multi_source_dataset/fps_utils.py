import numpy as np
from scipy.interpolate import interp1d


def resample_action_chunk(
    actions: np.ndarray,
    source_fps: int,
    target_fps: int,
) -> np.ndarray:
    """Resample action chunk from source FPS to target FPS.

    Uses linear interpolation for all dimensions (xyz positions and rotvec
    rotation vectors). This is valid because relative actions are small deltas.

    Args:
        actions: (source_fps, D) array of action values
        source_fps: original number of timesteps (= source FPS for 1 second)
        target_fps: target number of timesteps (= target FPS for 1 second)
    Returns:
        (target_fps, D) resampled action values
    """
    if source_fps == target_fps:
        return actions

    n_src = actions.shape[0]
    src_t = np.linspace(0.0, (n_src - 1) / source_fps, n_src)
    tgt_t = np.linspace(0.0, (target_fps - 1) / target_fps, target_fps)

    f = interp1d(src_t, actions, axis=0, kind="linear",
                 bounds_error=False, fill_value=(actions[0], actions[-1]))
    return f(tgt_t).astype(actions.dtype)


def build_bspline_resample_matrix(
    source_fps: int,
    target_fps: int,
    sample_seconds: float = 2.0,
    degree: int = 3,
) -> np.ndarray:
    """Precompute B-spline resampling matrix: (target_fps, n_src) shape.

    Fits a cubic B-spline through `sample_seconds` of source-FPS data,
    then evaluates at 1-second of target-FPS timestamps.

    Args:
        source_fps: source dataset FPS
        target_fps: target FPS for output
        sample_seconds: how many seconds of source data to load for fitting
        degree: B-spline polynomial degree
    Returns:
        (target_fps, n_src) resampling matrix R, where n_src = sample_seconds * source_fps
    """
    import torch
    from mp_pytorch.basis_gn import LinearPhaseGenerator, UniBSplineBasis

    n_src = int(sample_seconds * source_fps) + 1  # +1 for t=0 anchor
    # Use fewer basis functions than data points for smooth approximation.
    # Exact interpolation (num_basis=n_src) causes ringing with few points.
    num_basis = max(degree + 1, n_src // 2 + 1)

    phase_gen = LinearPhaseGenerator(tau=sample_seconds, delay=0.0)
    basis = UniBSplineBasis(
        phase_gen,
        num_basis=num_basis,
        degree_p=degree,
    )

    # Source timestamps: [0, 1/fps, 2/fps, ..., sample_seconds]
    src_t = torch.tensor([i / source_fps for i in range(0, n_src)],
                         dtype=torch.float64)
    # Target timestamps: actions at [0, 1/fps, ..., (target_fps-1)/target_fps]
    tgt_t = torch.tensor([i / target_fps for i in range(0, target_fps)],
                         dtype=torch.float64)

    B_src = basis.basis(src_t).double()  # (n_src, num_ctrlp)
    B_tgt = basis.basis(tgt_t).double()  # (target_fps, num_ctrlp)

    # R = B_tgt @ pinv(B_src) = B_tgt @ (B_src^T B_src + λI)^{-1} B_src^T
    reg = 1e-9
    BtB = B_src.T @ B_src + reg * torch.eye(B_src.shape[1], dtype=torch.float64)
    R = B_tgt @ torch.linalg.solve(BtB, B_src.T)

    return R.numpy().astype(np.float32)
