"""
Universal motion data structure that all readers produce and all writers consume.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class MotionData:
    """
    Universal representation of motion capture data.
    All readers convert to this format, all writers read from this format.
    """

    # Core motion data
    joint_names: List[str]  # Names of joints in the skeleton
    joint_positions: np.ndarray  # (N, J, 3) world-space positions
    joint_rotations: np.ndarray  # (N, J, 4) quaternions (xyzw)

    # Timing
    fps: float  # Frames per second
    n_frames: int = field(init=False)  # Number of frames (computed)
    n_joints: int = field(init=False)  # Number of joints (computed)

    # Optional hierarchy info
    joint_parents: Optional[np.ndarray] = None  # Parent index for each joint (-1 for root)

    # Metadata
    source_file: Optional[str] = None
    source_format: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_frames = self.joint_positions.shape[0]
        self.n_joints = len(self.joint_names)

        # Validate shapes
        assert self.joint_positions.shape == (
            self.n_frames,
            self.n_joints,
            3,
        ), f"positions shape mismatch: {self.joint_positions.shape}, expected ({self.n_frames}, {self.n_joints}, 3)"
        assert self.joint_rotations.shape == (
            self.n_frames,
            self.n_joints,
            4,
        ), f"rotations shape mismatch: {self.joint_rotations.shape}, expected ({self.n_frames}, {self.n_joints}, 4)"

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n_frames / self.fps

    def get_joint_index(self, name: str) -> int:
        """Get index of joint by name."""
        return self.joint_names.index(name)

    def get_root_joint(self) -> Optional[str]:
        """Find the root joint (parent = -1)."""
        if self.joint_parents is not None:
            root_idx = np.where(self.joint_parents == -1)[0]
            if len(root_idx) > 0:
                return self.joint_names[root_idx[0]]
        # Guess common root names
        for name in ["Hips", "pelvis", "Pelvis", "root", "Root", "hip", "Hip"]:
            if name in self.joint_names:
                return name
        return self.joint_names[0] if self.joint_names else None

    def resample(self, target_fps: float) -> "MotionData":
        """Resample motion to a different frame rate."""
        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation as R, Slerp

        if target_fps == self.fps:
            return self

        # Time arrays
        t_old = np.arange(self.n_frames) / self.fps
        t_new = np.arange(0, t_old[-1], 1.0 / target_fps)
        n_new = len(t_new)

        # Interpolate positions (linear)
        new_positions = np.zeros((n_new, self.n_joints, 3))
        for j in range(self.n_joints):
            for axis in range(3):
                f = interp1d(t_old, self.joint_positions[:, j, axis], kind="linear")
                new_positions[:, j, axis] = f(t_new)

        # Interpolate rotations (slerp)
        new_rotations = np.zeros((n_new, self.n_joints, 4))
        for j in range(self.n_joints):
            rots = R.from_quat(self.joint_rotations[:, j])
            slerp = Slerp(t_old, rots)
            new_rotations[:, j] = slerp(t_new).as_quat()

        return MotionData(
            joint_names=self.joint_names.copy(),
            joint_positions=new_positions,
            joint_rotations=new_rotations,
            fps=target_fps,
            joint_parents=self.joint_parents.copy() if self.joint_parents is not None else None,
            source_file=self.source_file,
            source_format=self.source_format,
            metadata=self.metadata.copy(),
        )

    def print_info(self):
        """Print summary information about the motion."""
        print(f"\n=== Motion Info ===")
        print(f"  Source: {self.source_format or 'unknown'}")
        print(f"  File: {self.source_file or 'unknown'}")
        print(f"  Joints: {self.n_joints}")
        print(f"  Frames: {self.n_frames}")
        print(f"  FPS: {self.fps}")
        print(f"  Duration: {self.duration:.2f}s")
        print(f"  Root joint: {self.get_root_joint()}")
        print(f"\n  Joint names:")
        for i, name in enumerate(self.joint_names):
            print(f"    [{i:2d}] {name}")
