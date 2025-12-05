"""AMASS NPZ writer for ASAP compatibility."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.writers.base import BaseWriter
from src.core.motion import MotionData
from src.core.registry import FormatRegistry
from src.core.mapping import JointMapping, SMPL_JOINTS


# Rotation to convert from Y-up (FBX) to Z-up (SMPL) coordinate system
# This is a +90 degree rotation around X axis: Y→Z, Z→-Y
Y_UP_TO_Z_UP = R.from_euler("x", 90, degrees=True)


@FormatRegistry.register_writer("amass", extensions=["npz"])
class AMASSWriter(BaseWriter):
    """
    Writer for AMASS NPZ format (ASAP-compatible).

    Output format:
        mocap_framerate: float
        poses: (N, 72) - SMPL joint rotations in axis-angle
        trans: (N, 3) - root translation
        betas: (16,) - SMPL shape parameters (zeros)
        gender: str - "neutral"
    """

    def __init__(self, mapping: Optional[JointMapping] = None):
        """
        Initialize AMASS writer.

        Args:
            mapping: Joint mapping configuration. If None, uses default mapping.
        """
        self.mapping = mapping or JointMapping.default()

    def get_extension(self) -> str:
        return ".npz"

    def write(self, motion: MotionData, filepath: str | Path, **kwargs) -> Path:
        """
        Convert MotionData to AMASS format and write to NPZ.

        Args:
            motion: Input motion data
            filepath: Output path
            **kwargs:
                target_fps: Resample to this FPS (default: keep original)
                gender: SMPL gender (default: "neutral")
                center: Center the motion at origin (default: True)
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() != ".npz":
            filepath = filepath.with_suffix(".npz")

        # Resample if requested
        target_fps = kwargs.get("target_fps")
        if target_fps and target_fps != motion.fps:
            print(f"[AMASSWriter] Resampling from {motion.fps} to {target_fps} fps")
            motion = motion.resample(target_fps)

        # Convert rotations to SMPL format (with Y-up to Z-up conversion)
        poses = self._convert_to_smpl_poses(motion)
        trans = self._extract_root_translation(motion)

        # Center the motion at origin
        # Detect if frame 0 is a calibration frame (huge jump to frame 1)
        skip_first_frame = False
        if trans.shape[0] > 1:
            frame0_to_1_dist = np.linalg.norm(trans[1, :2] - trans[0, :2])
            if frame0_to_1_dist > 1.0:  # More than 1 meter jump
                # Frame 0 is calibration - remove it and center on frame 1
                print(
                    f"[AMASSWriter] Detected calibration frame 0 (jump of {frame0_to_1_dist:.2f}m to frame 1)"
                )
                print(f"[AMASSWriter] Removing calibration frame and centering on frame 1")
                skip_first_frame = True
                poses = poses[1:]  # Remove first frame
                trans = trans[1:]  # Remove first frame
                xy_offset = trans[0, :2].copy()  # Now frame 0 is the old frame 1
            else:
                xy_offset = trans[0, :2].copy()

            trans[:, 0] -= xy_offset[0]
            trans[:, 1] -= xy_offset[1]
            print(
                f"[AMASSWriter] Centered at origin (offset: X={xy_offset[0]:.2f}m, Y={xy_offset[1]:.2f}m)"
            )
            print(f"[AMASSWriter] Final frame count: {trans.shape[0]}")

        # Apply scale factor to XY translation if specified
        scale = kwargs.get("scale")
        if scale and scale != 1.0:
            trans[:, 0] *= scale
            trans[:, 1] *= scale
            print(f"[AMASSWriter] Applied XY scale: {scale} (movement reduced by {1/scale:.1f}x)")
            xy_range = np.linalg.norm(
                [trans[:, 0].max() - trans[:, 0].min(), trans[:, 1].max() - trans[:, 1].min()]
            )
            print(f"[AMASSWriter] Scaled XY range: {xy_range:.2f}m")

        # Save
        np.savez(
            filepath,
            mocap_framerate=np.array(motion.fps, dtype=np.float32),
            poses=poses.astype(np.float32),
            trans=trans.astype(np.float32),
            betas=np.zeros(16, dtype=np.float32),
            gender=np.array(kwargs.get("gender", "neutral")),
        )

        mapped_count = self._count_mapped_joints(motion)
        print(f"[AMASSWriter] Saved: {filepath}")
        print(f"  Frames: {motion.n_frames}, FPS: {motion.fps}")
        print(f"  Mapped joints: {mapped_count}/{len(SMPL_JOINTS)} SMPL joints")
        print(f"  Mapping used: {self.mapping.name}")

        return filepath

    def _convert_to_smpl_poses(self, motion: MotionData) -> np.ndarray:
        """Convert quaternion rotations to SMPL axis-angle format with coordinate transform."""
        n_frames = motion.n_frames
        poses = np.zeros((n_frames, 72), dtype=np.float32)  # 24 joints × 3

        # Build mapping from input joints to SMPL indices
        input_to_smpl = {}
        for i, input_name in enumerate(motion.joint_names):
            # Try direct mapping
            smpl_name = self.mapping.get_smpl_joint(input_name)

            # Try with common prefixes removed
            if not smpl_name:
                clean_name = self._clean_joint_name(input_name)
                smpl_name = self.mapping.get_smpl_joint(clean_name)

            if smpl_name and self.mapping.is_valid_smpl_joint(smpl_name):
                smpl_idx = self.mapping.get_smpl_index(smpl_name)
                input_to_smpl[i] = smpl_idx

        # Convert each mapped joint
        for input_idx, smpl_idx in input_to_smpl.items():
            quats = motion.joint_rotations[:, input_idx]  # (N, 4) xyzw

            # Convert quaternions to rotation objects
            rot = R.from_quat(quats)

            # For the root (pelvis), apply coordinate system transformation
            # This converts from Y-up to Z-up
            if smpl_idx == 0:  # Pelvis
                rot = Y_UP_TO_Z_UP * rot

            # Convert to axis-angle
            axis_angle = rot.as_rotvec()  # (N, 3)

            # Store in poses
            poses[:, smpl_idx * 3 : smpl_idx * 3 + 3] = axis_angle

        return poses

    def _extract_root_translation(self, motion: MotionData) -> np.ndarray:
        """
        Extract root translation from the pelvis/hips joint.

        Converts from Y-up (FBX) to Z-up (SMPL) coordinate system:
        - SMPL X = FBX X
        - SMPL Y = -FBX Z
        - SMPL Z = FBX Y (height)
        """
        # Look for the actual pelvis/hips joint, not scene root
        pelvis_names = ["Hips", "hips", "pelvis", "Pelvis", "hip", "Hip"]

        for name in pelvis_names:
            if name in motion.joint_names:
                idx = motion.joint_names.index(name)
                fbx_trans = motion.joint_positions[:, idx].copy()

                # Convert Y-up to Z-up coordinate system
                # +90° rotation around X: Y→Z, Z→-Y
                smpl_trans = np.zeros_like(fbx_trans)
                smpl_trans[:, 0] = fbx_trans[:, 0]  # X stays X
                smpl_trans[:, 1] = fbx_trans[:, 2]  # Y becomes Z (forward/back)
                smpl_trans[:, 2] = fbx_trans[:, 1]  # Z becomes Y (height)

                print(f"[AMASSWriter] Using '{name}' for root translation")
                print(f"[AMASSWriter] Converted Y-up to Z-up coordinate system")
                print(
                    f"[AMASSWriter] Translation Z (height) range: {smpl_trans[:, 2].min():.3f} to {smpl_trans[:, 2].max():.3f} m"
                )
                return smpl_trans

        # Fall back to first joint with actual movement
        for i, name in enumerate(motion.joint_names):
            trans = motion.joint_positions[:, i]
            if np.abs(trans).max() > 0.01:  # Has movement
                print(f"[AMASSWriter] Using '{name}' for root translation (fallback)")
                # Apply same coordinate conversion
                smpl_trans = np.zeros_like(trans)
                smpl_trans[:, 0] = trans[:, 0]
                smpl_trans[:, 1] = trans[:, 2]
                smpl_trans[:, 2] = trans[:, 1]
                return smpl_trans

        # Ultimate fallback
        print("[AMASSWriter] Warning: No root translation found, using zeros")
        return np.zeros((motion.n_frames, 3), dtype=np.float32)

    def _count_mapped_joints(self, motion: MotionData) -> int:
        """Count how many joints were successfully mapped to SMPL."""
        count = 0
        for name in motion.joint_names:
            smpl_name = self.mapping.get_smpl_joint(name)
            if not smpl_name:
                smpl_name = self.mapping.get_smpl_joint(self._clean_joint_name(name))
            if smpl_name and self.mapping.is_valid_smpl_joint(smpl_name):
                count += 1
        return count

    def _clean_joint_name(self, name: str) -> str:
        """Remove common prefixes from joint names."""
        prefixes = [
            "mixamorig:",
            "mixamorig_",
            "Bip01_",
            "Bip01 ",
            "Bip001_",
            "Character1:",
            "Character1_",
            "Root:",
            "Armature|",
        ]
        clean = name
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
            # Case insensitive
            if clean.lower().startswith(prefix.lower()):
                clean = clean[len(prefix) :]
        return clean

    def get_mapping_report(self, motion: MotionData) -> Dict[str, str]:
        """
        Generate a report showing how joints are mapped.

        Returns:
            Dict mapping source joint name to SMPL joint name (or "NOT MAPPED")
        """
        report = {}
        for name in motion.joint_names:
            smpl_name = self.mapping.get_smpl_joint(name)
            if not smpl_name:
                smpl_name = self.mapping.get_smpl_joint(self._clean_joint_name(name))

            if smpl_name and self.mapping.is_valid_smpl_joint(smpl_name):
                report[name] = smpl_name
            else:
                report[name] = "NOT MAPPED"
        return report
