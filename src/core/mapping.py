"""Joint mapping configuration for SMPL conversion."""

import json
from pathlib import Path
from typing import Dict, List, Optional


# Package root for finding built-in mappings
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MAPPINGS_DIR = PACKAGE_ROOT / "mappings"


# SMPL's 24 joints (standard order)
SMPL_JOINTS = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]


class JointMapping:
    """Handles joint name mapping from source skeleton to SMPL."""

    def __init__(
        self,
        joints: Dict[str, str],
        name: str = "custom",
        description: str = "",
    ):
        """
        Initialize joint mapping.

        Args:
            joints: Dict mapping source joint names to SMPL joint names
            name: Human-readable name for this mapping
            description: Description of this mapping
        """
        self.joints = joints
        self.name = name
        self.description = description

    def get_smpl_joint(self, source_joint: str) -> Optional[str]:
        """Get SMPL joint name for a source joint."""
        return self.joints.get(source_joint)

    def is_valid_smpl_joint(self, smpl_joint: str) -> bool:
        """Check if a joint name is a valid SMPL joint."""
        return smpl_joint in SMPL_JOINTS

    def get_smpl_index(self, smpl_joint: str) -> int:
        """Get the index of a SMPL joint (0-23)."""
        return SMPL_JOINTS.index(smpl_joint)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "joints": self.joints,
        }

    def save(self, filepath: str | Path) -> None:
        """Save mapping to JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, filepath: str | Path) -> "JointMapping":
        """Load mapping from JSON file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Mapping file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(
            joints=data.get("joints", {}),
            name=data.get("name", filepath.stem),
            description=data.get("description", ""),
        )

    @classmethod
    def from_name(cls, name: str) -> "JointMapping":
        """
        Load a built-in mapping by name.

        Args:
            name: Mapping name (e.g., 'default', 'cmu', 'mixamo_prefixed')
                  or path to a custom JSON file

        Returns:
            JointMapping instance
        """
        # Check if it's a path to a custom file
        path = Path(name)
        if path.suffix == ".json" or path.exists():
            return cls.from_file(path)

        # Look for built-in mapping
        builtin_path = MAPPINGS_DIR / f"{name}.json"
        if builtin_path.exists():
            return cls.from_file(builtin_path)

        # List available mappings in error
        available = cls.list_available()
        raise ValueError(
            f"Unknown mapping: '{name}'\n"
            f"Available built-in mappings: {', '.join(available)}\n"
            f"Or provide a path to a custom .json file"
        )

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available built-in mapping names."""
        if not MAPPINGS_DIR.exists():
            return []
        return sorted([f.stem for f in MAPPINGS_DIR.glob("*.json")])

    @classmethod
    def default(cls) -> "JointMapping":
        """Get the default mapping."""
        return cls.from_name("default")

    def __repr__(self) -> str:
        return f"JointMapping(name='{self.name}', joints={len(self.joints)})"
