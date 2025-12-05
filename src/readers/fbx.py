import numpy as np
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from src.readers.base import BaseReader
from src.core.motion import MotionData
from src.core.registry import FormatRegistry


# FBX time units per second
FBX_TIME_UNIT = 46186158000


@FormatRegistry.register_reader("fbx", extensions=["fbx"])
class FbxReader(BaseReader):
    def __init__(self):
        self._backend = None

    def can_read(self, filepath: str | Path) -> bool:
        """Check if file is a readable FBX."""
        path = Path(filepath)
        return path.suffix.lower() == ".fbx" and path.exists()

    def read(self, filepath: str | Path) -> MotionData:
        """Read FBX file and return MotionData."""
        filepath = Path(filepath)

        if not self.can_read(filepath):
            raise ValueError(f"Cannot read file: {filepath}")

        # Check if ASCII or binary
        with open(filepath, "rb") as f:
            header = f.read(20)

        if header.startswith(b"Kaydara FBX Binary"):
            # Binary FBX - try external libraries
            joint_names, positions, rotations, fps, parents = self._load_binary_fbx(filepath)
        else:
            # ASCII FBX - use built-in parser
            joint_names, positions, rotations, fps, parents = self._load_ascii_fbx(filepath)

        return MotionData(
            joint_names=joint_names,
            joint_positions=positions,
            joint_rotations=rotations,
            fps=fps,
            joint_parents=parents,
            source_file=str(filepath),
            source_format="fbx",
        )

    def _load_ascii_fbx(
        self, filepath: Path
    ) -> Tuple[List[str], np.ndarray, np.ndarray, float, np.ndarray]:
        """Parse ASCII FBX file directly (no external dependencies)."""
        print(f"[FBXReader] Parsing ASCII FBX...")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extract FPS from GlobalSettings
        fps = self._extract_fps(content)
        print(f"[FBXReader] Detected FPS: {fps}")

        # Extract unit scale (for cm to m conversion)
        unit_scale = self._extract_unit_scale(content)
        print(f"[FBXReader] Unit scale: {unit_scale}")

        # Extract joint hierarchy
        joints, parents_dict = self._extract_joints(content)
        joint_names = list(joints.keys())
        n_joints = len(joint_names)
        print(f"[FBXReader] Found {n_joints} joints: {joint_names[:5]}...")

        if n_joints == 0:
            raise ValueError("No joints found in FBX file")

        # Build parent index array
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        parents = np.array(
            [name_to_idx.get(parents_dict.get(name), -1) for name in joint_names],
            dtype=np.int32,
        )

        # Extract animation curves
        curves = self._extract_curves(content)
        print(f"[FBXReader] Found {len(curves)} animation curves")

        # Build connection maps
        curve_to_node, node_to_joint = self._extract_connections(content, joints)
        print(f"[FBXReader] Curve->Node connections: {len(curve_to_node)}")
        print(f"[FBXReader] Node->Joint connections: {len(node_to_joint)}")

        # Map curves to joints
        anim_data = self._build_animation_data(curves, curve_to_node, node_to_joint, fps)
        print(f"[FBXReader] Animation channels: {len(anim_data)}")

        if not anim_data:
            raise ValueError("No animation data found in FBX file")

        # Determine frame count from animation curves
        n_frames = self._get_frame_count(anim_data)
        print(f"[FBXReader] Frame count: {n_frames}")

        if n_frames == 0:
            raise ValueError("No keyframes found in animation")

        # Build position and rotation arrays
        positions = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
        rotations = np.zeros((n_frames, n_joints, 4), dtype=np.float32)
        rotations[..., 3] = 1.0  # Default identity quaternion (w=1)

        # Get default/rest pose values from joint properties
        default_trans, default_rot = self._extract_default_pose(content, joints)

        for joint_idx, joint_name in enumerate(joint_names):
            joint_id = joints[joint_name]

            # Get translation curves
            tx = anim_data.get((joint_id, "T", "X"), {})
            ty = anim_data.get((joint_id, "T", "Y"), {})
            tz = anim_data.get((joint_id, "T", "Z"), {})

            # Get rotation curves (Euler angles in FBX)
            rx = anim_data.get((joint_id, "R", "X"), {})
            ry = anim_data.get((joint_id, "R", "Y"), {})
            rz = anim_data.get((joint_id, "R", "Z"), {})

            # Get defaults for this joint
            def_t = default_trans.get(joint_id, (0.0, 0.0, 0.0))
            def_r = default_rot.get(joint_id, (0.0, 0.0, 0.0))

            # Fill position data (convert units)
            for frame in range(n_frames):
                # Use animated value if exists, otherwise default
                x = tx.get(frame, def_t[0]) * unit_scale
                y = ty.get(frame, def_t[1]) * unit_scale
                z = tz.get(frame, def_t[2]) * unit_scale
                positions[frame, joint_idx] = [x, y, z]

            # Fill rotation data (convert Euler to quaternion)
            for frame in range(n_frames):
                euler_x = np.radians(rx.get(frame, def_r[0]))
                euler_y = np.radians(ry.get(frame, def_r[1]))
                euler_z = np.radians(rz.get(frame, def_r[2]))

                # Convert Euler ZYX (common in FBX) to quaternion
                # Note: FBX typically uses intrinsic ZYX order
                quat = self._euler_zyx_to_quat(euler_x, euler_y, euler_z)
                rotations[frame, joint_idx] = quat

        self._backend = "ascii"
        print(f"[FBXReader] Loaded: {n_frames} frames, {n_joints} joints, {fps:.1f} fps")
        return joint_names, positions, rotations, fps, parents

    def _extract_fps(self, content: str) -> float:
        """Extract frame rate from FBX GlobalSettings."""
        # Look for CustomFrameRate - format: P: "CustomFrameRate", "double", "Number", "",240
        # The pattern has 4 comma-separated fields before the value
        match = re.search(r'"CustomFrameRate"[^,]*,[^,]*,[^,]*,[^,]*,\s*(-?[\d.]+)', content)
        if match:
            fps = float(match.group(1))
            if fps > 0:
                print(f"[FBXReader] Found CustomFrameRate: {fps}")
                return fps

        # Look for TimeMode (enum values)
        match = re.search(r'"TimeMode"[^,]*,[^,]*,[^,]*,[^,]*,\s*(\d+)', content)
        if match:
            time_mode = int(match.group(1))
            time_mode_fps = {
                0: 30.0,
                1: 120.0,
                2: 100.0,
                3: 60.0,
                4: 50.0,
                5: 48.0,
                6: 30.0,
                7: 30.0,
                8: 29.97,
                9: 29.97,
                10: 25.0,
                11: 24.0,
                12: 24.0,
                13: 96.0,
                14: 240.0,  # Custom - but we already checked CustomFrameRate
            }
            return time_mode_fps.get(time_mode, 30.0)

        return 30.0

    def _extract_unit_scale(self, content: str) -> float:
        """
        Extract unit scale factor by inferring from actual data.

        We use the Hips/Pelvis height to infer units since human hip height
        should be approximately 0.8-1.2 meters when standing.
        """
        # First, find the Hips default translation (Y is height in FBX Y-up)
        hips_height = self._get_hips_default_height(content)

        if hips_height is not None and hips_height > 0:
            # Expected hip height in meters: 0.8-1.2m
            # Infer the scale based on the raw value
            expected_height = 0.95  # target ~0.95m hip height

            if 0.5 < hips_height < 2.0:
                # Already in meters
                print(f"[FBXReader] Inferred units: meters (hip height={hips_height:.2f})")
                return 1.0
            elif 50 < hips_height < 200:
                # Centimeters
                print(f"[FBXReader] Inferred units: centimeters (hip height={hips_height:.1f}cm)")
                return 0.01
            elif 500 < hips_height < 2000:
                # Millimeters
                print(f"[FBXReader] Inferred units: millimeters (hip height={hips_height:.0f}mm)")
                return 0.001
            else:
                # Try to compute the scale factor
                scale = expected_height / hips_height
                print(
                    f"[FBXReader] Inferred scale: {scale:.6f} (hip height={hips_height:.2f} -> {hips_height*scale:.2f}m)"
                )
                return scale

        # Fallback: try to read UnitScaleFactor from FBX metadata
        match = re.search(r'"UnitScaleFactor"[^,]*,[^,]*,[^,]*,[^,]*,\s*(-?[\d.]+)', content)
        if match:
            fbx_scale = float(match.group(1))
            # UnitScaleFactor in FBX: cm per file unit
            # scale=1 means cm, scale=100 means meters
            unit_scale = fbx_scale / 100.0
            print(f"[FBXReader] Using FBX UnitScaleFactor: {fbx_scale} -> scale={unit_scale}")
            return unit_scale

        # Default: assume cm, convert to m
        print("[FBXReader] Using default: centimeters")
        return 0.01

    def _get_hips_default_height(self, content: str) -> Optional[float]:
        """Get the default Y translation of Hips joint (height in Y-up FBX)."""
        # Find Hips model section
        hips_match = re.search(
            r'Model:\s*\d+,\s*"Model::Hips"[^{]*\{.*?Properties70:\s*\{([^}]+)\}',
            content,
            re.DOTALL,
        )
        if not hips_match:
            return None

        props = hips_match.group(1)
        # Look for Lcl Translation property
        lcl_trans = re.search(
            r'P:\s*"Lcl Translation"[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)',
            props,
        )
        if lcl_trans:
            # Y is height in FBX Y-up coordinate system
            y_height = abs(float(lcl_trans.group(2)))
            return y_height

        return None

    def _extract_joints(self, content: str) -> Tuple[Dict[str, int], Dict[str, str]]:
        """Extract joint names and IDs from Model definitions."""
        joints = {}
        parents_dict = {}

        # Find all Model definitions (joints/bones)
        # Format: Model: ID, "Model::Name", "Type"
        model_pattern = re.compile(r'Model:\s*(\d+),\s*"Model::([^"]+)",\s*"(LimbNode|Root|Null)"')

        for match in model_pattern.finditer(content):
            model_id = int(match.group(1))
            model_name = match.group(2)
            joints[model_name] = model_id

        # Find parent-child relationships in Connections section
        # Format: C: "OO",child_id,parent_id
        conn_pattern = re.compile(r'C:\s*"OO",\s*(\d+),\s*(\d+)\s*$', re.MULTILINE)

        # Build reverse lookup
        id_to_name = {v: k for k, v in joints.items()}

        for match in conn_pattern.finditer(content):
            child_id = int(match.group(1))
            parent_id = int(match.group(2))

            child_name = id_to_name.get(child_id)
            parent_name = id_to_name.get(parent_id)

            if child_name and parent_name:
                parents_dict[child_name] = parent_name

        return joints, parents_dict

    def _extract_curves(self, content: str) -> Dict[int, Dict]:
        """Extract all AnimationCurve data."""
        curves = {}

        # Find AnimationCurve headers and extract their blocks with brace counting
        header_pattern = re.compile(r'AnimationCurve:\s*(\d+),\s*"AnimCurve::"')

        for match in header_pattern.finditer(content):
            curve_id = int(match.group(1))
            start_pos = match.end()

            # Find the opening brace
            brace_pos = content.find("{", start_pos)
            if brace_pos == -1:
                continue

            # Extract block using brace counting
            curve_body = self._extract_brace_block(content, brace_pos)
            if not curve_body:
                continue

            times = self._extract_array(curve_body, "KeyTime")
            values = self._extract_array(curve_body, "KeyValueFloat")

            if times is not None and values is not None:
                min_len = min(len(times), len(values))
                if min_len > 0:
                    curves[curve_id] = {
                        "times": times[:min_len],
                        "values": values[:min_len],
                    }

        return curves

    def _extract_brace_block(self, content: str, start: int) -> Optional[str]:
        """Extract content between matching braces starting at position start."""
        if content[start] != "{":
            return None

        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return content[start + 1 : i]
        return None

    def _extract_connections(self, content: str, joints: Dict[str, int]) -> Tuple[Dict, Dict]:
        """Extract animation connections."""
        # AnimCurve -> AnimCurveNode connections
        # Format: C: "OP",curve_id,node_id, "d|X"
        curve_to_node = {}
        conn_pattern1 = re.compile(r'C:\s*"OP",\s*(\d+),\s*(\d+),\s*"d\|([XYZ])"')

        for match in conn_pattern1.finditer(content):
            curve_id = int(match.group(1))
            node_id = int(match.group(2))
            axis = match.group(3)
            curve_to_node[curve_id] = (node_id, axis)

        # AnimCurveNode -> Model connections
        # Format: C: "OP",node_id,model_id, "Lcl Translation" or "Lcl Rotation"
        node_to_joint = {}
        joint_ids = set(joints.values())
        conn_pattern2 = re.compile(r'C:\s*"OP",\s*(\d+),\s*(\d+),\s*"Lcl\s*(Translation|Rotation)"')

        for match in conn_pattern2.finditer(content):
            node_id = int(match.group(1))
            model_id = int(match.group(2))
            prop_type = match.group(3)

            if model_id in joint_ids:
                curve_type = "T" if prop_type == "Translation" else "R"
                node_to_joint[node_id] = (model_id, curve_type)

        return curve_to_node, node_to_joint

    def _build_animation_data(
        self,
        curves: Dict[int, Dict],
        curve_to_node: Dict,
        node_to_joint: Dict,
        fps: float,
    ) -> Dict[Tuple[int, str, str], Dict[int, float]]:
        """Build final animation data structure."""
        anim_data = {}

        for curve_id, curve_data in curves.items():
            if curve_id not in curve_to_node:
                continue

            node_id, axis = curve_to_node[curve_id]

            if node_id not in node_to_joint:
                continue

            joint_id, curve_type = node_to_joint[node_id]

            # Convert times to frame indices
            frame_values = {}
            for t, v in zip(curve_data["times"], curve_data["values"]):
                # FBX time to seconds, then to frame
                seconds = t / FBX_TIME_UNIT
                frame = int(round(seconds * fps))
                frame_values[frame] = v

            anim_data[(joint_id, curve_type, axis)] = frame_values

        return anim_data

    def _extract_default_pose(self, content: str, joints: Dict[str, int]) -> Tuple[Dict, Dict]:
        """Extract default translation and rotation for each joint."""
        default_trans = {}
        default_rot = {}

        for joint_name, joint_id in joints.items():
            # Find the Model block for this joint
            pattern = rf"Model:\s*{joint_id}[^{{]*\{{([^}}]+Properties70[^}}]+)\}}"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                props_block = match.group(1)

                # Extract Lcl Translation
                trans_match = re.search(
                    r'"Lcl Translation"[^,]*,[^,]*,[^,]*,[^,]*,\s*'
                    r"(-?[\d.]+(?:e[+-]?\d+)?)\s*,\s*"
                    r"(-?[\d.]+(?:e[+-]?\d+)?)\s*,\s*"
                    r"(-?[\d.]+(?:e[+-]?\d+)?)",
                    props_block,
                )
                if trans_match:
                    default_trans[joint_id] = (
                        float(trans_match.group(1)),
                        float(trans_match.group(2)),
                        float(trans_match.group(3)),
                    )

                # Extract Lcl Rotation
                rot_match = re.search(
                    r'"Lcl Rotation"[^,]*,[^,]*,[^,]*,[^,]*,\s*'
                    r"(-?[\d.]+(?:e[+-]?\d+)?)\s*,\s*"
                    r"(-?[\d.]+(?:e[+-]?\d+)?)\s*,\s*"
                    r"(-?[\d.]+(?:e[+-]?\d+)?)",
                    props_block,
                )
                if rot_match:
                    default_rot[joint_id] = (
                        float(rot_match.group(1)),
                        float(rot_match.group(2)),
                        float(rot_match.group(3)),
                    )

        return default_trans, default_rot

    def _extract_array(self, text: str, array_name: str) -> Optional[np.ndarray]:
        """Extract a numeric array from FBX text block."""
        # Look for: ArrayName: *count { a: values... }
        pattern = rf"{array_name}:\s*\*(\d+)\s*\{{\s*a:\s*([^}}]+)\}}"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None

        count = int(match.group(1))
        values_str = match.group(2)

        # Parse comma-separated values (may span multiple lines)
        values_str = values_str.replace("\n", "").replace("\t", "")
        values = []
        for v in values_str.split(","):
            v = v.strip()
            if v:
                try:
                    values.append(float(v))
                except ValueError:
                    continue

        return np.array(values) if values else None

    def _get_frame_count(self, anim_data: Dict[Tuple[int, str, str], Dict[int, float]]) -> int:
        """Get total frame count from animation data."""
        max_frame = 0
        for frame_values in anim_data.values():
            if frame_values:
                max_frame = max(max_frame, max(frame_values.keys()))
        return max_frame + 1

    def _euler_zyx_to_quat(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """
        Convert Euler angles (ZYX intrinsic order, common in FBX) to quaternion (xyzw).

        This is equivalent to: Rot = Rz * Ry * Rx
        """
        cx, sx = np.cos(rx / 2), np.sin(rx / 2)
        cy, sy = np.cos(ry / 2), np.sin(ry / 2)
        cz, sz = np.cos(rz / 2), np.sin(rz / 2)

        # ZYX intrinsic = XYZ extrinsic
        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    def _load_binary_fbx(
        self, filepath: Path
    ) -> Tuple[List[str], np.ndarray, np.ndarray, float, np.ndarray]:
        """Load binary FBX using available backend."""
        errors = []

        # Try Blender Python API
        try:
            result = self._load_with_blender(filepath)
            self._backend = "blender"
            return result
        except ImportError as e:
            errors.append(f"blender (bpy): {e}")
        except Exception as e:
            errors.append(f"blender error: {e}")

        raise ImportError(
            "Binary FBX detected but no loader available. Tried:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nFor binary FBX, install:\n"
            "  pip install bpy       # Blender Python (large)\n"
            "\nOr convert to ASCII FBX using Blender/Maya."
        )

    def _load_with_blender(
        self, filepath: Path
    ) -> Tuple[List[str], np.ndarray, np.ndarray, float, np.ndarray]:
        """Load FBX using Blender Python API."""
        import bpy

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(filepath))

        armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
        if armature is None:
            raise ValueError("No armature found in FBX file")

        scene = bpy.context.scene
        fps = float(scene.render.fps)
        frame_start = int(scene.frame_start)
        frame_end = int(scene.frame_end)
        n_frames = frame_end - frame_start + 1

        if n_frames <= 0:
            raise ValueError(f"Invalid frame range: {frame_start} to {frame_end}")

        bones = list(armature.pose.bones)
        joint_names = [bone.name for bone in bones]
        n_joints = len(joint_names)

        bone_to_idx = {bone.name: i for i, bone in enumerate(bones)}
        parents = np.array(
            [bone_to_idx[bone.parent.name] if bone.parent else -1 for bone in bones],
            dtype=np.int32,
        )

        positions = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
        rotations = np.zeros((n_frames, n_joints, 4), dtype=np.float32)

        for frame_idx, frame in enumerate(range(frame_start, frame_end + 1)):
            scene.frame_set(frame)
            for joint_idx, bone in enumerate(bones):
                world_pos = armature.matrix_world @ bone.head
                positions[frame_idx, joint_idx] = [world_pos.x, world_pos.y, world_pos.z]

                world_rot = (armature.matrix_world @ bone.matrix).to_quaternion()
                rotations[frame_idx, joint_idx] = [
                    world_rot.x,
                    world_rot.y,
                    world_rot.z,
                    world_rot.w,
                ]

        print(f"[FBXReader] Loaded with Blender: {n_frames} frames, {n_joints} joints, {fps} fps")
        return joint_names, positions, rotations, fps, parents
