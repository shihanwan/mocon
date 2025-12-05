# mocon

**Mo**tion **C**apture **Con**verter - converts mocap files to AMASS/ASAP format

## Installation

```bash
cd mocon
poetry install
```

That's it! ASCII FBX files (most common) are supported out of the box with no extra dependencies.

### For Binary FBX (optional)

If you have binary FBX files, you'll need:

```bash
pip install bpy       # Blender Python (large but reliable)
```

## Usage

```bash
# Convert FBX to AMASS format
mocon input.fbx

# Specify output path
mocon input.fbx -o output.npz

# Print file info without converting
mocon input.fbx --info

# Show joint mapping details
mocon input.fbx --show-mapping

# Resample to 30 FPS
mocon input.fbx --fps 30

# Use a different joint mapping
mocon input.fbx --mapping cmu
mocon input.fbx --mapping mixamo_prefixed
mocon input.fbx --mapping ./my_custom_mapping.json

# List available mappings
mocon --list-mappings
```

## Joint Mappings

mocon maps FBX joint names to SMPL's 24 joints. Different mocap software uses different naming conventions.

### Built-in Mappings

| Mapping | Description |
|---------|-------------|
| `default` | Mixamo/Xsens/Rokoko (most common) |
| `mixamo_prefixed` | Mixamo with `mixamorig:` prefix |
| `cmu` | CMU Motion Capture Database |
| `blender_rigify` | Blender's Rigify addon |

### Custom Mappings

Create a JSON file:

```json
{
  "name": "My Custom Rig",
  "description": "Mapping for my custom skeleton",
  "joints": {
    "MyHipBone": "Pelvis",
    "MyLeftThigh": "L_Hip",
    "MyRightThigh": "R_Hip"
  }
}
```

Use it:

```bash
mocon input.fbx --mapping ./my_mapping.json
```

### SMPL's 24 Joints

```
Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2,
L_Ankle, R_Ankle, Spine3, L_Foot, R_Foot, Neck,
L_Collar, R_Collar, Head, L_Shoulder, R_Shoulder,
L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hand, R_Hand
```

## Using with ASAP

After conversion:

```bash
# 1. Convert your FBX
mocon my_animation.fbx -o my_motion.npz

# 2. Copy to ASAP's motion folder
cp my_motion.npz /path/to/ASAP/humanoidverse/data/motions/raw_tairantestbed_smpl/

# 3. Run ASAP's retargeting pipeline
cd /path/to/ASAP
python scripts/data_process/fit_smpl_motion.py +robot=g1/g1_29dof_anneal_23dof
```

## Supported Formats

| Input | Output | Notes |
|-------|--------|-------|
| FBX (ASCII) | AMASS NPZ | Native support, no dependencies |
| FBX (Binary) | AMASS NPZ | Requires `bpy` |

## Output Format

The output NPZ file contains:

- `mocap_framerate`: Frame rate (float)
- `poses`: Joint rotations in axis-angle format (N, 72)
- `trans`: Root translation (N, 3)
- `betas`: SMPL shape parameters (16,) - zeros
- `gender`: "neutral"

## License

MIT
