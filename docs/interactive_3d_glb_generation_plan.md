# AutoDex Interactive 3D GLB Generation Plan

작성일: 2026-07-05

이 문서는 object tracking 결과를 이용해 AutoDex gallery에서 로봇팔, 손,
object를 함께 보여주는 interactive 3D panel을 만들기 위한 asset 생성 절차를
정리한다. 결론부터 말하면, 현재 완료된 GoTrack object tracking output과
episode 내부 robot state를 조합하면 3D panel 구현이 가능하다. 다만 첫 구현은
모든 animation을 하나의 `.glb`에 억지로 넣기보다, 정적 mesh bundle인
`scene.glb`와 시간축 데이터인 `trajectory.json`을 함께 제공하는 구조가 가장
현실적이다.

## 1. 목표

Gallery의 main panel에서 다음을 interactive 3D로 보여준다.

- XArm robot arm
- Allegro 또는 Inspire hand
- tracked object mesh
- object pose trajectory
- robot arm/hand trajectory
- video playback bar와 동기화되는 3D timestep

최종적으로는 사용자가 video mode와 interactive 3D mode를 전환할 수 있어야
한다. interactive 3D mode에서는 video와 같은 frame index 또는 timestamp를
기준으로 robot/object state가 함께 움직인다.

## 2. 현재 가능한 것

현재 데이터와 코드 상태 기준으로 가능한 범위는 다음과 같다.

### 가능한 것

- object trajectory:
  - 입력: `<episode>/object_tracking/gotrack_output/world_pose_records.json`
  - 각 record에 `frame_index`, `time_sec`, `pose_world`, tracking 품질 지표가
    들어 있다.
- robot arm trajectory:
  - 입력: `<episode>/arm/state.npy` 또는 `<episode>/raw/arm/position.npy`
  - selected_100 샘플 기준 `arm/state.npy`는 video frame에 맞춘 `(N, 6)`
    배열이다.
- hand trajectory:
  - 입력: `<episode>/hand/state.npy` 또는 `<episode>/raw/hand/position.npy`
  - Allegro 샘플은 `(N, 16)`, Inspire 샘플은 `(N, 6)`이다.
- object visual mesh:
  - 보통 `/home/robot/shared_data/AutoDex/object/paradex/<object>/raw_mesh/<object>.obj`
  - texture가 있는 경우 `material.mtl`, `material_0.png`가 같은 폴더에 있다.
- robot/hand mesh:
  - URDF와 mesh가 `/home/robot/shared_data/AutoDex/content/assets/robot/` 아래에
    있다.

### 구현된 MVP

- episode 단위 3D asset exporter
  - 코드: `autodex/interactive_3d/episode_exporter.py`
  - CLI: `scripts/export_episode_3d_assets.py`
- Allegro episode 기준 robot arm/hand FK export
- GoTrack object pose를 `C2R.npy`로 robot-base frame에 정렬
- `scene.glb`, `preview.glb`, `trajectory.json`, `manifest.json` 생성

### 아직 구현이 필요한 것

- Inspire 계열 episode에 대한 추가 검증
- object pose smoothing 옵션
- gallery의 3D viewer 연결
- web playback에서 GLB node transform을 `trajectory.json`으로 업데이트하는 코드

## 3. 권장 output 구조

3D asset은 episode 입력 폴더 안에 직접 저장하거나, overlay video처럼 별도
root에 relative path를 보존해 저장할 수 있다. Gallery 배포와 재생 속도를
생각하면 별도 root가 낫다.

권장 경로:

```text
/home/robot/shared_data/AutoDex/interactive_3d/
  <relative_episode_path>/
    scene.glb
    preview.glb
    trajectory.json
    manifest.json
    thumbnails/
      frame_0000.png
      contact.png
```

예:

```text
/home/robot/shared_data/AutoDex/interactive_3d/
  allegro/wood_organizer/20260326_121427/
    scene.glb
    preview.glb
    trajectory.json
    manifest.json
```

파일 역할:

- `scene.glb`
  - robot link meshes, hand link meshes, object mesh를 포함하는 정적 mesh bundle.
  - 각 움직이는 link/object는 node 이름을 안정적으로 갖는다.
  - animation data는 넣지 않아도 된다.
- `trajectory.json`
  - frame별 robot joint 값, object pose, time, quality metric.
  - web viewer가 slider/playback에 맞춰 node transform을 업데이트한다.
- `preview.glb`
  - 대표 frame 또는 contact frame의 정적 snapshot.
  - pose thumbnail, turntable preview, 빠른 로딩용.
- `manifest.json`
  - episode id, hand type, object id, frame count, 파일 경로, schema version.

## 4. 왜 `scene.glb + trajectory.json`부터 시작하는가

`.glb` 하나에 full animation까지 넣을 수는 있다. glTF는 node별
translation/rotation/scale animation channel을 지원한다. 그러나 첫 구현에서는
다음 이유로 분리 구조가 더 낫다.

- video 재생 bar와 3D 재생 bar를 같은 timeline으로 동기화하기 쉽다.
- object tracking jitter를 raw/smoothed toggle로 바꾸기 쉽다.
- 실패 frame, missing frame, low-confidence frame을 UI에서 처리하기 쉽다.
- GLB 파일이 커지는 것을 막을 수 있다.
- robot URDF joint mapping을 수정해도 trajectory JSON만 재생성하면 된다.
- GitHub Pages/Hugging Face asset hosting에서 cache와 lazy loading이 쉽다.

따라서 단계별 권장은 다음이다.

1. `preview.glb`: contact frame 정적 snapshot.
2. `scene.glb + trajectory.json`: interactive trajectory viewer.
3. 필요할 때 `animated.glb`: glTF animation channel까지 포함한 단일 파일.

## 5. 입력 파일

### Episode root

예:

```text
/home/robot/shared_data/AutoDex/experiment/selected_100/
  allegro/wood_organizer/20260326_121427/
```

주요 입력:

```text
<episode>/object_tracking/gotrack_output/world_pose_records.json
<episode>/object_tracking/gotrack_output/summary.json
<episode>/pose_world.npy
<episode>/arm/state.npy
<episode>/hand/state.npy
<episode>/raw/timestamps/timestamp.npy
<episode>/raw/timestamps/frame_id.npy
<episode>/result.json
```

주의:

- `pose_world.npy`는 first-frame 또는 init pose에 가깝다. full object trajectory는
  `world_pose_records.json`을 사용해야 한다.
- `world_pose_records.json`의 `pose_world`는 object-to-world transform으로
  해석하는 것이 현재 overlay 코드와 맞다.
- `summary.json`의 `mesh_path`, `mesh_scale`,
  `external_unit_scale_to_meter`, `resolved_mesh_scale`은 object mesh scale
  검증에 사용한다.

### Object mesh

우선순위:

```text
/home/robot/shared_data/AutoDex/object/paradex/<object>/visual_mesh/<object>.obj
/home/robot/shared_data/AutoDex/object/paradex/<object>/raw_mesh/<object>.obj
/home/robot/shared_data/AutoDex/object/paradex/<object>/processed_data/mesh/simplified.obj
```

권장:

- 웹용은 `visual_mesh`가 있으면 먼저 사용한다.
- 없으면 `raw_mesh`를 사용한다.
- 너무 무거우면 `processed_data/mesh/simplified.obj`를 사용한다.
- texture 보존을 위해 `trimesh.load(path, process=False)`를 기본값으로 둔다.

### Robot URDF

선택 기준:

```text
allegro:
  /home/robot/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf

inspire:
  /home/robot/shared_data/AutoDex/content/assets/robot/inspire_description/xarm_inspire.urdf

inspire_left 계열:
  /home/robot/shared_data/AutoDex/content/assets/robot/inspire_left_description/xarm_inspire_left.urdf
```

확인된 joint 구조:

```text
xarm_allegro.urdf:
  22 non-fixed joints = 6 arm joints + 16 Allegro hand joints

xarm_inspire.urdf:
  12 non-fixed joints = 6 arm joints + 6 Inspire hand driver joints
```

데이터 구조:

```text
Allegro episode:
  arm/state.npy  -> (N, 6)
  hand/state.npy -> (N, 16)

Inspire episode:
  arm/state.npy  -> (N, 6)
  hand/state.npy -> (N, 6)
```

따라서 Allegro는 `6 + 16`을 URDF joint order에 넣으면 된다. 현재 확인한
`xarm_inspire.urdf`는 6 arm + 6 Inspire driver joint 구조이므로,
`arm/state.npy + hand/state.npy`의 12차원 qpos를 그대로 넣을 수 있다. 다만
Inspire episode는 실제 샘플로 한 번 더 GLB bounds와 손/object alignment를
검증해야 한다.

## 6. Coordinate frame 원칙

현재 object tracking output과 robot state는 서로 다른 기준 좌표계를 쓴다.
GoTrack의 `pose_world`는 camera-calibration world 기준이고, robot URDF FK는
robot base 기준이다. 따라서 interactive 3D panel의 canonical frame은
`robot_base`로 둔다.

권장 원칙:

- exporter 내부 canonical frame은 `robot_base`, meter 단위로 둔다.
- object mesh vertices도 meter 단위로 둔다.
- `world_pose_records.json`의 `pose_world`는 원본 GoTrack pose로 보존한다.
- GLB/trajectory playback에 쓰는 object pose는 `inv(C2R.npy) @ pose_world`로
  robot base에 변환한다.
- robot URDF는 robot base frame에서 FK를 계산한다.

확인해야 할 transform:

```text
<episode>/C2R.npy
<episode>/cam_param/extrinsics.json
```

기존 `src/validation/robothome/visualize_capture.py`에 따르면 `C2R.npy`는
camera world 안에서의 robot frame transform으로 저장되어 있다. 그러므로
camera-world point 또는 pose를 robot-base로 옮길 때는 `inv(C2R.npy)`를 적용한다.

샘플 검증:

```text
episode:
  /home/robot/shared_data/AutoDex/experiment/selected_100/allegro/wood_organizer/20260326_121427

frame 167:
  GoTrack object translation:
    [-0.1812, -0.0894, 0.8213]
  robot-base object translation after inv(C2R):
    [0.6047, -0.0528, 0.0380]
```

이 변환 후 object가 robot hand 근처로 들어오므로, gallery용 3D asset은
robot-base frame을 기준으로 저장하는 것이 맞다.

Three.js/glTF 표시 시 주의:

- AutoDex world는 보통 Z-up으로 보는 것이 자연스럽다.
- glTF viewer는 Y-up convention으로 다루는 경우가 많다.
- 첫 구현에서는 GLB에 AutoDex Z-up을 그대로 저장하고, viewer에서 camera/up
  설정을 맞추는 편이 디버깅이 쉽다.
- 필요하면 나중에 root node에 `T_gltf_from_autodex`를 한 번만 적용한다.

## 7. Export 절차

### Step 1. Episode validation

Exporter는 먼저 입력이 충분한지 검사한다.

필수:

```text
world_pose_records.json exists
arm/state.npy exists
hand/state.npy exists
object mesh exists
robot URDF exists
```

완료 판단:

- `world_pose_records.json`이 존재하고,
- `status == "ok"`인 record가 하나 이상 있고,
- 각 ok record에 `pose_world`가 있어야 한다.

권장 skip:

- `interactive_3d/<relative_episode_path>/manifest.json`이 있고,
- `scene.glb`, `trajectory.json`, `preview.glb`가 모두 존재하며,
- manifest의 input mtime/hash가 현재 입력과 같으면 재생성하지 않는다.

### Step 2. Timeline 구성

기본 timeline은 video frame index를 사용한다.

```text
frame_count = min(
  len(raw/timestamps/timestamp.npy),
  len(arm/state.npy),
  len(hand/state.npy)
)
```

샘플 확인:

```text
raw/timestamps/timestamp.npy -> (341,)
arm/state.npy                -> (341, 6)
hand/state.npy               -> (341, 16)  # Allegro sample
world_pose_records.json      -> 335 records
```

`world_pose_records.json`이 모든 video frame을 덮지 않을 수 있다. 이 경우:

- `frame_index`가 있는 record는 해당 frame에 매핑한다.
- 없는 frame은 직전 valid pose를 hold하거나, 양옆 valid pose 사이를 보간한다.
- raw mode에서는 hold만 적용한다.
- smoothed mode에서는 translation low-pass와 quaternion slerp/smoothing을
  적용한다.

### Step 3. Robot full qpos 생성

Allegro:

```text
full_qpos[t] = [
  arm/state.npy[t, 0:6],
  hand/state.npy[t, 0:16],
]
```

단, 실제 URDF joint order와 배열 order가 다를 수 있으므로 joint name 기반으로
넣는다.

URDF joint order 확인 예:

```python
import yourdfpy

urdf = yourdfpy.URDF.load(
    urdf_path,
    mesh_dir=urdf_path.parent,
    build_collision_scene_graph=False,
    load_collision_meshes=False,
)
print(list(urdf.actuated_joint_names))
```

Inspire:

```text
full_qpos[t, joint1..joint6] = arm/state.npy[t]
full_qpos[t, HAND_DRIVERS]   = mapped hand/state.npy[t]
other hand joints            = 0 or mimic-derived value
```

Inspire driver mapping은 `visualize_capture.py`의 다음 개념을 따른다.

```text
HAND_DRIVERS = [
  little_1, ring_1, middle_1, index_1, thumb_pitch, thumb_yaw
]
```

주의:

- selected_100의 Inspire `hand/state.npy`는 이미 radian-like frame-aligned
  값일 수 있다.
- `raw/hand/position.npy`는 controller raw integer일 수 있다.
- 따라서 exporter는 기본으로 `hand/state.npy`를 사용하고, raw 값을 직접 쓰는
  경로는 별도 옵션으로 둔다.

### Step 4. Static `preview.glb` 생성

대표 frame을 고른다.

우선순위:

1. contact/grasp frame을 알고 있으면 그 frame.
2. `result.json`이나 plan metadata에서 grasp timing을 얻을 수 있으면 그 frame.
3. 없으면 tracking record 중 inlier가 많고 residual이 낮은 중간 frame.
4. 그래도 없으면 첫 `status == "ok"` frame.

생성 절차:

1. URDF load.
2. 해당 frame의 `full_qpos`로 `urdf.update_cfg(qpos)`.
3. `urdf.scene.dump()`로 world transform이 적용된 robot link meshes를 얻는다.
4. object mesh를 load하고 `pose_world`를 적용한다.
5. robot meshes + object mesh를 `trimesh.Scene`에 넣는다.
6. `scene.export("preview.glb")`.

Skeleton code:

```python
import json
import numpy as np
import trimesh
import yourdfpy
from pathlib import Path

episode = Path("/home/robot/shared_data/AutoDex/experiment/selected_100/allegro/wood_organizer/20260326_121427")
obj = "wood_organizer"
urdf_path = Path("/home/robot/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf")
mesh_path = Path(f"/home/robot/shared_data/AutoDex/object/paradex/{obj}/raw_mesh/{obj}.obj")

records = json.loads((episode / "object_tracking/gotrack_output/world_pose_records.json").read_text())
arm = np.load(episode / "arm/state.npy")
hand = np.load(episode / "hand/state.npy")

record = next(r for r in records if r.get("status") == "ok" and r.get("pose_world"))
frame = int(record["frame_index"])
pose_world = np.asarray(record["pose_world"], dtype=np.float64)

urdf = yourdfpy.URDF.load(
    str(urdf_path),
    mesh_dir=str(urdf_path.parent),
    build_collision_scene_graph=False,
    load_collision_meshes=False,
)
qpos = np.concatenate([arm[frame], hand[frame]])
urdf.update_cfg(qpos)

scene = trimesh.Scene()
for i, mesh in enumerate(urdf.scene.dump()):
    scene.add_geometry(mesh, node_name=f"robot_mesh_{i}")

obj_mesh = trimesh.load(str(mesh_path), process=False)
if isinstance(obj_mesh, trimesh.Scene):
    for name, geom in obj_mesh.geometry.items():
        geom = geom.copy()
        geom.apply_transform(pose_world)
        scene.add_geometry(geom, node_name=f"object_{name}")
else:
    obj_mesh = obj_mesh.copy()
    obj_mesh.apply_transform(pose_world)
    scene.add_geometry(obj_mesh, node_name="object")

scene.export("preview.glb")
```

이 코드는 static snapshot용이다. trajectory playback용 `scene.glb`에는 link별
node 이름과 transform update가 필요하므로, 단순히 `scene.dump()`로 flatten한
mesh만 넣으면 robot link를 나중에 따로 움직일 수 없다.

### Step 5. Playback용 `scene.glb` 생성

interactive playback에서는 각 robot link와 object를 개별 node로 움직여야 한다.
따라서 두 가지 구현 방식이 있다.

#### 방식 A: link mesh를 매 frame CPU에서 새로 만들지 않고, GLB node transform만 업데이트

권장 방식이다.

`scene.glb`에 넣을 것:

- robot link별 visual mesh node
- object mesh node
- optional coordinate axes
- optional camera frame markers
- optional object trajectory line

`trajectory.json`에 넣을 것:

```json
{
  "schema_version": 1,
  "coordinate_frame": "autodex_world_z_up_meters",
  "fps_hint": 30.0,
  "nodes": {
    "object": "object",
    "robot_base": "robot"
  },
  "frames": [
    {
      "frame_index": 0,
      "time_sec": 0.0,
      "object_pose_world": [[...], [...], [...], [0, 0, 0, 1]],
      "robot_qpos": [...],
      "tracking_status": "ok",
      "num_inlier_anchors": 254,
      "mean_anchor_fit_residual_mm": 1.08
    }
  ]
}
```

Viewer에서 할 일:

1. `scene.glb`를 로드한다.
2. `trajectory.json`을 로드한다.
3. frame slider 값에 맞춰 object node matrix를 `object_pose_world`로 업데이트한다.
4. robot은 JS 쪽에서 FK를 다시 계산하거나, exporter가 미리 계산한 link pose를
   `trajectory.json`에 저장해 둔 값을 사용한다.

웹 구현 단순화를 위해서는 `trajectory.json`에 link pose를 넣는 것이 좋다.

권장 `trajectory.json` frame:

```json
{
  "frame_index": 0,
  "time_sec": 0.0,
  "object_pose_world": [[...]],
  "link_poses_world": {
    "link_base": [[...]],
    "link1": [[...]],
    "link2": [[...]],
    "base_link": [[...]],
    "link_0.0": [[...]]
  },
  "tracking": {
    "status": "ok",
    "num_inlier_anchors": 254,
    "mean_anchor_fit_residual_mm": 1.08
  }
}
```

이렇게 하면 browser에는 URDF parser가 필요 없다. Three.js는 node 이름별로
matrix만 업데이트하면 된다.

#### 방식 B: animated GLB 하나에 glTF animation channel을 넣기

두 번째 단계로 고려한다.

필요한 것:

- 각 움직이는 node마다 translation/rotation animation sampler.
- object node animation.
- robot joint 또는 link node animation.
- `pygltflib` 또는 직접 glTF JSON buffer 작성.

장점:

- 단일 `.glb`로 self-contained.
- 일반 glTF viewer에서도 timeline animation 재생 가능.

단점:

- 구현량이 크다.
- 수정/디버깅이 어렵다.
- video와 frame-perfect sync를 맞추려면 viewer side control이 여전히 필요하다.
- tracking quality toggle, raw/smoothed toggle이 어렵다.

따라서 처음부터 animated GLB를 목표로 잡기보다, `scene.glb +
trajectory.json`을 먼저 만들고 안정화한 뒤 변환기를 추가하는 것이 좋다.

### Step 6. `trajectory.json` 생성

Exporter는 frame별로 다음을 만든다.

1. `frame_index`
2. `time_sec`
3. `object_pose_world`
4. `robot_qpos`
5. `link_poses_world`
6. tracking 품질 지표

Robot link pose 계산:

```python
urdf.update_cfg(qpos)
for link_name in urdf.link_map:
    T = urdf.get_transform(link_name, urdf.base_link)
```

만약 robot base와 object world가 다르면:

```python
T_link_world = T_world_robot @ T_link_robot
```

여기서 `T_world_robot`는 calibration으로 확정해야 한다. 한 episode에서
object와 gripper가 접촉하는 frame을 열어보고, 물리적으로 맞는지 반드시
검증한다.

### Step 7. Object pose smoothing

현재 overlay에서 jitter가 관찰되었으므로, interactive 3D에서는 raw와 smoothed를
둘 다 저장하는 것이 좋다.

권장:

```json
{
  "object_pose_world_raw": [[...]],
  "object_pose_world_smooth": [[...]],
  "smoothing": {
    "translation": "median+lowpass",
    "rotation": "quaternion_slerp_lowpass",
    "window": 5
  }
}
```

기본 viewer는 smoothed를 사용하되, debug toggle로 raw를 볼 수 있게 한다.

주의:

- 논문/정량 분석에는 raw tracking output을 보존한다.
- smoothing은 visualization layer로 취급한다.
- failure frame은 보간했음을 metadata에 표시한다.

### Step 8. Manifest 생성

예:

```json
{
  "version": 1,
  "status": "complete",
  "episode_root": "/home/robot/shared_data/AutoDex/experiment/selected_100/allegro/wood_organizer/20260326_121427",
  "relative_episode_path": "allegro/wood_organizer/20260326_121427",
  "hand": "allegro",
  "object": "wood_organizer",
  "episode": "20260326_121427",
  "coordinate_frame": "robot_base",
  "frame_count": 335,
  "preview_frame_index": 167,
  "robot_geometry_count": 28,
  "outputs": {
    "scene_glb": "scene.glb",
    "preview_glb": "preview.glb",
    "trajectory_json": "trajectory.json"
  },
  "inputs": {
    "world_pose_records": ".../object_tracking/gotrack_output/world_pose_records.json",
    "arm_state": ".../arm/state.npy",
    "hand_state": ".../hand/state.npy",
    "C2R": ".../C2R.npy",
    "object_mesh": "/home/robot/shared_data/AutoDex/object/paradex/wood_organizer/raw_mesh/wood_organizer.obj",
    "urdf": "/home/robot/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf"
  }
}
```

Gallery는 이 manifest를 읽어 3D mode availability를 판단한다.

## 8. 구현 파일 제안

기존 tracking 코드를 건드리지 않고 새 코드를 추가한다.

```text
autodex/interactive_3d/
  __init__.py
  episode_exporter.py        # path resolve, URDF FK, GLB, trajectory export

scripts/export_episode_3d_assets.py
scripts/export_interactive_3d_batch.py
```

단일 episode CLI 예:

```bash
/home/robot/anaconda3/envs/autodex/bin/python scripts/export_episode_3d_assets.py \
  --episode-root /home/robot/shared_data/AutoDex/experiment/selected_100/allegro/wood_organizer/20260326_121427 \
  --output-root /home/robot/shared_data/AutoDex/interactive_3d \
  --overwrite
```

현재 생성된 샘플 output:

```text
/home/robot/shared_data/AutoDex/interactive_3d/allegro/wood_organizer/20260326_121427/
  manifest.json     4.0K
  preview.glb       3.7M
  scene.glb         3.7M
  trajectory.json   2.9M
```

Batch CLI는 다음 단계에서 추가한다.

로컬 시각화:

```bash
/home/robot/anaconda3/envs/autodex/bin/python scripts/serve_episode_3d_viewer.py \
  --asset-dir /home/robot/shared_data/AutoDex/interactive_3d/allegro/wood_organizer/20260326_121427 \
  --host 127.0.0.1 \
  --port 8788
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8788/
```

이 viewer는 `scene.glb`를 한 번 로드하고, `trajectory.json`의 frame별
`object_pose_world`와 `robot_geometry_poses_world`를 slider/play button에 맞춰
적용한다. 기본값은 자동 재생이며, `?autoplay=0`을 붙이면 정지 상태로 시작한다.
기본 화면은 궤적 선을 투영하지 않고 GLB 안의 object/robot mesh node 자체를
frame별 pose로 움직인다. 디버깅용 궤적 선이 필요할 때만 `?paths=1`을 붙인다.

구현은 Three.js 기반이며, 최종 gallery page에 들어갈 방식과 가깝다. `viser`는
Python-side 디버깅에는 편하지만, GitHub Pages에 올라갈 browser panel을
검증하려면 Three.js viewer가 더 직접적이다.

## 9. Gallery 연결

Gallery metadata에는 다음 필드를 추가한다.

```json
{
  "interactive_3d": {
    "available": true,
    "scene_glb": "https://.../interactive_3d/allegro/wood_organizer/20260326_121427/scene.glb",
    "preview_glb": "https://.../interactive_3d/allegro/wood_organizer/20260326_121427/preview.glb",
    "trajectory_json": "https://.../interactive_3d/allegro/wood_organizer/20260326_121427/trajectory.json"
  }
}
```

Viewer 동작:

1. 초기 mode는 video.
2. 사용자가 Interactive 3D를 선택하면 그때 `scene.glb`와 `trajectory.json`을
   lazy load한다.
3. video와 3D를 좌우로 함께 보여주는 모드에서는 같은 playback state를 공유한다.
4. frame slider가 바뀌면 video currentTime과 3D frame index를 같이 갱신한다.

## 10. Validation checklist

한 episode에서 반드시 확인해야 할 항목:

- `preview.glb`가 브라우저에서 열리는가.
- object 크기가 실제 robot/hand 대비 맞는가.
- object 위치가 gripper와 같은 workspace에 있는가.
- contact frame에서 object와 hand가 물리적으로 말이 되는가.
- `world_pose_records.json` frame index와 video frame이 맞는가.
- `arm/state.npy`, `hand/state.npy`가 video frame과 맞는가.
- Allegro 16 joint mapping이 URDF 손가락 순서와 맞는가.
- Inspire 6 driver mapping이 실제 손가락 움직임과 맞는가.
- GLB 파일 크기가 gallery 로딩에 부담되지 않는가.
- smoothing 후 object가 과하게 미끄러지거나 contact를 뚫지 않는가.

권장 debug output:

```text
interactive_3d/<relative_episode_path>/debug/
  frame_0000.png
  contact_frame.png
  top_view.png
  transform_report.json
```

`transform_report.json`에는 object bbox, robot bbox, mesh scale, root transform,
frame count, missing frame 수를 남긴다.

## 11. 위험 요소와 대응

### Object tracking jitter

문제:

- overlay에서 이미 jitter가 관찰되었다.
- 3D에서는 jitter가 더 눈에 띌 수 있다.

대응:

- raw/smoothed pose를 둘 다 저장한다.
- viewer 기본값은 smoothed.
- residual이 큰 frame은 warning color 또는 confidence 표시.

### Mesh unit mismatch

문제:

- object mesh가 meter인지 millimeter인지 섞이면 object 크기가 틀어진다.

대응:

- `summary.json`의 `external_unit_scale_to_meter`,
  `resolved_mesh_scale`, `mesh_extents`를 읽어 검증한다.
- bbox extent가 0.01m보다 작거나 2m보다 크면 scale warning.

### Robot/world frame mismatch

문제:

- robot URDF base와 object pose world가 같은 frame이 아닐 수 있다.

대응:

- 한 episode contact frame에서 시각 검증한다.
- 필요하면 `T_world_robot`를 manifest에 명시하고 모든 link pose에 적용한다.
- `C2R.npy`와 camera extrinsics를 이용해 image overlay와 3D 위치가 같은지 검증한다.

### Full GLB animation complexity

문제:

- glTF animation channel 생성은 구현이 길고 디버깅이 어렵다.

대응:

- 1차: `scene.glb + trajectory.json`.
- 2차: 필요할 때만 `animated.glb` exporter 추가.

## 12. 추천 구현 순서

1. 단일 Allegro episode에서 `preview.glb` 생성.
2. 같은 episode에서 `scene.glb + trajectory.json` 생성.
3. local Three.js viewer로 frame slider playback 확인.
4. object pose raw/smoothed toggle 추가.
5. Inspire episode 한 개로 hand driver mapping 검증.
6. batch exporter 추가.
7. gallery metadata에 interactive_3d asset URL 추가.
8. gallery Interactive 3D mode lazy loading 연결.
9. video + 3D side-by-side sync mode 추가.
10. 필요 시 animated `.glb` exporter 추가.

## 13. 결론

현재 object tracking이 마무리되었다면, interactive 3D panel에 필요한 핵심
데이터는 대부분 준비된 상태다. object는 `world_pose_records.json`, robot은
`arm/state.npy`와 `hand/state.npy`, geometry는 object OBJ와 robot URDF에서
얻으면 된다.

가장 현실적인 첫 산출물은 episode별 `preview.glb`, `scene.glb`,
`trajectory.json`, `manifest.json`이다. 이 구조는 GitHub Pages/Hugging Face
gallery와 잘 맞고, video sync, smoothing, quality overlay를 구현하기 쉽다.
