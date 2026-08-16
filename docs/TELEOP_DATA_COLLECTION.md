# 텔레오퍼레이션 데이터 수집 → LeRobot 데이터셋

## 목적

`run_teleop.py`로 leader/follower SO101 팔을 조작(mouse teleoperation)하면서 발생하는
관절 데이터를 수집하고, 이를 [LeRobot](https://github.com/huggingface/lerobot) 표준
데이터셋 포맷(`LeRobotDataset`, parquet 기반)으로 디스크에 저장한다. 저장된 데이터셋의
구조와 구성을 직접 확인(검증)할 수 있도록 스키마와 확인 방법을 정리한다.

## 전체 흐름

```
leader 팔 조작 (마우스)
      │  매 물리 스텝(500Hz)
      ▼
run_teleop.py 메인 루프
  ├─ observation.state = follower의 현재 관절각(qpos)  ← mj_step 적용 전
  ├─ action            = leader의 관절각(qpos)          ← follower로 그대로 전달
  ├─ mj_step, viewer.sync()
  └─ 녹화 중이면 30fps로 다운샘플링해 dataset.add_frame(...)
      │  'S' 키: 녹화 종료 시
      ▼
dataset.save_episode()   → episode 1개를 parquet에 기록
      │  프로그램 종료 시(try/finally)
      ▼
dataset.finalize()       → parquet writer를 닫아 디스크에 완전히 flush
      │
      ▼
./datasets/so101_teleop/  (LeRobotDataset 포맷)
```

## 사용법

```bash
python run_teleop.py --task "설명 문구" --repo-id local/so101_teleop --root ./datasets/so101_teleop
```

- **S 키**: episode 녹화 시작/종료 (종료 시 저장)
- **X 키**: 현재 녹화 중인 episode를 저장하지 않고 버림
- 같은 `--root` 경로로 다시 실행하면 기존 데이터셋을 이어서 기록한다(episode_index가 이어짐).
- 30~50 episode를 모으려면 프로그램을 계속 띄워둔 채 S로 여러 번 반복하거나,
  여러 번에 걸쳐 스크립트를 재실행하면 된다(하드 제한 없음).

## 기록되는 데이터 스키마

| feature 이름 | dtype | shape | 설명 |
|---|---|---|---|
| `action` | float32 | (6,) | leader에서 읽은 목표 관절각. names: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| `observation.state` | float32 | (6,) | 그 스텝 시작 시점 follower의 실제 관절각(위와 동일한 6개 조인트) |
| `timestamp` | float32 | (1,) | episode 내 상대 시간(초), `frame_index / fps`로 자동 계산 |
| `frame_index` | int64 | (1,) | episode 내 프레임 순번(0부터) |
| `episode_index` | int64 | (1,) | 전체 데이터셋 내 episode 순번(0부터) |
| `index` | int64 | (1,) | 데이터셋 전체 프레임에 대한 전역 순번 |
| `task_index` | int64 | (1,) | `meta/tasks.parquet`에서 task 문자열을 가리키는 인덱스 |

- 카메라 이미지(`observation.images.*`)는 **포함하지 않음** (현재 씬에 카메라 미정의, 사용자 확인 후 관절 상태만 기록하기로 결정).
- `fps=30`으로 다운샘플링해서 저장한다 (물리 시뮬레이션 자체는 500Hz로 돈다).

## 디스크 상 구조

`--root`로 지정한 폴더 아래 실제로 생성되는 파일들 (3 episode, 총 63 frame으로 테스트한 예):

```
datasets/so101_teleop/
├── data/
│   └── chunk-000/
│       └── file-000.parquet          # 모든 episode의 프레임(action/observation.state/...)이 누적 저장됨
└── meta/
    ├── info.json                     # fps, robot_type, feature 스키마, total_episodes/total_frames 등
    ├── tasks.parquet                 # task 문자열 ↔ task_index 매핑
    ├── stats.json                    # feature별 전체 통계(min/max/mean/std/quantile)
    └── episodes/
        └── chunk-000/
            └── file-000.parquet      # episode별 메타(length, task, chunk/file 위치, 통계 등)
```

(카메라를 쓰지 않으므로 `videos/` 폴더는 생성되지 않는다.)

`meta/info.json` 예시(3 episode 기록 후):

```json
{
  "codebase_version": "v3.0",
  "robot_type": "so101_follower",
  "total_episodes": 3,
  "total_frames": 63,
  "total_tasks": 1,
  "fps": 30,
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": null,
  "features": { "action": {...}, "observation.state": {...}, "timestamp": {...}, ... }
}
```

## 데이터 직접 확인하는 방법

### 1) LeRobotDataset API로 로드 (권장)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/so101_teleop", root="./datasets/so101_teleop")

print(ds.meta.total_episodes, ds.meta.total_frames)
print(ds.features)              # 전체 feature 스키마
print(len(ds))                  # 전체 프레임 수

sample = ds[0]                  # 첫 프레임
print(sample["action"], sample["observation.state"])
```

### 2) parquet 파일을 pandas로 직접 열어보기

```python
import pandas as pd

df = pd.read_parquet("./datasets/so101_teleop/data/chunk-000/file-000.parquet")
print(df.columns.tolist())
# ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
print(df.head())

episodes = pd.read_parquet("./datasets/so101_teleop/meta/episodes/chunk-000/file-000.parquet")
print(episodes[["episode_index", "length", "tasks"]])

tasks = pd.read_parquet("./datasets/so101_teleop/meta/tasks.parquet")
print(tasks)
```

### 3) 메타 정보만 빠르게 확인

```bash
cat ./datasets/so101_teleop/meta/info.json
```

## 주의사항

- **`finalize()` 호출 필수**: `LeRobotDataset`은 episode 메타데이터와 프레임 데이터를 parquet
  writer에 버퍼링해뒀다가 `finalize()`를 호출해야 디스크에 완전히 flush된다. `run_teleop.py`는
  `try/finally`로 감싸 정상 종료·Ctrl+C·예외 상황 모두에서 `finalize()`가 호출되도록 처리했다.
  (검증 중 이걸 빠뜨리면 재실행 시 이전 episode를 못 읽어와 Hugging Face Hub에 네트워크 요청을
  시도하다 실패하는 문제를 실제로 재현했음.)
- **환경 이슈**: `mujoco_env`의 `pandas`가 3.0.5로 설치되어 있어 DLL load 실패로 `lerobot` import
  자체가 안 되던 문제가 있었다. `pandas>=2.2.2,<2.4.0`(lerobot이 명시한 범위, 실제로는 2.3.3 설치)로
  다운그레이드해 해결했다. 이 과정에서 딸려 올라간 `numpy`도 `opencv-python-headless` 요구사항에
  맞춰 `2.2.6`으로 재고정했다.
- 카메라 관측을 나중에 추가하려면 `build_teleop_scene.py`에 오프스크린 카메라를 추가하고,
  `mujoco.Renderer`로 매 기록 프레임을 렌더링해 `observation.images.<cam>` feature로 넣어야 한다
  (ffmpeg 미설치 상태이므로 `use_videos=False`로 PNG 저장하거나 ffmpeg 설치 후 비디오 인코딩 사용).
