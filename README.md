# SO-101 Imitation Learning Dev Toolkit

[SO-101](https://github.com/TheRobotStudio/SO-ARM100) 로봇팔을 위한 end-to-end 모방학습(imitation learning) 툴킷입니다. MuJoCo/URDF 로봇 모델, 텔레옵 기반 데이터 수집(시뮬레이션·실물), 녹화 데이터셋을 검증하는 QA 레이어, ACT policy 학습·평가, MuJoCo 기반 추론, 그리고 이 네 단계를 브라우저에서 실행하는 Streamlit 대시보드로 구성됩니다.

## 파이프라인 개요

```
pipeline/
├── 1_data_collection/   leader→follower teleoperation + LeRobotDataset recording
├── 2_qa/                TeamRobotDataset: outlier/integrity checks, visualization
├── 3_training/          ACT (lerobot-train) checkpoint analysis
└── 4_inference/         MuJoCo rollout of trained ACT checkpoints
```

### ① 데이터 수집 ([`pipeline/1_data_collection/`](pipeline/1_data_collection/))

- `run_teleop_real.py` — 실물 SO-101 leader arm → MuJoCo 시뮬레이션 follower. `action`/`observation.state`와 손목 카메라 영상을 [LeRobot](https://github.com/huggingface/lerobot) `LeRobotDataset`(v3.0, parquet 기반)으로 녹화합니다. 물체를 매 에피소드 같은 위치(추론과 동일)에서 시작하는 기본 모드와 `--randomize-object`로 켜는 위치 randomizer, 시작/폐기 키보드 조작, 실시간 카메라 미리보기를 포함합니다.
- `run_teleop_real_to_real.py` — 실물 leader arm → 실물 follower arm (시뮬레이션 없음). USB 손목 카메라로 병 파지·따르기(grasp-and-pour) 시연 데이터를 수집하는 데 사용했으며, leader의 실측 관절 범위를 follower의 실측 캘리브레이션 범위에 맞춰 비례 스케일링합니다.
- `run_teleop_follower.py` (repo 루트) — 반대 방향입니다. MuJoCo 뷰어에서 마우스로 움직이는 가상 leader arm이 시리얼로 실물 follower arm을 제어합니다. 텔레옵 전용이며 녹화는 하지 않습니다. ([아래 섹션](#leaderfollower-텔레옵-마우스-조작) 참고)

### ② QA 및 검증 ([`pipeline/2_qa/`](pipeline/2_qa/))

`team_robot_dataset.py`에 정의된 **`TeamRobotDataset`**은 `lerobot.datasets.LeRobotDataset`을 감싸는 composition wrapper입니다(base class는 수정하지 않음). 추가 기능:

- `check_integrity()` — 모든 프레임에서 NaN/Inf 값을 검사합니다(로드 시 자동 실행).
- `compute_custom_stats()`, `detect_outliers()` — raw 값 기준 관절별 z-score 이상치 탐지.
- `detect_outliers_delta()` — 프레임 간 변화량(delta) z-score 탐지. raw 값 z-score로는 잡히지 않는 간헐적인 계단형 전이를 잡아냅니다.
- `get_clean_indices()` — 위 결과를 합쳐 원본을 건드리지 않는 clean/excluded 인덱스 분할을 만듭니다. `torch.utils.data.Subset`에 바로 사용할 수 있습니다.
- `split_episodes()` — 에피소드 단위 train/val 분할.
- `normalize()` / `unnormalize()`, `visualize_episode()`, `plot_action_distribution()`, `describe_preprocessing()` — ACT 학습 경로에서 실제로 쓰이는 정규화·전처리 방식을 그대로 반영한 QA용 진단 도구.

`run_outlier_detection.py`, `run_visualization.py`, `run_clean_subset_example.py`가 녹화된 데이터셋에 위 기능을 실행합니다. 출력 예시는 [`reports/`](reports/)에 있습니다([`outlier_report.md`](reports/outlier_report.md), 궤적 플롯, action 분포 히스토그램).

### ③ ACT 학습 ([`pipeline/3_training/`](pipeline/3_training/))

학습 자체는 upstream `lerobot-train` CLI로 [`config/`](config/)의 YAML 설정을 사용해 실행합니다(pick-and-place는 `train_main_run_config.yaml`, 병 작업은 `train_bottle_config.yaml`). 이 폴더에는 읽기 전용 분석 스크립트만 있습니다.

- `run_val_loss_check.py` — `lerobot-train`의 forward/loss 경로를 그대로 재구성해 체크포인트별 held-out validation loss를 계산합니다(`lerobot-train`은 이 설정에서 held-out 평가를 기본 제공하지 않음).
- `plot_checkpoint_sweep.py` — 위 결과를 플롯으로 그립니다.

### ④ 추론 ([`pipeline/4_inference/`](pipeline/4_inference/))

- `run_inference_mujoco.py` — 학습된 ACT 체크포인트를 불러와 MuJoCo 안에서 closed-loop pick-and-place rollout을 실행합니다(뷰어 모드 또는 headless 스모크 테스트 모드).
- `run_inference_mujoco_bottle.py` — 병 작업 체크포인트용 버전입니다. 이 policy는 실제 웹캠 영상으로 학습됐기 때문에 MuJoCo 합성 렌더로 실행하면 파이프라인이 정상 동작하는지(NaN/Inf 없음, 유한한 action)만 확인할 수 있으며, 실제 파지·따르기 성공 여부는 검증하지 않습니다. 이 체크포인트의 실물 로봇 실행은 별도 프로젝트에서 다룹니다.

## 웹 대시보드 ([`dashboard/`](dashboard/))

위 네 단계를 감싸는 Streamlit 앱(`streamlit run dashboard/app.py`)으로, 팀원이 터미널 없이 전체 루프를 실행할 수 있습니다.

1. **데이터 수집** — `run_teleop_real.py`를 subprocess로 시작/중지하며, 로컬 COM 포트 설정 폼을 제공합니다.
2. **QA 검증** — `TeamRobotDataset`을 직접 import해(subprocess 없이 순수 계산) 이상치/무결성 검사를 실행합니다.
3. **ACT 학습** — 신규 또는 이어서(resume) `lerobot-train` 실행을 subprocess로 시작합니다. GPU 리소스 lock으로 추론과 동시에 실행되지 않습니다.
4. **추론** — `run_inference_mujoco.py`를 동기식 headless 헬스 체크 또는 subprocess 기반 인터랙티브 뷰어 세션으로 실행합니다.
5. **전처리 구조** — `TeamRobotDataset.describe_preprocessing()`의 읽기 전용 뷰입니다. 일반적인 IL 전처리 단계(resize, augmentation, action chunking, frame stacking 등)가 QA 레이어, ACT policy, `lerobot-train` CLI 중 실제로 어디서 처리되는지(또는 처리되지 않는지) 보여줍니다.

단계 간 리소스 충돌(시리얼 포트 / GPU / MuJoCo 뷰어)은 `dashboard/lib/process_manager.py`의 lock 시스템으로 직렬화됩니다. 대시보드는 파이프라인 모듈을 읽기 전용으로만 import하며 수정하지 않습니다.

## 로봇 모델 ([`robot_model/`](robot_model/))

SO-101 로봇팔의 URDF 및 MuJoCo(MJCF) 모델입니다.

- Onshape에서 설계한 CAD 모델을 [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) 플러그인으로 변환해 생성했습니다.
- 생성된 URDF는 `package://...` 대신 상대 경로로 mesh를 참조하도록 수정했습니다.
- 시뮬레이션·플래닝 중 충돌 동작에 문제가 있어 base의 collision mesh는 제거했습니다.
- `scene.xml`은 범용 pick-and-place 씬(`pick_object`/`place_plate`), `scene_bottle.xml`은 병 파지·따르기 씬(`bottle`/`cup`), `scene_teleop.xml`은 마우스 텔레옵용으로 SO-101 두 대(`leader_`/`follower_` prefix)를 배치한 씬입니다.

### 캘리브레이션

`scene.xml`은 `so101_new_calib.xml`을 include합니다. 이 캘리브레이션에서는 각 관절의 가상 영점이 관절 가동 범위의 **중앙**에 설정됩니다.

> 원본 SO-ARM100 모델에는 로봇이 **수평으로 완전히 뻗은 자세**를 영점으로 하는 `so101_old_calib.xml`도 있지만, 이 repo에는 포함되어 있지 않습니다. 사용하려면 파일을 추가한 뒤 `scene.xml`의 include 대상을 바꾸면 됩니다.

### 모터 파라미터

로봇에 사용된 STS3215 모터의 물성값은 [Open Duck Mini 프로젝트](https://github.com/apirrone/Open_Duck_Mini)에서 가져왔습니다.

### 그리퍼 참고

LeRobot에서 그리퍼는 **linear joint**로 표현됩니다.

* `0` = 완전히 닫힘
* `100` = 완전히 열림

이 매핑은 현재 URDF와 MuJoCo 파일에는 **아직 반영되어 있지 않습니다**.

## Leader/Follower 텔레옵 (마우스 조작)

`run_teleop_follower.py`(repo 루트)는 `robot_model/scene_teleop.xml`을 로드합니다. 이 씬에는 액추에이터가 없고 중력 보상(`gravcomp="1"`)이 완전히 적용된 leader와 follower, 두 대의 SO-101이 있습니다. 매 제어 스텝마다 leader의 관절각을 MuJoCo에 표시되는 follower와 시리얼로 연결된 실물 follower arm 양쪽에 복사합니다.

leader arm은 공중에 떠서 놓아둔 자세를 그대로 유지합니다. MuJoCo 뷰어에서 마우스로 움직입니다.

1. `Ctrl` + 더블클릭으로 leader arm의 링크를 선택합니다.
2. `Ctrl` + 우클릭 드래그로 선택한 body를 이동하고, `Ctrl` + `Shift` + 우클릭 드래그로 회전합니다.

```bash
python run_teleop_follower.py --port COM7
```

캘리브레이션되지 않은 실물 follower에 처음 연결하면 인터랙티브 캘리브레이션이 진행됩니다(팔을 중간 자세로 옮긴 뒤 각 관절을 가동 범위 끝까지 움직임). 이후 실행에서는 `--id`로 지정한 캘리브레이션 파일을 재사용합니다. 안전을 위해 `--max-relative-target`(기본값 15)으로 한 번의 명령에서 실물 팔이 움직일 수 있는 양을 제한합니다. 이 스크립트는 텔레옵 전용이며 데이터셋을 녹화하지 않습니다. 녹화는 `pipeline/1_data_collection/`을 참고하세요.

## 설치

전체 과정(`environment.yml` 기반 conda 환경, CUDA용 torch, 하드웨어 연결)은 [`docs/NEW_TEAMMATE_SETUP.md`](docs/NEW_TEAMMATE_SETUP.md)를 참고하세요. 핵심 스택: Python 3.11, `lerobot`(ACT policy + `LeRobotDataset`), `mujoco`, `streamlit`, Feetech STS3215 서보용 `feetech-servo-sdk`/`pyserial`, PyTorch(CUDA, conda 환경과 별도로 설치).

`docs/`의 기타 문서:

- [`TELEOP_DATA_COLLECTION.md`](docs/TELEOP_DATA_COLLECTION.md)
- [`REAL_LEADER_SETUP.md`](docs/REAL_LEADER_SETUP.md)
- [`CAMERA_TUNING_AND_TROUBLESHOOTING.md`](docs/CAMERA_TUNING_AND_TROUBLESHOOTING.md)
- [`GRIPPER_SLIP_TROUBLESHOOTING.md`](docs/GRIPPER_SLIP_TROUBLESHOOTING.md)
- [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)

---

이슈 등록이나 개선 기여는 언제든 환영합니다!
