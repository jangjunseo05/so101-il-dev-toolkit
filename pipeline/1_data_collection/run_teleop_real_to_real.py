"""
SO101 실물 리더암 -> 실물 팔로워암 텔레오퍼레이션 + 녹화.

run_teleop_real.py(실물 리더암 -> MuJoCo 시뮬레이션 팔로워)와 달리, 이 스크립트는
팔로워도 실물이다. MuJoCo/물리 시뮬레이션을 전혀 쓰지 않는다 -- 두 팔 다 실물이라
시뮬레이션할 대상이 없다. 대신:
  - lerobot.teleoperators.so_leader.SO101Leader로 실물 리더암을 읽고
  - lerobot.robots.so_follower.SO101Follower로 실물 팔로워암에 명령을 보내고
  - 팔로워 그리퍼에 달린 실물 USB 카메라(lerobot.cameras.opencv.OpenCVCamera,
    SO101Follower의 cameras= 설정을 통해 자동으로 연결/캡처됨)로 손목 이미지를
    얻는다.

리더/팔로워는 서로 다른 실물 개체라 캘리브레이션된 실제 가동범위(도 단위)가
다를 수 있다(run_teleop_real.py의 leader_action_to_radians()가 리더 실측 range를
MuJoCo 팔로워 모델 range에 맞춰 비례 스케일링했던 것과 같은 이유) -- 이 스크립트는
리더의 실측 range를 팔로워의 실측 range에 비례 스케일링한다(leader_action_to_follower_action()).
그리퍼는 양쪽 다 자기 자신의 완전열림/완전닫힘 기준 0~100% 정규화라 그대로
전달한다(범위 재매핑 불필요).

action/observation.state는 라디안이 아니라 lerobot의 기본 단위(5개 관절은 도,
그리퍼는 0~100%)로 그대로 저장한다 -- 물리 엔진이 없어 라디안으로 바꿀 이유가
없고, get_action()/get_observation()/send_action()이 원래 쓰는 단위 그대로다.
학습 시 정규화(meta.stats 기반 MEAN_STD)는 단위를 신경 쓰지 않으므로 문제없다.
단, 나중에 실물 카메라 기반 추론 스크립트를 만들 때는 반드시 같은 단위(도/%)를
그대로 써야 한다.

사용법:
  python run_teleop_real_to_real.py --leader-port COM5 --follower-port COM7 --camera-index 0

  카메라 인덱스는 `lerobot-find-cameras opencv`로 먼저 확인할 것. 리더/팔로워
  캘리브레이션 파일이 없으면 최초 연결 시 대화형 캘리브레이션이 진행된다
  (run_teleop_real.py/run_teleop_follower.py와 동일한 흐름).

조작법:
  - 실물 리더암을 손으로 움직이면 실물 팔로워암이 그대로 추종한다.
  - S 키: episode 녹화 시작/종료(종료 시 LeRobot 데이터셋으로 저장). 카메라를
    쓰는 경우 MuJoCo 뷰어가 없으므로 손목 카메라 미리보기 창(tkinter)에 키
    입력을 바인딩했다 -- **미리보기 창을 한 번 클릭해 포커스를 준 뒤에 눌러야
    인식된다.** --no-camera 모드에서는 GUI 창을 아예 안 띄우고(GUI 이벤트
    펌프가 실제 서보 이동과 겹치면 시리얼 통신이 불안정해짐이 실측 확인됨)
    콘솔 키 입력으로 대체한다 -- 이 경우 **콘솔 창에 포커스가 있어야 하며,
    종료는 Ctrl+C로 한다.**
  - X 키: 녹화 중인 episode를 저장하지 않고 버림.
  - MuJoCo 시뮬레이션의 pick_object 랜덤 재배치 같은 자동 물체 재배치 기능은
    없다(실물 객체를 프로그램이 옮길 방법이 없음) -- episode 저장/폐기 후 물체는
    사람이 직접 재배치해야 한다.
"""

import argparse
import ctypes
import multiprocessing as mp
import queue
import signal
import sys
import time
import traceback
from pathlib import Path


def _setup_file_logging() -> str:
    """세션 시작 시 프로젝트 루트의 logs/ 아래 session_YYYYMMDD_HHMMSS.log를 만들고,
    이 프로세스의 stdout/stderr를 콘솔 + 그 파일에 동시에 쓰도록 한다
    (run_teleop_real.py와 동일한 로직).
    """
    import datetime
    import os
    import sys

    # so101_web/pipeline/1_data_collection/run_teleop_real_to_real.py -> so101_web/logs
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    )
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)

        def flush(self):
            for s in self._streams:
                s.flush()

        def isatty(self):
            return self._streams[0].isatty()

        def fileno(self):
            return self._streams[0].fileno()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[run_teleop_real_to_real] 세션 로그 시작: {log_path}")
    return log_path


# Windows는 multiprocessing이 "spawn" 방식이라, 아래 모듈 top-level 코드
# 전체(이 로깅 설정 포함)가 자식 프로세스(데이터셋 기록 전용, 아래
# _dataset_writer_process 참고)에서도 그대로 다시 실행된다. 로그 파일을
# 자식이 또 하나 만들면 세션당 로그가 중복 생성되므로, 진짜 메인 프로세스일
# 때만 연다 -- DLL 우회/xxhash shim(_suppress_windows_dll_popups(),
# _ensure_xxhash_importable())은 자식도 각자 필요(자식이 torch/datasets를
# import하므로)해서 그대로 양쪽에서 실행한다.
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

if _IS_MAIN_PROCESS:
    _LOG_PATH = _setup_file_logging()
else:
    _LOG_PATH = None


def _suppress_windows_dll_popups() -> None:
    """일부 Windows 환경(애플리케이션 제어 정책/백신)에서 서드파티 패키지가 쓰는
    컴파일된 DLL 로딩이 차단되면 "잘못된 이미지" 하드 에러 팝업이 뜨며 스크립트가
    멈출 수 있다(run_teleop_real.py에서 mujoco 플러그인/xxhash로 실제 겪은 문제).
    이 스크립트는 mujoco를 쓰지 않지만 lerobot -> datasets -> xxhash, 그리고
    opencv(cv2)의 네이티브 확장도 같은 부류의 DLL이라 동일한 정책에 걸릴 수 있으므로,
    프로세스 전역으로 팝업을 억제해 (뜨더라도) 조용한 예외로만 받게 한다.
    """
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_NOOPENFILEERRORBOX = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
    )


_suppress_windows_dll_popups()


def _ensure_xxhash_importable():
    """lerobot -> datasets가 쓰는 xxhash의 컴파일된 확장 모듈이 애플리케이션 제어
    정책에 막히는 경우 hashlib 기반 대체 모듈로 바꿔치기한다(run_teleop_real.py와
    동일한 우회). mujoco와 무관하게 이 프로젝트 환경에서 독립적으로 발생하는
    문제라 이 스크립트에도 그대로 필요하다.
    """
    try:
        import xxhash  # noqa: F401
        return
    except ImportError as exc:
        print(f"[run_teleop_real_to_real] xxhash 로딩 실패, hashlib 기반 대체 모듈 사용: {exc}")

    import hashlib
    import types

    class _Xxh64Shim:
        def __init__(self, seed: int = 0):
            self._h = hashlib.blake2b(digest_size=8)

        def update(self, data) -> None:
            if isinstance(data, str):
                data = data.encode("utf-8")
            self._h.update(data)

        def hexdigest(self) -> str:
            return self._h.hexdigest()

        def digest(self) -> bytes:
            return self._h.digest()

    shim = types.ModuleType("xxhash")
    shim.xxh64 = _Xxh64Shim
    sys.modules["xxhash"] = shim


import numpy as np

_ensure_xxhash_importable()
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

# 의도적으로 여기서 `from lerobot.datasets.lerobot_dataset import LeRobotDataset`를
# import하지 않는다. 진단 결과(대화 기록 참고): LeRobotDataset을 이 모듈
# top-level에서 import하면 torch/pyarrow/datasets/huggingface_hub 같은 무거운
# 라이브러리가 이 프로세스(리더/팔로워 50Hz 실시간 시리얼 통신을 담당)에 같이
# 로드되어, 그 라이브러리들의 백그라운드 스레드/GIL 점유가 시리얼 통신 타이밍을
# 깨는 것으로 실측 확인됨(순수 폴링/부하 테스트는 전부 100% 성공했지만 이 import를
# 추가하는 순간 실패율이 급증했다 -- CUDA 비활성화로도 재현돼 CUDA가 원인은 아님).
# 대신 데이터셋 기록은 _dataset_writer_process()가 별도 프로세스에서 전담하고,
# 이 메인 프로세스는 큐로 프레임/명령만 넘긴다 -- 이 프로세스는 LeRobotDataset을
# 영영 import하지 않는다.

# so101_web/pipeline/1_data_collection/run_teleop_real_to_real.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
from camera_config import WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH  # noqa: E402

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
GRIPPER_JOINT = "gripper"

WRIST_CAMERA = "wrist_cam"
IMAGE_HEIGHT = WRIST_IMAGE_HEIGHT
IMAGE_WIDTH = WRIST_IMAGE_WIDTH

CONTROL_HZ = 50
RECORD_FPS = 30
DEFAULT_DATASET_ROOT = str(PROJECT_ROOT / "data" / "so101_real_to_real")
DEFAULT_REPO_ID = "local/so101_real_to_real"
DEFAULT_TASK = "teleoperate SO101 real follower arm from real leader arm"

DEFAULT_LEADER_ID = "full_arm_calibration_leader"
DEFAULT_FOLLOWER_ID = "full_arm_calibration_follower"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "config"
DEFAULT_MAX_RELATIVE_TARGET = 15.0
DEFAULT_CAMERA_INDEX = "0"
DEFAULT_STARTUP_RAMP_SECONDS = 3.0
DEFAULT_STARTUP_RAMP_MIN_TARGET = 2.0

# 스톨(명령을 받고도 실제로 안 움직이는 상태) 감지 파라미터. wrist_roll 서보가
# 실측 기계적 한계(배선 감김 등)를 넘는 목표를 계속 명령받아 막힌 채로 계속
# 힘을 쓰다가 과열/손상된 사고(대화 기록 참고) 재발 방지용 -- 특정 관절에
# 국한하지 않고 6개 관절 전부에 일반적으로 적용한다(다른 원인의 기계적
# 걸림도 같은 방식으로 보호됨).
STALL_MIN_COMMANDED_DELTA = 2.0  # 이 값(도/%) 이상 움직이라고 명령했는데
STALL_MAX_ACTUAL_DELTA = 0.5  # 실제로는 이 값(도/%) 미만만 움직였다면 "안 움직임"으로 카운트
STALL_TICKS_THRESHOLD = 10  # 녹화 주기(30Hz) 기준 연속 10틱(~0.3초) 지속되면 스톨로 판정


def build_dataset_features(include_camera: bool = True) -> dict:
    features = {
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
    }
    if include_camera:
        features[f"observation.images.{WRIST_CAMERA}"] = {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def load_or_create_dataset(
    repo_id: str, root: str, include_camera: bool = True, confirm_fn=input
) -> "LeRobotDataset":
    """run_teleop_real.py의 load_or_create_dataset()과 동일한 로직(디스크 상태
    확인/손상 감지/사용자 확인 없이는 자동 삭제하지 않음) -- 데이터셋 관리 자체는
    MuJoCo와 무관한 로직이라 그대로 재사용한다.

    이 함수는 _dataset_writer_process() 안에서만 호출된다(메인 프로세스는
    LeRobotDataset을 import하지 않으므로) -- import를 함수 안으로 늦춘 것은
    바로 그 때문이다.

    confirm_fn: 손상 감지 시 사용자 확인을 받는 함수(prompt: str) -> str.
    기본값은 내장 input()이지만, _dataset_writer_process()는 이 함수를
    자식 프로세스 자신이 아니라 메인 프로세스가 실제로 콘솔 입력을 받도록
    큐 기반 콜백으로 넘겨준다 -- Windows multiprocessing 자식 프로세스에서
    직접 input()을 호출하면 콘솔 stdin이 제대로 연결되지 않아 사용자가 답을
    입력하기도 전에 빈 문자열을 즉시 받아버리는 문제가 실측 확인됨(사용자가
    'y'를 누르기도 전에 취소 경로로 빠짐).
    """
    import json
    import os
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info_path = os.path.join(root, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            dataset = LeRobotDataset(repo_id, root=root)
            print(
                f"[run_teleop_real_to_real] 기존 데이터셋을 이어서 기록합니다: {root} "
                f"(현재 {dataset.meta.total_episodes} episodes)"
            )
            return dataset
        except Exception as exc:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            total_episodes = info.get("total_episodes", 0)
            if total_episodes == 0:
                backup = root.rstrip("/\\") + "_corrupted"
                print(
                    f"[run_teleop_real_to_real] 경고: {root} 이 손상된 상태로 감지됨 "
                    f"(meta/info.json total_episodes=0, 로딩 실패 사유: {exc}). "
                    f"진행하면 {root} -> {backup} 로 이동한 뒤 새 데이터셋을 생성합니다."
                )
                try:
                    answer = confirm_fn(
                        f"[run_teleop_real_to_real] {root} 을 백업하고 새로 생성할까요? 진행하려면 y, "
                        "그 외 입력(또는 Enter)은 취소하고 종료합니다 [y/N]: "
                    )
                except EOFError:
                    answer = ""
                if answer.strip().lower() != "y":
                    print(f"[run_teleop_real_to_real] 사용자 확인 없음 -- {root} 을 건드리지 않고 종료합니다.")
                    raise SystemExit(1)
                if os.path.exists(backup):
                    shutil.rmtree(backup)
                shutil.move(root, backup)
                print(f"[run_teleop_real_to_real] 실행됨: {root} -> {backup} 백업 후 새 데이터셋 생성 진행")
            else:
                raise RuntimeError(
                    f"{root} 데이터셋 로딩에 실패했지만 이미 {total_episodes}개의 episode가 "
                    "기록되어 있어 자동으로 삭제/백업하지 않습니다. 폴더 상태를 직접 확인한 뒤 "
                    "복구하거나 --root로 다른 경로를 지정해 다시 실행하세요."
                ) from exc

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=RECORD_FPS,
        features=build_dataset_features(include_camera=include_camera),
        root=root,
        robot_type="so101_follower",
        use_videos=False,
    )
    print(f"[run_teleop_real_to_real] 새 데이터셋을 생성합니다: {root}")
    return dataset


def _dataset_writer_process(
    cmd_queue: "mp.Queue", ack_queue: "mp.Queue", repo_id: str, root: str, include_camera: bool
) -> None:
    """리더/팔로워 제어 루프와 완전히 분리된 별도 프로세스에서 LeRobotDataset
    기록만 전담한다(위 top-level import 설명 참고 -- torch/pyarrow/datasets는
    이 함수 안, 즉 이 프로세스 안에서만 로드된다).

    cmd_queue로 받는 명령: {"type": "frame", "data": {...}} / "start_episode" /
    "save_episode" / "discard_episode" / "shutdown". 프레임과 명령이 같은
    단일 FIFO 큐를 통해 메인 프로세스의 같은 스레드에서 순서대로 들어오므로,
    "이 episode의 프레임들 다음에 저장/폐기 명령"이라는 인과 순서가 큐 자체의
    FIFO 성질로 자동 보장된다(별도 동기화 불필요).

    ack_queue로 보내는 응답: {"type": "ready"/"aborted"/"episode_saved"/
    "episode_empty"/"episode_discarded"/"error"/"finalized", ...}. 메인
    프로세스는 이 응답을 논블로킹으로 드레인해서 사용자에게 진행 상황을
    출력한다(디스크 I/O가 실제로 얼마나 걸리든 제어 루프는 절대 안 기다림).
    """
    def _confirm_via_main_process(prompt: str) -> str:
        """Windows multiprocessing 자식 프로세스는 input()이 콘솔에 제대로
        연결되지 않을 수 있어(위 load_or_create_dataset() docstring 참고),
        직접 묻지 않고 메인 프로세스에 질문을 전달한 뒤 답을 기다린다. 이
        시점엔 아직 메인 루프(cmd_queue.get() 반복)가 시작되지 않았으므로
        여기서 블로킹 get()을 해도 다른 명령과 섞이지 않는다.
        """
        ack_queue.put({"type": "confirm", "prompt": prompt})
        reply = cmd_queue.get()
        return reply.get("answer", "")

    try:
        dataset = load_or_create_dataset(
            repo_id, root, include_camera=include_camera, confirm_fn=_confirm_via_main_process
        )
    except SystemExit:
        ack_queue.put({"type": "aborted", "reason": "사용자가 손상된 데이터셋 복구를 취소함"})
        return
    except Exception as exc:  # noqa: BLE001 - 메인 프로세스에 반드시 보고해야 함
        ack_queue.put({"type": "aborted", "reason": str(exc)})
        return

    ack_queue.put({"type": "ready", "total_episodes": dataset.meta.total_episodes})

    while True:
        cmd = cmd_queue.get()
        kind = cmd.get("type")
        try:
            if kind == "frame":
                dataset.add_frame(cmd["data"])
            elif kind == "start_episode":
                pass  # episode_buffer는 add_frame()이 첫 프레임에서 알아서 만듦
            elif kind == "save_episode":
                if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                    frame_count = dataset.episode_buffer["size"]
                    dataset.save_episode()
                    ack_queue.put(
                        {
                            "type": "episode_saved",
                            "frame_count": frame_count,
                            "total_episodes": dataset.meta.total_episodes,
                        }
                    )
                else:
                    dataset.episode_buffer = None
                    ack_queue.put({"type": "episode_empty"})
            elif kind == "discard_episode":
                dataset.episode_buffer = None
                ack_queue.put({"type": "episode_discarded"})
            elif kind == "shutdown":
                break
        except Exception as exc:  # noqa: BLE001 - 기록 프로세스가 죽지 않고 계속 받도록
            ack_queue.put({"type": "error", "message": f"{kind}: {exc}"})

    try:
        dataset.finalize()
        total_episodes = dataset.meta.total_episodes
    except Exception as exc:  # noqa: BLE001 - 실패해도 메인 프로세스를 무한 대기시키면 안 됨
        ack_queue.put({"type": "error", "message": f"finalize: {exc}"})
        try:
            total_episodes = dataset.meta.total_episodes
        except Exception:
            total_episodes = -1  # meta 자체도 못 읽을 정도로 심각한 손상 -- 알 수 없음 표시
    ack_queue.put({"type": "finalized", "total_episodes": total_episodes})


class RecorderControl:
    """키 입력(CameraPreviewWindow)과 메인 루프 사이에서 주고받는 요청 플래그.
    run_teleop_real.py의 동명 클래스와 동일 -- MuJoCo key_callback이든 tkinter
    키 바인딩이든 이 클래스 자체는 입력 소스와 무관하다.
    """

    def __init__(self):
        self.toggle_requested = False
        self.discard_requested = False


class ConsoleKeyListener:
    """--no-camera 모드 전용 S/X 키 입력 감지. tkinter GUI 창을 아예 띄우지
    않는다.

    실측 확인(대화 기록 참고): tkinter의 update_idletasks()/update() 이벤트
    펌프를 리더+팔로워 동시 폴링 + 실제 서보 이동과 같이 돌리면 시리얼 통신
    실패율이 급증함(격리 테스트에서 80% 실패 재현, GUI 없이 같은 조건은
    100% 성공) -- LeRobotDataset 프로세스 분리로는 전혀 해결 안 됐고 원인은
    GUI 이벤트 펌프 자체였다. 카메라가 없어 이미지를 보여줄 필요가 없는
    --no-camera 모드에서는 GUI 창 자체가 불필요하므로, 아예 없애고 콘솔
    자체의 키 입력을 msvcrt로 논블로킹 폴링하는 방식으로 대체해 이 문제를
    원천 차단한다(Windows 전용 -- 이 프로젝트가 이미 Windows 전용으로
    개발됨, run_teleop_real.py/run_teleop_follower.py도 ctypes.windll 등
    Windows 전용 API를 씀).

    카메라를 실제로 쓰는 경로(CameraPreviewWindow, 이미지를 보여줘야 하므로
    GUI 자체를 없앨 수 없음)에도 같은 종류의 간섭이 있을 가능성이 있으나,
    이번 조사 범위는 --no-camera 모드까지다 -- 카메라 경로는 별도 확인
    필요(아래 CameraPreviewWindow의 update_image()가 여전히 이 위험을
    안고 있음).
    """

    def __init__(self, control: "RecorderControl"):
        import msvcrt

        self._msvcrt = msvcrt
        self._control = control
        self._closed = False

    def pump(self) -> None:
        while self._msvcrt.kbhit():
            ch = self._msvcrt.getch()
            try:
                key = ch.decode("utf-8", errors="ignore").lower()
            except Exception:
                continue
            if key == "s":
                self._control.toggle_requested = True
            elif key == "x":
                self._control.discard_requested = True

    @property
    def closed(self) -> bool:
        return self._closed  # 콘솔 자체를 "닫는" 개념이 없어 항상 False -- Ctrl+C로만 종료

    def close(self) -> None:
        self._closed = True


class CameraPreviewWindow:
    """손목 카메라(wrist_cam) 실시간 미리보기 창.

    run_teleop_real.py의 동명 클래스와 거의 동일(tkinter + Pillow, 논블로킹
    update_idletasks()/update() 폴링)하지만, 이 스크립트에는 MuJoCo 뷰어가
    없으므로 S/X 녹화 제어 키 입력도 이 창에서 받는다(on_key 콜백) -- 창이
    OS 포커스를 가지고 있어야 키 입력이 들어온다는 점에 주의.
    """

    def __init__(self, width: int, height: int, title: str = "wrist_cam", on_key=None):
        import tkinter as tk

        self._closed = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo = None  # PhotoImage 참조를 들고 있지 않으면 GC로 사라짐

        self._on_key = on_key
        if on_key is not None:
            self.root.bind("<KeyPress>", self._on_keypress)
            self.root.focus_set()

    def _on_keypress(self, event) -> None:
        # tkinter 이벤트 콜백에서 예외가 새어나가면 tkinter 자체 핸들러가 콘솔에
        # traceback을 찍긴 하지만 앱이 죽지는 않는다 -- 그래도 이 프로젝트의
        # 로그 태그 규칙([run_teleop_real_to_real])을 지키기 위해 직접 잡는다.
        try:
            if self._on_key is not None:
                self._on_key(event.keysym.lower())
        except Exception:
            print("[run_teleop_real_to_real] 키 입력 처리 중 오류(무시):", flush=True)
            traceback.print_exc()

    def _on_close(self):
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def update_image(self, rgb_image) -> None:
        if self._closed:
            return
        from PIL import Image, ImageTk

        pil_image = Image.fromarray(rgb_image)
        self._photo = ImageTk.PhotoImage(pil_image)
        self.label.configure(image=self._photo)
        self.root.update_idletasks()
        self.root.update()

    def show_message(self, text: str) -> None:
        """카메라가 없는 --no-camera 모드용. 이미지 대신 안내 문구를 표시한다
        (S/X 키 입력을 계속 받으려면 이 창 자체는 여전히 필요하므로 창을 없애지
        않고 내용만 텍스트로 바꾼다).
        """
        if self._closed:
            return
        self.label.configure(image="", text=text, font=("Consolas", 12), justify="left")
        self.root.update_idletasks()
        self.root.update()

    def pump(self) -> None:
        """이미지 갱신 없이 tkinter 이벤트 루프만 한 번 돌린다(--no-camera 모드에서
        update_image() 대신 호출해 창이 멈추지 않고 키 입력을 계속 받게 한다).
        """
        if self._closed:
            return
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        if not self._closed:
            self._on_close()


def _handle_preview_key(keysym: str, control: RecorderControl) -> None:
    if keysym == "s":
        control.toggle_requested = True
    elif keysym == "x":
        control.discard_requested = True


def prepare_port(port: str, retries: int = 3, retry_delay: float = 1.0) -> None:
    """연결 전에 포트를 열고 바로 닫아, 이전 실행이 비정상 종료되면서 남아있을
    수 있는 스테일 상태를 정리하고 가용성을 미리 확인한다(run_teleop_real.py와
    동일). 실패해도 예외를 던지지 않고 경고만 출력한다.
    """
    import serial

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            probe = serial.Serial(port)
            probe.reset_input_buffer()
            probe.reset_output_buffer()
            probe.close()
            if attempt > 1:
                print(f"[run_teleop_real_to_real] 포트 '{port}' 확보 성공 (시도 {attempt}/{retries})")
            return
        except (serial.SerialException, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)

    print(
        f"[run_teleop_real_to_real] 경고: 포트 '{port}'를 열지 못했습니다 ({last_error}).\n"
        "  이전에 비정상 종료된 python.exe가 아직 포트를 점유 중일 수 있습니다. PowerShell에서:\n"
        "    Get-Process python* | Select-Object Id,ProcessName,Path\n"
        "    Stop-Process -Id <PID> -Force\n"
        "  로 확인/종료한 뒤 다시 실행해 보세요. (일단 연결은 계속 시도합니다)"
    )


def safe_disconnect(device, label: str) -> None:
    """어떤 상황에서도 예외를 밖으로 전파하지 않고 연결 해제를 시도한다. 리더/
    팔로워 둘 다 같은 형태의 인터페이스(is_connected/disconnect())를 가지므로
    하나의 함수로 처리한다(같은 파일 안에서 두 번 호출되므로, 별도 스크립트간
    독립성을 위한 중복과 달리 이건 그냥 중복임).
    """
    try:
        if device.is_connected:
            device.disconnect()
            print(f"[run_teleop_real_to_real] {label} 연결 해제 완료.")
    except Exception as exc:  # noqa: BLE001 - 종료 경로이므로 모든 예외를 삼킨다
        print(f"[run_teleop_real_to_real] {label} 연결 해제 중 오류(무시): {exc}")


def compute_motor_degree_range(calibration, model_resolution: int) -> tuple[float, float]:
    """캘리브레이션의 range_min/range_max가 실제로 매핑되는 도(degree) 양 끝값을
    계산한다. lerobot FeetechMotorsBus._normalize()의 DEGREES 공식과 동일
    (run_teleop_real.py의 compute_leader_degree_range(), run_teleop_follower.py의
    동명 함수와 같은 공식 -- 리더/팔로워 어느 쪽 캘리브레이션에도 적용되는 일반
    공식이라 이 스크립트에도 독립적으로 둔다).
    """
    mid = (calibration.range_min + calibration.range_max) / 2
    max_res = model_resolution - 1
    lo = (calibration.range_min - mid) * 360 / max_res
    hi = (calibration.range_max - mid) * 360 / max_res
    return lo, hi


def _linear_map(value: float, src_lo: float, src_hi: float, dst_lo: float, dst_hi: float) -> float:
    frac = (value - src_lo) / (src_hi - src_lo)
    return dst_lo + frac * (dst_hi - dst_lo)


def leader_action_to_follower_action(
    action: dict,
    leader_deg_ranges: dict[str, tuple[float, float]],
    follower_deg_ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """리더의 get_action() 결과를 팔로워의 send_action() 형식으로 변환한다.

    5개 관절(그리퍼 제외)은 리더의 실측 캘리브레이션 도(degree) range를 팔로워의
    실측 캘리브레이션 도 range로 비례 스케일링한다(run_teleop_real.py의
    leader_action_to_radians()와 같은 방식 -- 서로 다르게 캘리브레이션된 두 실물을
    짝짓는 것뿐, 목적지가 MuJoCo 라디안이 아니라 팔로워 실물 도 단위라는 점만
    다르다).

    그리퍼는 리더/팔로워 둘 다 "자기 자신의 완전 닫힘~완전 열림"을 0~100%로
    정규화한 값이라(FeetechMotorsBus의 RANGE_0_100 모드) 그대로 전달하면 된다 --
    range_min/range_max로 다시 매핑할 "출발 range" 자체가 없는, 이미 정규화된
    값이기 때문(run_teleop_real.py가 리더 그리퍼 0~100을 팔로워 목적지 range로만
    매핑하고 리더 쪽 range는 아예 조회하지 않는 것과 같은 이유). 값 자체는 방어적으로
    [0, 100]에 클램프한다.
    """
    follower_action: dict[str, float] = {}
    for name in JOINT_NAMES:
        value = action[f"{name}.pos"]
        if name == GRIPPER_JOINT:
            follower_action[f"{name}.pos"] = float(np.clip(value, 0.0, 100.0))
        else:
            leader_lo, leader_hi = leader_deg_ranges[name]
            follower_lo, follower_hi = follower_deg_ranges[name]
            mapped = _linear_map(value, leader_lo, leader_hi, follower_lo, follower_hi)
            follower_action[f"{name}.pos"] = float(
                np.clip(mapped, min(follower_lo, follower_hi), max(follower_lo, follower_hi))
            )
    return follower_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True, help="실물 리더암 시리얼 포트 (예: COM5)")
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID, help="리더암 캘리브레이션 식별자")
    parser.add_argument(
        "--leader-calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="리더암 캘리브레이션 JSON(<id>.json)이 있는 디렉터리. 기본값은 config/.",
    )
    parser.add_argument("--follower-port", required=True, help="실물 팔로워암 시리얼 포트 (예: COM7)")
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID, help="팔로워암 캘리브레이션 식별자")
    parser.add_argument(
        "--follower-calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="팔로워암 캘리브레이션 JSON(<id>.json)이 있는 디렉터리. 기본값은 config/.",
    )
    parser.add_argument(
        "--camera-index",
        default=DEFAULT_CAMERA_INDEX,
        help=(
            "손목 카메라의 OpenCV 장치 인덱스 또는 경로. 실제 값은 "
            "`lerobot-find-cameras opencv`로 먼저 확인할 것 (기본값 0)."
        ),
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help=(
            "임시 경로: 손목 카메라를 아예 연결하지 않고 리더->팔로워 텔레오퍼레이션만 "
            "테스트한다(카메라 하드웨어가 아직 없을 때 사용). 녹화(S/X)는 계속 가능하지만 "
            "이미지 없이 action/observation.state만 기록되므로, --repo-id/--root를 "
            "지정하지 않으면 카메라 포함 데이터셋과 섞이지 않도록 자동으로 "
            "'_nocam' 접미사가 붙은 별도 경로를 쓴다."
        ),
    )
    parser.add_argument("--hz", type=float, default=CONTROL_HZ, help="제어 루프 주파수(Hz)")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=DEFAULT_MAX_RELATIVE_TARGET,
        help=(
            "한 번의 전송에서 허용하는 최대 목표 이동량(5개 관절은 도, gripper는 "
            "%%). 0 이하로 주면 비활성화(권장하지 않음)."
        ),
    )
    parser.add_argument(
        "--startup-ramp-seconds",
        type=float,
        default=DEFAULT_STARTUP_RAMP_SECONDS,
        help=(
            "세션 시작 직후 몇 초에 걸쳐 max_relative_target을 --startup-ramp-min-target에서 "
            "--max-relative-target까지 선형으로 늘릴지(초). 리더/팔로워가 서로 다른 자세로 "
            "시작하면(특히 wrist_roll처럼 자연스러운 '중앙'이 없는 연속회전 관절) 첫 틱부터 "
            "여러 관절을 동시에 최대 속도로 '따라잡으려' 하면서 순간 전류가 튀어 시리얼 통신이 "
            "불안정해지는 문제가 실측 확인됨 -- 이를 완화하기 위한 완만한 시작(soft start). "
            "0 이하로 주면 램프 없이 처음부터 --max-relative-target을 그대로 씀."
        ),
    )
    parser.add_argument(
        "--startup-ramp-min-target",
        type=float,
        default=DEFAULT_STARTUP_RAMP_MIN_TARGET,
        help="램프 시작 시점의 max_relative_target 값(도/%%, --startup-ramp-seconds 참고).",
    )
    parser.add_argument(
        "--wrist-roll-safe-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_DEG", "MAX_DEG"),
        help=(
            "wrist_roll에 대해 캘리브레이션과 별개로 추가로 강제할 실측 안전 회전 범위(도, "
            "follower 기준). wrist_roll은 소프트웨어상 연속회전 관절(캘리브레이션 range가 "
            "항상 0~4095 풀레인지)로 등록돼 있지만, 실제로는 관절 내부 배선 다발 때문에 "
            "물리적으로 무한 회전이 불가능한 경우가 있다(실측 확인 -- 배선이 감기는 지점에서 "
            "막힌 채 계속 힘을 쓰다 서보가 손상된 사고 있었음). 이 값을 지정하면 손으로 "
            "직접 두 방향의 기계적 한계 지점을 찾아 follower.get_observation()으로 측정한 "
            "각도값을(여유를 두고 안쪽으로) 넣어서, 리더가 어떤 값을 보내든 이 범위 밖으로는 "
            "절대 명령이 안 나가게 한다. 지정하지 않으면(기본값) 이 추가 클램프 없이 기존 "
            "캘리브레이션 range를 그대로 쓴다."
        ),
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="episode에 붙일 작업 설명(자연어)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot 데이터셋 repo_id")
    parser.add_argument("--root", default=DEFAULT_DATASET_ROOT, help="데이터셋 저장 경로")
    args = parser.parse_args()

    if args.no_camera:
        if args.repo_id == DEFAULT_REPO_ID:
            args.repo_id = DEFAULT_REPO_ID + "_nocam"
        if args.root == DEFAULT_DATASET_ROOT:
            args.root = DEFAULT_DATASET_ROOT + "_nocam"

    # 외부(웹 대시보드 등)에서 이 프로세스를 정상 종료시키기 위한 신호 처리
    # (run_teleop_real.py의 동일 패턴, CLAUDE.md 30-2절 참고). 아직 이 스크립트를
    # 대시보드에 연결하지는 않았지만, process_manager.py의 graceful stop은
    # 특정 스크립트를 가리지 않고 동작하는 일반 메커니즘이라 지금 붙여둬도 손해가
    # 없고, 나중에 잊어버려 하드 kill로 dataset.finalize()가 스킵되는(데이터 유실)
    # 사고를 미리 막을 수 있다.
    shutdown_requested = {"flag": False}

    def _handle_termination_signal(signum, frame):
        print(
            f"[run_teleop_real_to_real] 종료 신호({signum}) 수신 -- 안전 종료 절차 시작"
            "(연결 해제/데이터셋 finalize)",
            flush=True,
        )
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_termination_signal)

    max_relative_target = args.max_relative_target if args.max_relative_target > 0 else None
    wrist_roll_safe_range = tuple(sorted(args.wrist_roll_safe_range)) if args.wrist_roll_safe_range else None

    if args.no_camera:
        cameras = {}
    else:
        camera_index_arg = args.camera_index
        try:
            camera_index: int | str = int(camera_index_arg)
        except ValueError:
            camera_index = camera_index_arg

        cameras = {
            WRIST_CAMERA: OpenCVCameraConfig(
                index_or_path=camera_index,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                fps=RECORD_FPS,
            )
        }

    prepare_port(args.leader_port)
    prepare_port(args.follower_port)

    leader = SO101Leader(
        SOLeaderTeleopConfig(
            port=args.leader_port, id=args.leader_id, calibration_dir=args.leader_calibration_dir
        )
    )
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            calibration_dir=args.follower_calibration_dir,
            max_relative_target=max_relative_target,
            use_degrees=True,
            cameras=cameras,
        )
    )

    leader_cal_fpath = args.leader_calibration_dir / f"{args.leader_id}.json"
    follower_cal_fpath = args.follower_calibration_dir / f"{args.follower_id}.json"
    print(
        f"[run_teleop_real_to_real] 리더암 캘리브레이션 파일: {leader_cal_fpath}"
        f" ({'존재함' if leader_cal_fpath.is_file() else '없음 -- 최초 연결 시 새로 캘리브레이션 진행됨'})"
    )
    print(
        f"[run_teleop_real_to_real] 팔로워암 캘리브레이션 파일: {follower_cal_fpath}"
        f" ({'존재함' if follower_cal_fpath.is_file() else '없음 -- 최초 연결 시 새로 캘리브레이션 진행됨'})"
    )

    # preview는 leader/follower.connect() 이전에 만들어 둔다 -- 아래 최종
    # finally에서 항상 참조되므로, connect() 실패로 그 finally에 도달했을 때도
    # NameError 없이 안전하게 None 체크만 하면 되도록 하기 위함(run_teleop_real.py도
    # preview를 leader 연결 이전에 만드는 것과 같은 이유).
    control = RecorderControl()
    if args.no_camera:
        # tkinter GUI 창 자체를 아예 안 띄운다 -- ConsoleKeyListener docstring
        # 참고(GUI 이벤트 펌프가 실제 서보 이동과 겹치면 시리얼 통신이 크게
        # 불안정해짐이 실측 확인됨). 카메라가 없어 이미지를 보여줄 필요가
        # 없는 이 모드에서는 대체 불가능한 이유가 없으므로 콘솔 키 입력으로
        # 대체한다.
        preview = ConsoleKeyListener(control)
    else:
        try:
            preview = CameraPreviewWindow(
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                title=WRIST_CAMERA,
                on_key=lambda keysym: _handle_preview_key(keysym, control),
            )
        except Exception:
            print(
                "[run_teleop_real_to_real] 카메라 미리보기 창 생성 실패 -- 이번 세션에서는 "
                "S/X로 녹화를 제어할 수 없습니다:",
                flush=True,
            )
            traceback.print_exc()
            preview = None

    try:
        leader.connect()
        follower.connect()

        # gripper를 제외한 5개 관절: 리더/팔로워 각자의 실측 캘리브레이션 range를
        # 도(degree) 단위로 변환해(leader_deg_ranges/follower_deg_ranges) 서로
        # 비례 스케일링에 쓴다.
        degree_joint_names = [n for n in JOINT_NAMES if n != GRIPPER_JOINT]
        leader_deg_ranges: dict[str, tuple[float, float]] = {}
        follower_deg_ranges: dict[str, tuple[float, float]] = {}
        for name in degree_joint_names:
            l_cal = leader.bus.calibration[name]
            l_model = leader.bus.motors[name].model
            l_res = leader.bus.model_resolution_table[l_model]
            leader_deg_ranges[name] = compute_motor_degree_range(l_cal, l_res)

            f_cal = follower.bus.calibration[name]
            f_model = follower.bus.motors[name].model
            f_res = follower.bus.model_resolution_table[f_model]
            follower_deg_ranges[name] = compute_motor_degree_range(f_cal, f_res)

        print("[run_teleop_real_to_real] 리더 range -> 팔로워 range 스케일링:")
        for name in degree_joint_names:
            l_lo, l_hi = leader_deg_ranges[name]
            f_lo, f_hi = follower_deg_ranges[name]
            print(
                f"  {name:15s} leader=[{l_lo:+.2f}, {l_hi:+.2f}]deg -> "
                f"follower=[{f_lo:+.2f}, {f_hi:+.2f}]deg"
            )

        # 데이터셋 기록은 별도 프로세스(_dataset_writer_process)가 전담한다(위
        # top-level import 설명 참고) -- 이 프로세스는 큐로 프레임/명령만 넘기고
        # 응답을 논블로킹으로 받는다. cmd_queue는 프레임 데이터와 제어 명령이
        # 섞여 들어가는 단일 FIFO라 순서가 자동 보장된다.
        cmd_queue: mp.Queue = mp.Queue()
        ack_queue: mp.Queue = mp.Queue()
        writer_proc = mp.Process(
            target=_dataset_writer_process,
            args=(cmd_queue, ack_queue, args.repo_id, args.root, not args.no_camera),
            daemon=False,
        )
        writer_proc.start()

        print("[run_teleop_real_to_real] 데이터셋 기록 프로세스 시작 -- 준비될 때까지 대기 중...")
        known_total_episodes = 0
        while True:
            ack = ack_queue.get()  # load_or_create_dataset()이 끝날 때까지 블로킹 대기
            ack_type = ack.get("type")
            if ack_type == "confirm":
                # Process B가 손상 감지 시 직접 input()을 못 부르는 이유는
                # load_or_create_dataset()의 confirm_fn docstring 참고 --
                # 여기, 메인 프로세스에서 실제로 콘솔 입력을 받는다.
                print(ack["prompt"], end="", flush=True)
                try:
                    answer = input()
                except EOFError:
                    answer = ""
                cmd_queue.put({"type": "confirm_answer", "answer": answer})
                continue
            elif ack_type == "ready":
                known_total_episodes = ack["total_episodes"]
                break
            else:
                writer_proc.join(timeout=5)
                raise SystemExit(
                    f"[run_teleop_real_to_real] 데이터셋 초기화 실패: {ack.get('reason')}"
                )
        print(
            f"[run_teleop_real_to_real] 데이터셋 준비 완료 (현재 {known_total_episodes} episodes): {args.root}"
        )

        def _drain_acks() -> None:
            """ack_queue에 쌓인 응답을 논블로킹으로 전부 소비해 출력한다.
            제어 루프를 절대 블로킹하지 않는다(디스크 I/O가 오래 걸려도 다음
            ack가 올 때까지 그냥 안 찍힐 뿐, 루프 타이밍에는 영향 없음).
            """
            nonlocal known_total_episodes
            while True:
                try:
                    ack = ack_queue.get_nowait()
                except queue.Empty:
                    return
                ack_type = ack.get("type")
                if ack_type == "episode_saved":
                    known_total_episodes = ack["total_episodes"]
                    print(
                        f"[REC] 저장 완료: {ack['frame_count']} frames "
                        f"(총 {known_total_episodes} episodes)"
                    )
                    print("[REC] 다음 episode를 위해 물체를 손으로 재배치하세요.")
                elif ack_type == "episode_empty":
                    print("[REC] 빈 episode라 저장하지 않음")
                elif ack_type == "episode_discarded":
                    pass  # discard 요청 시점에 이미 "[REC] episode 버림"을 출력함
                elif ack_type == "error":
                    print(f"[run_teleop_real_to_real] 데이터셋 기록 프로세스 오류(계속 진행): {ack['message']}")
                elif ack_type == "finalized":
                    known_total_episodes = ack["total_episodes"]

        print("=" * 60)
        print("SO101 실물 리더암 -> 실물 팔로워암 텔레오퍼레이션 + 녹화")
        print(f"리더암 포트: {args.leader_port}  (id={args.leader_id})")
        print(f"팔로워암 포트: {args.follower_port}  (id={args.follower_id})")
        print(f"제어 주기: {args.hz} Hz   녹화 주기: {RECORD_FPS} Hz   max_relative_target={max_relative_target}")
        if max_relative_target is not None and args.startup_ramp_seconds > 0:
            print(
                f"시작 완만화(ramp): max_relative_target을 {args.startup_ramp_min_target} -> "
                f"{max_relative_target}까지 {args.startup_ramp_seconds}초에 걸쳐 선형 증가"
                "(리더/팔로워 초기 자세 차이로 인한 순간 고전류 완화)"
            )
        if wrist_roll_safe_range is not None:
            print(
                f"wrist_roll 안전 범위 강제: [{wrist_roll_safe_range[0]:+.2f}, "
                f"{wrist_roll_safe_range[1]:+.2f}]deg (이 범위 밖으로는 목표를 절대 보내지 않음)"
            )
        else:
            print(
                "경고: wrist_roll 안전 범위(--wrist-roll-safe-range)가 설정되지 않음 -- "
                "캘리브레이션 range를 그대로 신뢰함. 이 관절이 실제로는 물리적 한계가 있는 "
                "것으로 이미 확인됐으니(대화 기록 참고), 측정값이 있다면 반드시 지정할 것."
            )
        print(
            f"스톨(움직임 없음) 감지: 활성화 -- 명령을 받고도 {STALL_TICKS_THRESHOLD}틱 연속 "
            f"실제로 안 움직이는 관절은 자동으로 토크를 끕니다."
        )
        if args.no_camera:
            print(f"--no-camera 모드: 카메라 없이 텔레오퍼레이션만 검증 (데이터셋: repo_id={args.repo_id}, root={args.root})")
        if preview is not None:
            print(f"S 키: episode 녹화 시작/종료   X 키: 녹화 중인 episode 버리기 (task='{args.task}')")
            if args.no_camera:
                print("(GUI 창 없음 -- 이 콘솔 창에 포커스가 있어야 키 입력이 인식됩니다. 종료는 Ctrl+C)")
            else:
                print("(미리보기 창을 한 번 클릭해 포커스를 준 뒤에 눌러야 키 입력이 인식됩니다)")
        else:
            print("경고: 미리보기 창이 없어 이번 세션에서는 S/X로 녹화를 제어할 수 없습니다.")
        print("=" * 60)

        recording = False
        episode_frame_count = 0
        frame_accum = 0.0
        frame_interval = 1.0 / RECORD_FPS
        loop_dt = 1.0 / args.hz
        ramp_start_time = time.time()
        ramp_done_announced = False
        prev_observation = None
        stall_counters = {name: 0 for name in JOINT_NAMES}
        stalled_joints: set[str] = set()

        try:
            while not shutdown_requested["flag"] and (preview is None or not preview.closed):
                step_start = time.time()

                try:
                    if control.toggle_requested:
                        control.toggle_requested = False
                        if recording:
                            if episode_frame_count > 0:
                                print(
                                    f"[REC] 저장 요청 전송... ({episode_frame_count} frames, "
                                    "완료되면 별도로 '저장 완료' 메시지가 뜹니다)"
                                )
                                cmd_queue.put({"type": "save_episode"})
                            else:
                                cmd_queue.put({"type": "discard_episode"})
                                print("[REC] 빈 episode라 저장하지 않음")
                            recording = False
                        else:
                            recording = True
                            episode_frame_count = 0
                            frame_accum = 0.0
                            cmd_queue.put({"type": "start_episode"})
                            print(f"[REC] 녹화 시작 (episode {known_total_episodes})")

                    if control.discard_requested:
                        control.discard_requested = False
                        if recording:
                            cmd_queue.put({"type": "discard_episode"})
                            recording = False
                            episode_frame_count = 0
                            print("[REC] episode 버림")

                    _drain_acks()

                    leader_action = leader.get_action()
                    follower_action = leader_action_to_follower_action(
                        leader_action, leader_deg_ranges, follower_deg_ranges
                    )

                    if wrist_roll_safe_range is not None:
                        safe_lo, safe_hi = wrist_roll_safe_range
                        follower_action["wrist_roll.pos"] = float(
                            np.clip(follower_action["wrist_roll.pos"], safe_lo, safe_hi)
                        )

                    # 스톨로 판정된 관절은 토크가 이미 꺼져 있지만(아래 감지 로직 참고),
                    # 방어적으로 목표도 마지막으로 확인된 실제 위치에 고정한다(진짜
                    # no-op) -- 세션을 재시작하기 전까지는 계속 그 자리를 유지한다.
                    for stalled_name in stalled_joints:
                        if prev_observation is not None:
                            follower_action[f"{stalled_name}.pos"] = prev_observation[f"{stalled_name}.pos"]

                    # 시작 완만화(ramp): 세션 초반 max_relative_target을 작게 시작해
                    # 서서히 늘린다. follower.config.max_relative_target은
                    # send_action()이 호출될 때마다 매번 새로 읽으므로(so_follower.py
                    # 확인됨), lerobot 소스를 건드리지 않고 이 값을 여기서 갱신하는
                    # 것만으로 안전하게 조절된다.
                    if max_relative_target is not None and args.startup_ramp_seconds > 0:
                        ramp_elapsed = time.time() - ramp_start_time
                        if ramp_elapsed < args.startup_ramp_seconds:
                            ramp_fraction = ramp_elapsed / args.startup_ramp_seconds
                            follower.config.max_relative_target = (
                                args.startup_ramp_min_target
                                + (max_relative_target - args.startup_ramp_min_target) * ramp_fraction
                            )
                        elif not ramp_done_announced:
                            follower.config.max_relative_target = max_relative_target
                            ramp_done_announced = True
                            print(
                                f"[run_teleop_real_to_real] 시작 완만화 종료 -- "
                                f"max_relative_target={max_relative_target}로 정상 적용"
                            )

                    frame_accum += loop_dt
                    if frame_accum >= frame_interval:
                        frame_accum -= frame_interval

                        # 녹화되는 프레임의 observation.state/wrist_cam은 이번 틱에
                        # 새로 보낼 action을 적용하기 "이전"의 실제 상태여야
                        # (state_t, action_t) 쌍이 되므로, get_observation()을
                        # send_action()보다 먼저 호출한다(run_teleop_real.py가
                        # data.ctrl을 쓰기 전에 data.qpos를 읽던 순서와 동일).
                        observation = follower.get_observation()
                        applied_action = follower.send_action(follower_action)

                        # 스톨 감지: 이번 틱에 의미 있는 크기로 움직이라고 명령했는데
                        # (commanded_delta) 실제 위치는 거의 안 바뀐(actual_delta) 상태가
                        # STALL_TICKS_THRESHOLD틱 연속되면, 그 관절은 물리적으로 막혀서
                        # 계속 힘만 쓰고 있는 것으로 판단해 토크를 꺼버린다(과열/손상
                        # 재발 방지, wrist_roll 실제 사고 이후 추가된 안전장치 -- 특정
                        # 관절에 국한하지 않고 6개 관절 전부에 동일하게 적용).
                        if prev_observation is not None:
                            # gripper는 제외한다 -- 물체를 쥐면 모터가 계속 힘을 쓰면서도
                            # 실제로는 안 움직이는 게 정상적인 의도된 동작이라(스톨과
                            # 겉보기 패턴이 같음), 여기 포함하면 정상적인 파지 동작을
                            # 스톨로 오판해 토크를 꺼버리게 된다.
                            for name in degree_joint_names:
                                if name in stalled_joints:
                                    continue
                                commanded_delta = abs(
                                    applied_action[f"{name}.pos"] - prev_observation[f"{name}.pos"]
                                )
                                actual_delta = abs(
                                    observation[f"{name}.pos"] - prev_observation[f"{name}.pos"]
                                )
                                if (
                                    commanded_delta > STALL_MIN_COMMANDED_DELTA
                                    and actual_delta < STALL_MAX_ACTUAL_DELTA
                                ):
                                    stall_counters[name] += 1
                                else:
                                    stall_counters[name] = 0
                                if stall_counters[name] >= STALL_TICKS_THRESHOLD:
                                    stalled_joints.add(name)
                                    print(
                                        f"[안전] {name} 관절이 명령을 받고도 "
                                        f"{STALL_TICKS_THRESHOLD}틱 연속 실제로 움직이지 않았습니다 "
                                        "-- 스톨로 판단해 토크를 즉시 해제합니다. 이 세션에서는 "
                                        "이 관절이 더 이상 반응하지 않습니다(재시작 필요).",
                                        flush=True,
                                    )
                                    try:
                                        follower.bus.disable_torque(name)
                                    except Exception:
                                        traceback.print_exc()
                        prev_observation = observation

                        if preview is not None:
                            if args.no_camera:
                                preview.pump()
                            else:
                                preview.update_image(observation[WRIST_CAMERA])

                        if recording:
                            observation_state = np.array(
                                [observation[f"{name}.pos"] for name in JOINT_NAMES],
                                dtype=np.float32,
                            )
                            action_array = np.array(
                                [applied_action[f"{name}.pos"] for name in JOINT_NAMES],
                                dtype=np.float32,
                            )
                            frame = {
                                "observation.state": observation_state,
                                "action": action_array,
                                "task": args.task,
                            }
                            if not args.no_camera:
                                frame[f"observation.images.{WRIST_CAMERA}"] = observation[WRIST_CAMERA]
                            cmd_queue.put({"type": "frame", "data": frame})
                            episode_frame_count += 1
                    else:
                        # 녹화/카메라 주기가 아닌 틱에도 팔로워는 계속 리더를
                        # 추종해야 하므로 send_action()은 매 틱(--hz) 호출한다 --
                        # 실물 서보는 명령 주기가 낮으면 움직임이 뚝뚝 끊기므로,
                        # 물리 엔진이 없어졌다고 제어 주기까지 30Hz로 낮추지 않는다.
                        follower.send_action(follower_action)
                except Exception:
                    print(
                        "[run_teleop_real_to_real] 제어 스텝 중 오류(이번 프레임 건너뜀):",
                        flush=True,
                    )
                    traceback.print_exc()

                time_until_next_step = loop_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            if recording and episode_frame_count > 0:
                print(f"[REC] 종료 시 자동 저장 요청 전송... ({episode_frame_count} frames)")
                cmd_queue.put({"type": "save_episode"})
        finally:
            print("[run_teleop_real_to_real] 데이터셋 기록 프로세스에 종료 요청 전송")
            cmd_queue.put({"type": "shutdown"})
            # shutdown 이후 큐에 남아있던 episode_saved 등 응답을 전부 출력하고,
            # 마지막으로 반드시 오는 "finalized" 응답까지 블로킹 대기한다 --
            # 원래 코드가 dataset.finalize()를 동기 호출로 기다리던 것과 동일하게,
            # 데이터 안전을 위해 타임아웃 없이 기다린다(하드 kill이 아닌 한 반드시 옴).
            while True:
                ack = ack_queue.get()
                ack_type = ack.get("type")
                if ack_type == "episode_saved":
                    print(
                        f"[REC] 저장 완료: {ack['frame_count']} frames "
                        f"(총 {ack['total_episodes']} episodes)"
                    )
                elif ack_type == "episode_empty":
                    print("[REC] 빈 episode라 저장하지 않음")
                elif ack_type == "error":
                    print(f"[run_teleop_real_to_real] 데이터셋 기록 프로세스 오류(계속 진행): {ack['message']}")
                elif ack_type == "finalized":
                    print("[run_teleop_real_to_real] dataset.finalize() 호출 완료")
                    print(f"[run_teleop_real_to_real] 총 {ack['total_episodes']} episodes 저장됨: {args.root}")
                    break
            writer_proc.join()
    finally:
        if preview is not None:
            preview.close()
        safe_disconnect(leader, "리더암")
        safe_disconnect(follower, "팔로워암")


if __name__ == "__main__":
    main()
