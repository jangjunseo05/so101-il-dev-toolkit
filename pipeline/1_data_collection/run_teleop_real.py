"""
SO101 실물 리더암 -> MuJoCo 팔로워암 텔레오퍼레이션.

scene.xml(팔로워 1대)만 MuJoCo에 불러오고, 매 제어 주기마다 실물 SO101
리더암(Feetech 서보, 시리얼 포트로 연결)의 관절각을 읽어 MuJoCo 팔로워의
포지션 액추에이터로 그대로 전달한다.

run_teleop_simul.py와 달리 리더암은 MuJoCo 안에 존재하지 않는다. 대신
lerobot.teleoperators.so_leader.SO101Leader로 실물 로봇과 통신한다.

사용법:
  python run_teleop_real.py --port COM5

  실물 리더암 캘리브레이션 파일이 없으면 최초 연결 시 대화형 캘리브레이션이
  진행된다(팔을 중간 자세로 이동 후 각 관절을 가동 범위 끝까지 움직이라는
  안내가 콘솔에 출력됨). 이후 실행부터는 --id로 지정한 이름의 캘리브레이션
  파일을 자동으로 재사용한다.

조작법:
  - 실물 리더암을 손으로 움직이면 MuJoCo 뷰어의 팔로워암이 그대로 추종한다.
  - S 키: episode 녹화 시작/종료(종료 시 LeRobot 데이터셋으로 저장하고,
    pick_object를 place_plate 기준 랜덤 위치로 재배치).
  - X 키: 녹화 중인 episode를 저장하지 않고 버림(pick_object는 초기 위치로 복귀).

녹화되는 데이터에는 관절 값(action/observation.state)뿐 아니라 팔로워 그리퍼에
달린 손목 카메라(wrist_cam)의 시각 데이터(observation.images.wrist_cam)도 포함된다.
손목 카메라 화면은 별도의 미리보기 창으로 항상(녹화 여부와 무관하게) 표시된다.
"""

import argparse
import ctypes
import random
import signal
import sys
import time
import traceback
from pathlib import Path


def _setup_file_logging() -> str:
    """세션 시작 시 프로젝트 루트의 logs/ 아래 session_YYYYMMDD_HHMMSS.log를 만들고,
    이 프로세스의 stdout/stderr를 콘솔 + 그 파일에 동시에 쓰도록 한다.

    기존 print()/traceback.print_exc() 호출부는 하나도 건드리지 않는다 -- 그
    호출들이 실제로 쓰는 sys.stdout/sys.stderr 객체 자체를, 같은 내용을 파일에도
    같이 쓰는 객체로 바꿔치기하는 방식이라 모든 콘솔 출력이 로그 파일에도 그대로
    남는다.
    """
    import datetime
    import os
    import sys

    # so101_web/pipeline/1_data_collection/run_teleop_real.py -> so101_web/logs
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
    print(f"[run_teleop_real] 세션 로그 시작: {log_path}")
    return log_path


_LOG_PATH = _setup_file_logging()


def _import_mujoco_with_plugin_bypass():
    """일부 Windows 환경(애플리케이션 제어 정책/백신)에서 mujoco 번들 플러그인
    DLL(plugin/sensor.dll 등) 로딩이 차단되어 import mujoco 자체가 실패하는
    문제를 우회한다. 이 프로젝트의 XML은 MuJoCo <plugin> 태그를 쓰지 않으므로
    차단된 플러그인은 조용히 건너뛰어도 기능에 영향이 없다.
    """
    # LoadLibrary가 차단된 DLL을 만나면 Windows가 "잘못된 이미지" 팝업(하드
    # 에러)을 띄운다. SetErrorMode로 팝업을 억제하고 예외로만 받게 한다.
    # 주의: 이 플래그는 프로세스 전역이다. import 직후 원래 값으로 되돌리는 것도
    # 고려했지만, 뒤이어 import되는 lerobot -> datasets -> xxhash 등 다른 서드파티
    # 패키지의 DLL이 같은 애플리케이션 제어 정책에 걸릴 경우 억제가 풀려 있으면
    # 그 시점에 하드 에러 팝업이 뜨며 스크립트가 멈출 수 있음을 실제로 확인했다.
    # 그래서 프로세스 전체 수명 동안 유지한다(하드 크래시가 나면 콘솔에 아무 표시
    # 없이 종료될 수 있다는 트레이드오프가 있지만, 팝업으로 인한 행(hang)이 더
    # 나쁜 실패 모드라 이쪽을 택함). 대신 아래 key_callback/제어 루프는 예외를
    # 직접 잡아 로그를 남기도록 해 "이유 없이 조용히 꺼짐"을 최대한 방지한다.
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_NOOPENFILEERRORBOX = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
    )

    real_cdll = ctypes.CDLL

    def _tolerant_cdll(name, *args, **kwargs):
        try:
            return real_cdll(name, *args, **kwargs)
        except OSError as exc:
            print(f"[run_teleop_real] 플러그인 DLL 로딩 실패, 건너뜀: {name} ({exc})")
            return None

    ctypes.CDLL = _tolerant_cdll
    try:
        import mujoco
        import mujoco.viewer
    finally:
        ctypes.CDLL = real_cdll
    return mujoco


def _ensure_xxhash_importable():
    """일부 Windows 환경에서는 mujoco 플러그인 DLL뿐 아니라, lerobot이 의존하는
    datasets 패키지가 내부적으로 쓰는 xxhash의 컴파일된 확장 모듈(_xxhash)도
    애플리케이션 제어 정책에 막혀 'import xxhash'가 실패한다(동일한 원인,
    다른 DLL). datasets는 xxhash를 로컬 캐시 fingerprint 생성에만 쓰고(허깅페이스
    허브와 동기화되는 값이 아니라 이 프로세스 안에서만 일관되면 됨) 이 프로젝트는
    로컬 데이터셋만 다루므로, import가 막히면 hashlib 기반 대체 모듈로 바꿔치기해
    이후 import 체인이 계속 진행되게 한다.
    """
    try:
        import xxhash  # noqa: F401
        return
    except ImportError as exc:
        print(f"[run_teleop_real] xxhash 로딩 실패, hashlib 기반 대체 모듈 사용: {exc}")

    import hashlib
    import sys
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


mujoco = _import_mujoco_with_plugin_bypass()
import numpy as np

_ensure_xxhash_importable()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

# so101_web/pipeline/1_data_collection/run_teleop_real.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = str(PROJECT_ROOT / "robot_model" / "scene.xml")

sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
from camera_config import (  # noqa: E402
    DATA_COLLECTION_STREAM_PORT,
    SCENE_STREAM_FPS,
    SCENE_STREAM_HEIGHT,
    SCENE_STREAM_WIDTH,
    WRIST_IMAGE_HEIGHT,
    WRIST_IMAGE_WIDTH,
)
from mjpeg_stream import SceneStreamServer  # noqa: E402

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
DEFAULT_DATASET_ROOT = str(PROJECT_ROOT / "data" / "so101_teleop_real")
DEFAULT_REPO_ID = "local/so101_teleop_real"
DEFAULT_TASK = "teleoperate SO101 follower arm from real leader arm"
# so101_web 구조에서 캘리브레이션 파일은 config/full_arm_calibration_leader.json에
# 있다(205_all_motor_calibration_leader.py는 폐기됨, 23절 참고 -- 공식
# lerobot-calibrate CLI가 이 경로에 저장하도록 --calibration-dir로 지정해서 쓸 것).
DEFAULT_LEADER_ID = "full_arm_calibration_leader"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "config"

# episode 녹화 종료(S) 시 pick_object를 place_plate 기준 상대offset으로 랜덤
# 재배치하기 위한 (x, y) offset 범위. 임의로 정한 값이 아니라, 기존 28-episode
# so101_teleop_real 데이터셋을 forward kinematics로 분석해서 얻은 것:
#   1) gripper action이 하위 20%(닫힘 쪽, 그래프상 물체를 쥔 구간과 일치 확인)인
#      프레임만 추림
#   2) 그중 gripperframe site의 world z가 0.03m 미만(pick_object 반높이 0.01 +
#      여유 -- 이송 중 들어올린 상태 제외, 집기/내려놓기 순간만 남김)인 프레임만
#      다시 추림 (9006 프레임 중 719개)
#   3) 그 프레임들의 gripperframe world XY 5~95 percentile을 place_plate 절대
#      위치(0.20, 0.15)에 대한 상대offset으로 환산
# 즉 "그리퍼가 실제로 테이블 높이에서 뭔가를 쥔 채 있었던(집거나 내려놓은)
# 위치들"의 범위이므로, 이 안에서 뽑으면 최소한 그리퍼가 도달해서 집을 수 있는
# 위치라는 근거가 있음(완전한 그랩 성공을 보장하진 않지만, 무작위 추측보다는
# 훨씬 근거 있는 범위).
PICK_OBJECT_OFFSET_X_RANGE = (0.00, 0.11)
PICK_OBJECT_OFFSET_Y_RANGE = (-0.18, 0.01)
# 재배치된 물체가 접시 위에 떨어지지 않도록, 접시 중심(offset=(0,0))으로부터
# 이 반경 안쪽은 후보에서 제외한다(rejection sampling). 실제 제외 반경은
# main()에서 place_plate_radius + pick_object_half_extent + 이 마진으로 계산됨.
PICK_OBJECT_EXCLUSION_MARGIN = 0.01
PICK_OBJECT_MAX_SAMPLE_TRIES = 1000

KEY_TOGGLE_RECORD = ord("S")
KEY_DISCARD_EPISODE = ord("X")


def build_dataset_features() -> dict:
    return {
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
        # ffmpeg가 없는 환경이라 use_videos=False(PNG 저장)로 데이터셋을 만들므로
        # dtype은 "video"가 아니라 "image"로 둔다.
        f"observation.images.{WRIST_CAMERA}": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }


def load_or_create_dataset(repo_id: str, root: str) -> LeRobotDataset:
    import json
    import os
    import shutil

    info_path = os.path.join(root, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            dataset = LeRobotDataset(repo_id, root=root)
            print(
                f"[run_teleop_real] 기존 데이터셋을 이어서 기록합니다: {root} "
                f"(현재 {dataset.meta.total_episodes} episodes)"
            )
            return dataset
        except Exception as exc:
            # 이전 실행이 dataset.finalize() 전에 중단되면 meta/info.json만 남고
            # tasks.parquet 등이 없는 반쪽짜리 폴더가 생긴다. 이 상태로 로드를
            # 시도하면 lerobot이 repo_id를 HF Hub 저장소로 착각해 조회하다
            # 401/RepositoryNotFound로 죽는다. episode가 0개면(잃을 데이터가
            # 없으므로) 백업 후 새로 만들 수 있는 후보지만, 예전엔 이 분기가
            # *자동으로* move+recreate를 실행해 so101_teleop(동일 패턴을 쓰는
            # run_teleop_simul.py 쪽)이 말없이 반복 유실된 전례가 있다
            # (CLAUDE.md 9-1/20절). 그래서 더 이상 자동 진행하지 않고, 경고를
            # 콘솔+로그(파일 logging은 _setup_file_logging()의 stdout Tee를
            # 통해 print()만으로 이미 남는다)에 남긴 뒤 사용자 확인(y 입력)을
            # 받고서만 진행한다.
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            total_episodes = info.get("total_episodes", 0)
            if total_episodes == 0:
                backup = root.rstrip("/\\") + "_corrupted"
                print(
                    f"[run_teleop_real] 경고: {root} 이 손상된 상태로 감지됨 "
                    f"(meta/info.json total_episodes=0, 로딩 실패 사유: {exc}). "
                    f"진행하면 {root} -> {backup} 로 이동한 뒤 새 데이터셋을 생성합니다."
                )
                try:
                    answer = input(
                        f"[run_teleop_real] {root} 을 백업하고 새로 생성할까요? 진행하려면 y, "
                        "그 외 입력(또는 Enter)은 취소하고 종료합니다 [y/N]: "
                    )
                except EOFError:
                    # 비대화형 실행(입력을 받을 수 없는 환경) -- 취소와 동일하게 처리
                    answer = ""
                if answer.strip().lower() != "y":
                    print(f"[run_teleop_real] 사용자 확인 없음 -- {root} 을 건드리지 않고 종료합니다.")
                    raise SystemExit(1)
                if os.path.exists(backup):
                    shutil.rmtree(backup)
                shutil.move(root, backup)
                print(f"[run_teleop_real] 실행됨: {root} -> {backup} 백업 후 새 데이터셋 생성 진행")
            else:
                raise RuntimeError(
                    f"{root} 데이터셋 로딩에 실패했지만 이미 {total_episodes}개의 episode가 "
                    "기록되어 있어 자동으로 삭제/백업하지 않습니다. 폴더 상태를 직접 확인한 뒤 "
                    "복구하거나 --root로 다른 경로를 지정해 다시 실행하세요."
                ) from exc

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=RECORD_FPS,
        features=build_dataset_features(),
        root=root,
        robot_type="so101_follower",
        use_videos=False,
    )
    print(f"[run_teleop_real] 새 데이터셋을 생성합니다: {root}")
    return dataset


class RecorderControl:
    """key_callback(뷰어 스레드)과 메인 루프 사이에서 주고받는 요청 플래그."""

    def __init__(self):
        self.toggle_requested = False
        self.discard_requested = False


class CameraPreviewWindow:
    """손목 카메라(wrist_cam) 실시간 미리보기 창.

    이 환경엔 numpy 호환성 때문에 GUI 기능이 빠진 opencv-python-headless가
    깔려 있어(cv2.imshow 사용 불가) 표준 라이브러리 tkinter + Pillow로
    대신 구현했다. mainloop()을 직접 돌리지 않고, 메인 제어 루프에서 매번
    update_idletasks()/update()만 호출하는 논블로킹 방식이라 MuJoCo 뷰어의
    자체 스텝/렌더 루프를 막지 않는다.
    """

    def __init__(self, width: int, height: int, title: str = "wrist_cam"):
        import tkinter as tk

        self._closed = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo = None  # PhotoImage 참조를 들고 있지 않으면 GC로 사라짐

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

    def close(self) -> None:
        if not self._closed:
            self._on_close()


def prepare_port(port: str, retries: int = 3, retry_delay: float = 1.0) -> None:
    """리더암 연결 전에 포트를 열고 바로 닫아, 이전 실행이 비정상 종료되면서
    남아있을 수 있는 스테일 상태를 정리하고 가용성을 미리 확인한다.

    실패해도 예외를 던지지 않고 경고만 출력한다 (실제 연결은 뒤이어
    leader.connect()가 시도하며, 거기서 나는 에러가 최종적으로 사용자에게
    전달됨).
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
                print(f"[run_teleop_real] 포트 '{port}' 확보 성공 (시도 {attempt}/{retries})")
            return
        except (serial.SerialException, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)

    print(
        f"[run_teleop_real] 경고: 포트 '{port}'를 열지 못했습니다 ({last_error}).\n"
        "  이전에 비정상 종료된 python.exe가 아직 포트를 점유 중일 수 있습니다. PowerShell에서:\n"
        "    Get-Process python* | Select-Object Id,ProcessName,Path\n"
        "    Stop-Process -Id <PID> -Force\n"
        "  로 확인/종료한 뒤 다시 실행해 보세요. (일단 연결은 계속 시도합니다)"
    )


def safe_disconnect(leader: SO101Leader) -> None:
    """어떤 상황에서도 예외를 밖으로 전파하지 않고 리더암 연결 해제를 시도한다."""
    try:
        if leader.is_connected:
            leader.disconnect()
            print("[run_teleop_real] 리더암 연결 해제 완료.")
    except Exception as exc:  # noqa: BLE001 - 종료 경로이므로 모든 예외를 삼킨다
        print(f"[run_teleop_real] 리더암 연결 해제 중 오류(무시): {exc}")


def compute_leader_degree_range(calibration, model_resolution: int) -> tuple[float, float]:
    """캘리브레이션의 range_min/range_max가 실제로 매핑되는 도(degree) 양 끝값을
    계산한다. lerobot FeetechMotorsBus._normalize()의 DEGREES 공식
    (mid=(range_min+range_max)/2, max_res=model_resolution-1,
    deg=(raw-mid)*360/max_res)을 그대로 재현한 것 -- 직접 호출해서 확인된 공식.
    """
    mid = (calibration.range_min + calibration.range_max) / 2
    max_res = model_resolution - 1
    lo = (calibration.range_min - mid) * 360 / max_res
    hi = (calibration.range_max - mid) * 360 / max_res
    return lo, hi


def leader_action_to_radians(
    action: dict,
    gripper_range: tuple[float, float],
    leader_deg_ranges: dict[str, tuple[float, float]],
    follower_ctrl_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    """SO101Leader.get_action() 결과를 MuJoCo 조인트 각도(라디안)로 변환.

    gripper는 0~100 정규화 값을 follower의 gripper 조인트 range로 선형
    매핑한다(기존 로직 그대로, 변경 없음).

    나머지 5개 관절(shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll)은
    "leader 도 = follower 도"로 그냥 라디안 변환만 하던 기존 1:1 직접 매핑
    대신, gripper와 동일한 방식(leader 실측 캘리브레이션 range -> follower
    actuator_ctrlrange로 비례 스케일링)을 쓴다.

    배경: 이 리더 개체의 실측 캘리브레이션 가동범위가 관절에 따라 follower
    MJCF 모델의 actuator_ctrlrange보다 넓은 경우(wrist_roll +9.9/+4.3deg,
    shoulder_lift +8.9deg, wrist_flex +14.7deg -- 5개 관절 전부 실측 확인됨,
    shoulder_pan/elbow_flex는 반대로 leader가 더 좁아 문제 없었음)가 있어,
    1:1 직접 매핑으로는 그 관절의 가장자리에서 항상 follower 하한/상한에
    clip되는 문제가 있었다(실측: 20 episode 중 15개에서 wrist_roll이 follower
    하한에 고정). 재캘리브레이션으로는 "leader 실측범위가 follower 모델범위보다
    넓다"는 구조 자체가 안 바뀌므로 근본 해결이 안 되고, 비례 스케일링으로
    바꿔야 leader 전체 가동범위가 clip 없이 follower 전체 범위에 대응된다.
    """
    targets = np.empty(len(JOINT_NAMES), dtype=np.float32)
    lo, hi = gripper_range
    for i, name in enumerate(JOINT_NAMES):
        value = action[f"{name}.pos"]
        if name == GRIPPER_JOINT:
            targets[i] = lo + (value / 100.0) * (hi - lo)
        else:
            leader_lo, leader_hi = leader_deg_ranges[name]
            ctrl_lo_j, ctrl_hi_j = follower_ctrl_ranges[name]
            frac = (value - leader_lo) / (leader_hi - leader_lo)
            targets[i] = ctrl_lo_j + frac * (ctrl_hi_j - ctrl_lo_j)
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="실물 리더암 시리얼 포트 (예: COM5)")
    parser.add_argument("--id", default=DEFAULT_LEADER_ID, help="리더암 캘리브레이션 식별자")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help=(
            "리더암 캘리브레이션 JSON(<id>.json)이 있는 디렉터리. 기본값은 이 스크립트와 "
            "같은 폴더로, 205_all_motor_calibration_leader.py의 기본 저장 위치와 일치한다."
        ),
    )
    parser.add_argument("--hz", type=float, default=CONTROL_HZ, help="제어 루프 주파수(Hz)")
    parser.add_argument("--task", default=DEFAULT_TASK, help="episode에 붙일 작업 설명(자연어)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot 데이터셋 repo_id")
    parser.add_argument("--root", default=DEFAULT_DATASET_ROOT, help="데이터셋 저장 경로")
    args = parser.parse_args()

    # 외부(웹 대시보드 등)에서 이 프로세스를 정상 종료시키기 위한 신호 처리
    # (CLAUDE.md 30-2절). 기존에도 while-루프가 자연 종료될 때(뷰어 창을 직접
    # 닫을 때)는 아래 main loop 바깥의 try/finally가 포트 닫기(safe_disconnect)와
    # dataset.finalize()를 이미 수행하지만, 외부 종료 신호는 그 finally를 아예
    # 타지 않고 프로세스가 즉시 회수돼(하드 kill) 정리가 전혀 안 됐었다 --
    # shutdown_requested 플래그로 while 루프를 정상 경로로 빠져나가게 해서
    # 기존 finally 정리 로직을 그대로 타게 만든다(새 정리 로직을 추가하는 게
    # 아니라, 있던 걸 신호에 연결하는 것).
    shutdown_requested = {"flag": False}

    def _handle_termination_signal(signum, frame):
        print(f"[run_teleop_real] 종료 신호({signum}) 수신 -- 안전 종료 절차 시작(포트 닫기/데이터셋 finalize)", flush=True)
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    if hasattr(signal, "SIGBREAK"):
        # Windows 전용(CTRL_BREAK_EVENT). subprocess를 CREATE_NEW_PROCESS_GROUP으로
        # 띄웠을 때 부모가 이 프로세스만 선택적으로 겨냥해 보낼 수 있는 신호라
        # dashboard/lib/process_manager.py의 graceful stop이 이걸 사용한다.
        # 핸들러를 등록하지 않으면 OS 기본 동작(즉시 종료, cleanup 미실행)이 적용된다.
        signal.signal(signal.SIGBREAK, _handle_termination_signal)

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    follower_qpos_adr = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    follower_actuator_id = [model.actuator(name).id for name in JOINT_NAMES]
    ctrl_lo = model.actuator_ctrlrange[follower_actuator_id, 0]
    ctrl_hi = model.actuator_ctrlrange[follower_actuator_id, 1]
    gripper_range = tuple(model.jnt_range[model.joint(GRIPPER_JOINT).id])

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # episode 녹화 종료/폐기(S로 종료, X로 버림) 시 pick_object를 초기 자세로
    # 되돌려 같은 task를 한 번의 실행에서 연속으로 여러 episode 녹화할 수 있게
    # 한다. 시작 시점의 자유관절 qpos(위치 3 + 쿼터니언 4)를 미리 저장해둔다.
    # 씬에 이 조인트가 없으면(다른 scene으로 교체된 경우 등) 기능만 조용히 비활성화.
    try:
        pick_object_joint_id = model.joint("pick_object_free").id
        pick_object_qpos_adr = model.jnt_qposadr[pick_object_joint_id]
        pick_object_qvel_adr = model.jnt_dofadr[pick_object_joint_id]
        pick_object_initial_qpos = data.qpos[
            pick_object_qpos_adr : pick_object_qpos_adr + 7
        ].copy()
    except KeyError:
        pick_object_qpos_adr = None
        pick_object_qvel_adr = None
        pick_object_initial_qpos = None
        print(
            "[run_teleop_real] 'pick_object_free' 조인트를 찾을 수 없어 "
            "녹화 종료/폐기 시 물체 위치 초기화 기능은 비활성화됩니다."
        )

    def reset_pick_object():
        if pick_object_qpos_adr is None:
            return
        data.qpos[pick_object_qpos_adr : pick_object_qpos_adr + 7] = pick_object_initial_qpos
        data.qvel[pick_object_qvel_adr : pick_object_qvel_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        print("[REC] pick_object 위치를 초기 상태로 리셋")

    # place_plate는 자유관절 없는 고정 body(scene.xml에서 world에 고정 확인)라
    # model.body(...).pos가 항상 그 절대 위치와 같다 -- pick_object처럼 매 스텝
    # 바뀌는 qpos를 읽을 필요가 없음. 씬에 없으면(다른 scene 등) 랜덤 재배치
    # 기능만 조용히 비활성화하고, 기존 reset_pick_object()(초기 위치 복귀)는
    # 그대로 유지한다.
    try:
        place_plate_xy = model.body("place_plate").pos[:2].copy()
        place_plate_radius = float(model.geom("place_plate_geom").size[0])
        pick_object_half_extent = float(model.geom("pick_object_geom").size[0])
    except KeyError:
        place_plate_xy = None
        place_plate_radius = None
        pick_object_half_extent = None
        print(
            "[run_teleop_real] 'place_plate' body를 찾을 수 없어 "
            "녹화 종료 시 물체 랜덤 재배치 기능은 비활성화됩니다(초기 위치로만 리셋됩니다)."
        )

    pick_object_rng = random.Random()

    def randomize_pick_object():
        """place_plate 기준 PICK_OBJECT_OFFSET_*_RANGE 안에서 (x, y)를 무작위로
        뽑아 pick_object를 재배치한다. 단 접시 반경 안쪽(place_plate_radius +
        물체 half-extent + 마진)에 떨어지는 후보는 접시 위에 놓이는 걸 막기 위해
        재추첨한다(rejection sampling). z와 방향(quaternion)은 원래 spawn 값을
        그대로 재사용(바닥에 닿는 half-extent 높이 그대로) -- 물체가 허공에서
        시작해 자유낙하하며 튀는 일이 없도록 함(offline 테스트로 z=half-extent
        그대로 두면 정지 상태에서 사실상 즉시(수 스텝 내) 안정화됨을 확인,
        1cm 마진을 주고 낙하시키는 경우와 비교해도 최종 정지 위치는 동일했음).
        그래도 만일을 대비해 재배치 직후 소수 스텝(SETTLE_STEPS) 미리 진행해
        완전히 안정된 상태에서 다음 episode를 시작하게 한다.
        """
        if pick_object_qpos_adr is None or place_plate_xy is None:
            reset_pick_object()
            return
        SETTLE_STEPS = 20
        exclusion_radius = place_plate_radius + pick_object_half_extent + PICK_OBJECT_EXCLUSION_MARGIN
        for _ in range(PICK_OBJECT_MAX_SAMPLE_TRIES):
            offset_x = pick_object_rng.uniform(*PICK_OBJECT_OFFSET_X_RANGE)
            offset_y = pick_object_rng.uniform(*PICK_OBJECT_OFFSET_Y_RANGE)
            if (offset_x**2 + offset_y**2) ** 0.5 >= exclusion_radius:
                break
        else:
            print(
                f"[run_teleop_real] 경고: {PICK_OBJECT_MAX_SAMPLE_TRIES}회 시도해도 "
                "접시 밖 위치를 못 찾음 -- 마지막 후보로 진행합니다(offset 범위와 "
                "제외 반경을 다시 확인해볼 것)."
            )
        new_x = place_plate_xy[0] + offset_x
        new_y = place_plate_xy[1] + offset_y
        data.qpos[pick_object_qpos_adr] = new_x
        data.qpos[pick_object_qpos_adr + 1] = new_y
        data.qpos[pick_object_qpos_adr + 2] = pick_object_initial_qpos[2]
        data.qpos[pick_object_qpos_adr + 3 : pick_object_qpos_adr + 7] = pick_object_initial_qpos[3:7]
        data.qvel[pick_object_qvel_adr : pick_object_qvel_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)
        print(
            f"[REC] 공 위치 재배치: ({new_x:.4f}, {new_y:.4f})  "
            f"(plate 기준 offset=({offset_x:+.4f}, {offset_y:+.4f}))"
        )

    substeps = max(1, round(1.0 / args.hz / model.opt.timestep))

    # 녹화 프레임에 넣을 손목 카메라 이미지를 매번 오프스크린으로 렌더링하기 위한
    # 렌더러. 뷰어(GUI)와는 별도의 렌더 타깃이라 뷰어 창 표시 여부와 무관하게 동작함.
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    # 브라우저 내 라이브 스트리밍용 3인칭 씬 렌더러(CLAUDE.md 44절). 위
    # wrist_cam renderer(녹화/미리보기 경로)와는 완전히 분리된 별도 렌더러/카메라라,
    # 이 스트림이 실패해도 녹화 자체에는 영향이 없다. mujoco.viewer가 처음 열릴 때
    # 쓰는 것과 같은 방식(mjv_defaultFreeCamera)으로 모델 전체가 보이는 기본 시점을
    # 잡는다 -- scene.xml에 씬 전용 카메라를 새로 정의할 필요가 없다.
    scene_renderer = mujoco.Renderer(model, height=SCENE_STREAM_HEIGHT, width=SCENE_STREAM_WIDTH)
    scene_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, scene_cam)

    mjpeg_server = None
    try:
        mjpeg_server = SceneStreamServer(port=DATA_COLLECTION_STREAM_PORT)
        mjpeg_server.start()
        print(
            f"[run_teleop_real] 브라우저 스트리밍: http://127.0.0.1:{DATA_COLLECTION_STREAM_PORT}/stream"
        )
    except Exception:
        # 포트 점유 등으로 스트리밍 서버가 안 떠도 녹화 자체는 계속 진행해야 하므로
        # 실패를 흡수한다(CameraPreviewWindow 생성 실패를 다루는 아래 패턴과 동일).
        print("[run_teleop_real] 스트리밍 서버 시작 실패(녹화는 계속 진행):", flush=True)
        traceback.print_exc()
        mjpeg_server = None

    # 손목 카메라 실시간 미리보기 창. 디스플레이가 없거나 tkinter 초기화가
    # 실패해도(예: 원격 세션) 녹화 자체는 계속 동작해야 하므로 실패를 흡수한다.
    try:
        preview = CameraPreviewWindow(IMAGE_WIDTH, IMAGE_HEIGHT, title=WRIST_CAMERA)
    except Exception:
        print("[run_teleop_real] 카메라 미리보기 창 생성 실패(녹화는 계속 진행):", flush=True)
        traceback.print_exc()
        preview = None

    prepare_port(args.port)
    leader = SO101Leader(
        SOLeaderTeleopConfig(port=args.port, id=args.id, calibration_dir=args.calibration_dir)
    )
    print(
        f"[run_teleop_real] 리더암 캘리브레이션 파일: {args.calibration_dir / f'{args.id}.json'}"
        f" ({'존재함' if (args.calibration_dir / f'{args.id}.json').is_file() else '없음 -- 최초 연결 시 새로 캘리브레이션 진행됨'})"
    )

    try:
        leader.connect()

        # gripper를 제외한 5개 관절: leader의 실측 캘리브레이션 range를 도(degree)
        # 단위 양 끝값으로 변환(leader_deg_ranges)하고, 이를 follower의
        # actuator_ctrlrange(ctrl_lo/ctrl_hi, 라디안)와 짝지어 leader_action_to_radians()의
        # 비례 스케일링에 넘긴다. leader.bus.calibration은 __init__에서 로드했거나
        # connect() 중 calibrate()가 새로 썼을 값과 항상 동일(같은 dict 객체를
        # bus 생성 시 그대로 공유) -- 별도로 JSON을 다시 읽지 않아도 항상
        # 실제 적용된 캘리브레이션과 일치함.
        degree_joint_names = [n for n in JOINT_NAMES if n != GRIPPER_JOINT]
        leader_deg_ranges = {}
        for name in degree_joint_names:
            calibration = leader.bus.calibration[name]
            model_name = leader.bus.motors[name].model
            resolution = leader.bus.model_resolution_table[model_name]
            leader_deg_ranges[name] = compute_leader_degree_range(calibration, resolution)
        follower_ctrl_ranges = {
            name: (float(ctrl_lo[i]), float(ctrl_hi[i]))
            for i, name in enumerate(JOINT_NAMES)
            if name != GRIPPER_JOINT
        }
        print("[run_teleop_real] leader 실측 range -> follower ctrlrange 스케일링:")
        for name in degree_joint_names:
            l_lo, l_hi = leader_deg_ranges[name]
            f_lo, f_hi = follower_ctrl_ranges[name]
            print(
                f"  {name:15s} leader=[{l_lo:+.2f}, {l_hi:+.2f}]deg -> "
                f"follower=[{f_lo:+.4f}, {f_hi:+.4f}]rad"
            )

        dataset = load_or_create_dataset(args.repo_id, args.root)

        control = RecorderControl()

        def key_callback(keycode):
            # 뷰어의 내부 렌더 스레드에서 호출된다. 여기서 예외가 새어나가면 그
            # 스레드가 조용히 죽으면서 뷰어 창이 아무 설명 없이 닫혀버릴 수 있으므로
            # 반드시 이 안에서 잡아 로그만 남긴다.
            try:
                if keycode == KEY_TOGGLE_RECORD:
                    control.toggle_requested = True
                elif keycode == KEY_DISCARD_EPISODE:
                    control.discard_requested = True
            except Exception:
                print("[run_teleop_real] key_callback 중 오류(무시):", flush=True)
                traceback.print_exc()

        print("=" * 60)
        print("SO101 실물 리더암 -> MuJoCo 팔로워암 텔레오퍼레이션")
        print(f"리더암 포트: {args.port}  (id={args.id})")
        print(f"제어 주기: {args.hz} Hz")
        print(
            f"S 키: episode 녹화 시작/종료(종료 시 물체 랜덤 재배치)   "
            f"X 키: 녹화 중인 episode 버리기(초기 위치로 복귀) (task='{args.task}')"
        )
        if preview is not None:
            print(f"손목 카메라({WRIST_CAMERA}) 미리보기 창이 열립니다.")
        print("=" * 60)

        recording = False
        episode_frame_count = 0
        frame_accum = 0.0
        frame_interval = 1.0 / RECORD_FPS
        loop_dt = 1.0 / args.hz
        scene_frame_accum = 0.0
        scene_frame_interval = 1.0 / SCENE_STREAM_FPS

        try:
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                while viewer.is_running() and not shutdown_requested["flag"]:
                    step_start = time.time()

                    try:
                        if control.toggle_requested:
                            control.toggle_requested = False
                            if recording:
                                if episode_frame_count > 0:
                                    # dataset.save_episode()는 datasets 라이브러리의
                                    # Map() 인코딩을 동기적으로 돌려 프레임 수에 따라
                                    # 몇 초씩 메인 스레드를 블로킹한다. 그동안 뷰어가
                                    # 메시지를 처리 못해 Windows가 창을
                                    # "응답 없음"(하얗게 덮임)으로 표시할 수 있는데,
                                    # 멈추거나 크래시난 게 아니라 저장 중인 것이므로
                                    # 미리 알려서 오해를 방지한다.
                                    print(
                                        f"[REC] 저장 중... ({episode_frame_count} frames, "
                                        "몇 초 걸릴 수 있음 -- 이 동안 창이 잠깐 "
                                        "'응답 없음'처럼 보여도 정상입니다)"
                                    )
                                    dataset.save_episode()
                                    print(
                                        f"[REC] 저장 완료: {episode_frame_count} frames "
                                        f"(총 {dataset.meta.total_episodes} episodes)"
                                    )
                                else:
                                    dataset.episode_buffer = None
                                    print("[REC] 빈 episode라 저장하지 않음")
                                recording = False
                                randomize_pick_object()
                            else:
                                recording = True
                                episode_frame_count = 0
                                frame_accum = 0.0
                                print(f"[REC] 녹화 시작 (episode {dataset.meta.total_episodes})")

                        if control.discard_requested:
                            control.discard_requested = False
                            if recording:
                                dataset.episode_buffer = None
                                recording = False
                                episode_frame_count = 0
                                print("[REC] episode 버림")
                                reset_pick_object()

                        action = leader.get_action()
                        targets = leader_action_to_radians(
                            action, gripper_range, leader_deg_ranges, follower_ctrl_ranges
                        )
                        # ctrl_lo/ctrl_hi는 model.actuator_ctrlrange(MuJoCo mjtNum=double)에서
                        # 온 float64라, np.clip()의 numpy 타입 승격 규칙에 의해 targets가
                        # float32 -> float64로 올라간다. 이 float64 값이 그대로
                        # dataset.add_frame()의 "action"으로 들어가면 features 스키마
                        # (build_dataset_features()에서 float32로 선언)와 어긋나 매 프레임
                        # "not of the expected dtype 'float32'" 에러가 난다.
                        targets = np.clip(targets, ctrl_lo, ctrl_hi).astype(np.float32)

                        observation_state = data.qpos[follower_qpos_adr].astype(np.float32)

                        for act_id, val in zip(follower_actuator_id, targets):
                            data.ctrl[act_id] = val
                    except Exception:
                        # 리더암 시리얼 통신 순간 오류 등으로 이번 스텝의 목표값을 못
                        # 구했더라도, 마지막으로 적용된 ctrl 그대로 물리 스텝은 계속
                        # 진행해 뷰어가 멈추거나 죽지 않게 한다(원인은 로그로 남김).
                        print("[run_teleop_real] 제어 스텝 중 오류(이번 프레임 건너뜀):", flush=True)
                        traceback.print_exc()

                    for _ in range(substeps):
                        mujoco.mj_step(model, data)
                    viewer.sync()

                    # 손목 카메라 렌더링은 녹화 여부와 무관하게 항상 RECORD_FPS
                    # 주기로 수행해 미리보기 창이 녹화 중이 아닐 때도 계속 갱신되게
                    # 한다. 녹화 중이면 같은 프레임을 그대로 데이터셋에도 추가한다.
                    try:
                        frame_accum += loop_dt
                        if frame_accum >= frame_interval:
                            frame_accum -= frame_interval
                            renderer.update_scene(data, camera=WRIST_CAMERA)
                            wrist_image = renderer.render()
                            # 41절: run_inference_mujoco.py도 같은 pipeline/camera_config.py
                            # 상수를 쓰지만, 렌더러 자체의 실제 출력은 각 스크립트의
                            # mujoco.Renderer(height=..., width=...) 호출 인자에 달려있어
                            # 상수만 공유한다고 실제 셰이프까지 보장되진 않는다 -- 렌더러
                            # 설정이 실수로 상수와 어긋나면 여기서 즉시 크게 실패하게 한다
                            # (아래 except가 이 AssertionError는 삼키지 않고 다시 던짐).
                            assert wrist_image.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH), (
                                f"wrist_cam 렌더 shape {wrist_image.shape[:2]}가 pipeline/"
                                f"camera_config.py의 기대값 ({IMAGE_HEIGHT}, {IMAGE_WIDTH})과 "
                                "다릅니다 -- 이 상태로 계속 녹화하면 데이터셋에 잘못된 해상도의 "
                                "프레임이 섞여 들어갑니다."
                            )

                            if preview is not None:
                                preview.update_image(wrist_image)

                            if recording:
                                dataset.add_frame(
                                    {
                                        "observation.state": observation_state,
                                        "action": targets,
                                        f"observation.images.{WRIST_CAMERA}": wrist_image,
                                        "task": args.task,
                                    }
                                )
                                episode_frame_count += 1
                    except AssertionError:
                        # 해상도 불일치는 이번 프레임만 건너뛰고 넘어갈 문제가 아니라
                        # 매 프레임 계속 재발할 설정 오류이므로, 아래 except Exception과
                        # 달리 삼키지 않고 세션 자체를 중단시킨다.
                        raise
                    except Exception:
                        print("[run_teleop_real] 카메라 프레임 처리 중 오류(이번 프레임 건너뜀):", flush=True)
                        traceback.print_exc()

                    # 브라우저 스트리밍용 씬 렌더 -- wrist_cam 녹화 경로와는 독립된
                    # accumulator/try-except를 쓴다. 이 블록이 실패하거나(해상도 등
                    # 어떤 이유로든) 스트리밍 서버가 아예 없어도 위 녹화 경로에는
                    # 어떤 영향도 주지 않는다(순수 모니터링용이라 AssertionError로
                    # 세션을 중단시키지도 않음 -- wrist_cam 블록과의 의도적인 차이).
                    if mjpeg_server is not None:
                        try:
                            scene_frame_accum += loop_dt
                            if scene_frame_accum >= scene_frame_interval:
                                scene_frame_accum -= scene_frame_interval
                                scene_renderer.update_scene(data, camera=scene_cam)
                                mjpeg_server.publish_frame(scene_renderer.render())
                        except Exception:
                            print(
                                "[run_teleop_real] 스트리밍 프레임 렌더 중 오류(이번 프레임 건너뜀, 녹화엔 영향 없음):",
                                flush=True,
                            )
                            traceback.print_exc()

                    time_until_next_step = loop_dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                if recording and episode_frame_count > 0:
                    dataset.save_episode()
                    print(
                        f"[REC] 종료 시 자동 저장: {episode_frame_count} frames "
                        f"(총 {dataset.meta.total_episodes} episodes)"
                    )
        finally:
            print("[run_teleop_real] dataset.finalize() 호출 시작")
            dataset.finalize()
            print("[run_teleop_real] dataset.finalize() 호출 완료")
            print(f"[run_teleop_real] 총 {dataset.meta.total_episodes} episodes 저장됨: {args.root}")
    finally:
        renderer.close()
        scene_renderer.close()
        if mjpeg_server is not None:
            mjpeg_server.stop()
        if preview is not None:
            preview.close()
        safe_disconnect(leader)


if __name__ == "__main__":
    main()
