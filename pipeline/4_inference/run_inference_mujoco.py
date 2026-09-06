"""
학습된 ACT 체크포인트로 MuJoCo 시뮬레이션에서 policy가 자율적으로
pick&place를 시도하는 걸 시각적으로 확인하기 위한 순수 추론 스크립트.

run_teleop_real.py의 제어 루프 구조(50Hz 물리 스텝, 30Hz 카메라 렌더,
mujoco.viewer.launch_passive + key_callback, 로그 파일 이중 기록)를
참고해 작성했다. run_teleop_real.py 자체는 수정하지 않았고, 이 스크립트는
그것과 독립적으로 실행된다(리더암 연결도, 데이터셋 녹화도 하지 않음).

leader.get_action() 자리를 ACTPolicy.select_action()으로 그대로 교체한
구조다:
  MuJoCo에서 observation(qpos, wrist_cam 렌더) 읽기
    -> {"observation.state": tensor, "observation.images.wrist_cam": tensor}
    -> preprocessor(obs)  (체크포인트에 저장된 정규화 설정 자동 적용)
    -> policy.select_action(obs)  (action chunking은 내부에서 자동 처리)
    -> postprocessor(action)
    -> MuJoCo data.ctrl에 적용

이미지 렌더링과 policy 호출은 물리 스텝(기본 50Hz)과 분리된 30Hz
캐던스로 수행한다 -- 학습 데이터셋의 fps(30)와 맞추기 위함이며,
run_teleop_real.py가 카메라 프레임을 RECORD_FPS로 스로틀링하던 것과
같은 메커니즘을 재사용했다.

물체(pick_object) 위치는 매 episode 고정 위치(scene.xml 기본 스폰
위치)로 시작한다. run_teleop_real.py의 randomize_pick_object()처럼
학습 때 쓴 것과 동일한 랜덤화 로직을 재사용할 수도 있었지만, 이번
체크포인트는 30 step짜리 파이프라인 스모크 테스트용이라 일반화 성능을
따질 단계가 아니라고 판단했다. 매 시도의 시작 조건을 고정해두면
"policy가 뭔가 의미 있는 방향으로 팔을 움직이는지" 자체를 관찰하기
쉽고, 재현 가능한 조건에서 여러 번 반복 관찰할 수 있다는 이점이 더
크다고 봤다. 본격적으로 학습된 체크포인트로 일반화 성능을 보고 싶어지면
randomize_pick_object() 로직을 그대로 가져와 켜면 된다.

사용법:
  뷰어로 직접 관찰(기본):
    python run_inference_mujoco.py
    python run_inference_mujoco.py --checkpoint <다른 체크포인트 경로>

  R 키: 현재 episode를 버리고 초기 상태(로봇+물체 모두 스폰 위치)로 리셋
        후 policy.reset()도 함께 호출(액션 청킹 큐 초기화).

  헤드리스 검증(뷰어 없이 짧게 몇 초만 돌려서 에러 유무만 확인):
    python run_inference_mujoco.py --headless --duration 5
"""

import argparse
import ctypes
import sys
import time
import traceback
from pathlib import Path


def _setup_file_logging() -> str:
    """세션 시작 시 프로젝트 루트의 logs/ 아래 session_YYYYMMDD_HHMMSS.log를 만들고,
    이 프로세스의 stdout/stderr를 콘솔 + 그 파일에 동시에 쓰도록 한다.

    run_teleop_real.py의 동일한 이름의 함수를 그대로 재사용(복붙)한 것 --
    run_teleop_real.py를 import하면 그 모듈 top-level의 리더암/argparse
    관련 코드까지 같이 실행돼버리므로, 이 스크립트는 독립 실행 가능하도록
    필요한 부분만 별도로 들고 있는다.
    """
    import datetime
    import os
    import sys

    # so101_web/pipeline/4_inference/run_inference_mujoco.py -> so101_web/logs
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
    print(f"[run_inference_mujoco] 세션 로그 시작: {log_path}")
    return log_path


_LOG_PATH = _setup_file_logging()


def _import_mujoco_with_plugin_bypass():
    """일부 Windows 환경(애플리케이션 제어 정책/백신)에서 mujoco 번들 플러그인
    DLL(plugin/sensor.dll 등) 로딩이 차단되어 import mujoco 자체가 실패하는
    문제를 우회한다. run_teleop_real.py와 동일한 로직.
    """
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
            print(f"[run_inference_mujoco] 플러그인 DLL 로딩 실패, 건너뜀: {name} ({exc})")
            return None

    ctypes.CDLL = _tolerant_cdll
    try:
        import mujoco
        import mujoco.viewer
    finally:
        ctypes.CDLL = real_cdll
    return mujoco


def _ensure_xxhash_importable():
    """lerobot이 의존하는 datasets 패키지가 쓰는 xxhash 확장 모듈이 일부
    Windows 환경에서 로딩 차단될 때의 우회. run_teleop_real.py와 동일한 로직
    (이 스크립트는 LeRobotDataset을 쓰지 않지만, lerobot.policies import
    체인이 datasets를 거쳐가므로 동일하게 필요).
    """
    try:
        import xxhash  # noqa: F401
        return
    except ImportError as exc:
        print(f"[run_inference_mujoco] xxhash 로딩 실패, hashlib 기반 대체 모듈 사용: {exc}")

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
import torch

_ensure_xxhash_importable()
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# so101_web/pipeline/4_inference/run_inference_mujoco.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = str(PROJECT_ROOT / "robot_model" / "scene.xml")

sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
from camera_config import (  # noqa: E402
    INFERENCE_STREAM_PORT,
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

WRIST_CAMERA = "wrist_cam"
IMAGE_HEIGHT = WRIST_IMAGE_HEIGHT
IMAGE_WIDTH = WRIST_IMAGE_WIDTH

CONTROL_HZ = 50  # 물리 스텝 주파수 (run_teleop_real.py와 동일)
POLICY_HZ = 30  # 카메라 렌더 + policy 호출 주파수 (학습 데이터셋 fps와 동일, run_teleop_real.py의 RECORD_FPS와 동일 값)

DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "checkpoints" / "010000" / "pretrained_model")
# so101_teleop_real 데이터셋(meta/tasks.parquet)에 실제로 기록된 task 문자열
# 그대로. 학습 때 이 문자열로 학습됐으므로 추론 때도 동일하게 맞춰야 한다
# (빈 문자열/다른 문자열을 넣었을 때 ACT가 이를 실제로 사용하는지는 조사에서
# 확정하지 못했지만, 동일 문자열을 쓰면 최소한 불일치로 인한 위험은 없다).
DEFAULT_TASK = "teleoperate SO101 follower arm from real leader arm"

KEY_RESET_EPISODE = ord("R")


class CameraPreviewWindow:
    """손목 카메라(wrist_cam) 실시간 미리보기 창. run_teleop_real.py의 동일 클래스와
    같음 -- policy가 실제로 무엇을 보고 행동하는지 같이 관찰할 수 있어 그대로 재사용.
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
        self._photo = None

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


class EpisodeResetControl:
    """key_callback(뷰어 스레드)과 메인 루프 사이에서 주고받는 요청 플래그."""

    def __init__(self):
        self.reset_requested = False


def build_observation(
    qpos_state: np.ndarray, wrist_image_hwc_uint8: np.ndarray, task: str
) -> dict:
    """MuJoCo에서 읽은 raw 관측치를 policy 입력 dict로 변환.

    LeRobotDataset이 저장하는 이미지 포맷(float32, [0,1] 범위, channel-first)과
    동일하게 맞춰야 한다 -- lerobot/datasets/video_utils.py의 디코딩 결과 포맷을
    조사에서 확인한 내용 그대로 재현. 배치 차원은 붙이지 않는다: policy
    preprocessor의 to_batch_processor 단계가 1D state/3D image 텐서에 자동으로
    배치 차원을 붙여준다(조사에서 소스로 확인).
    """
    state_tensor = torch.from_numpy(qpos_state)
    image_tensor = (
        torch.from_numpy(wrist_image_hwc_uint8).permute(2, 0, 1).contiguous().float() / 255.0
    )
    return {
        "observation.state": state_tensor,
        f"observation.images.{WRIST_CAMERA}": image_tensor,
        "task": task,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="ACTPolicy 체크포인트 디렉터리 (config.json + model.safetensors가 있는 pretrained_model 폴더)",
    )
    parser.add_argument("--device", default="cuda", help="추론에 사용할 디바이스 (cuda/cpu)")
    parser.add_argument("--hz", type=float, default=CONTROL_HZ, help="물리 스텝 주파수(Hz)")
    parser.add_argument(
        "--policy-hz", type=float, default=POLICY_HZ, help="카메라 렌더 + policy 호출 주파수(Hz)"
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="policy에 전달할 task 설명 문자열")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="뷰어/미리보기 창 없이, 실시간 페이싱 없이 --duration초 분량만 최대한 빨리 돌리고 종료(자동 검증용)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="--headless일 때 시뮬레이션할 시간(초). 뷰어 모드에서는 사용하지 않음",
    )
    args = parser.parse_args()

    print(f"[run_inference_mujoco] 체크포인트 로딩: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint, device=args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    print(
        f"[run_inference_mujoco] policy 로딩 완료: device={policy.config.device} "
        f"chunk_size={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps} "
        f"normalization_mapping={policy.config.normalization_mapping}"
    )

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    follower_qpos_adr = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    follower_actuator_id = [model.actuator(name).id for name in JOINT_NAMES]
    ctrl_lo = model.actuator_ctrlrange[follower_actuator_id, 0]
    ctrl_hi = model.actuator_ctrlrange[follower_actuator_id, 1]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    substeps = max(1, round(1.0 / args.hz / model.opt.timestep))
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    policy.reset()  # episode 시작 -- action chunking 큐 초기화

    if args.headless:
        _run_headless(
            args=args,
            model=model,
            data=data,
            renderer=renderer,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            follower_qpos_adr=follower_qpos_adr,
            follower_actuator_id=follower_actuator_id,
            ctrl_lo=ctrl_lo,
            ctrl_hi=ctrl_hi,
            substeps=substeps,
        )
        renderer.close()
        return

    # 브라우저 내 라이브 스트리밍용 3인칭 씬 렌더러(CLAUDE.md 44절). headless
    # 모드는 위에서 이미 return했으므로 뷰어 모드에서만 생성된다 -- 헤드리스는
    # 동기 헬스체크 용도라 지켜볼 "세션" 자체가 없어 스트리밍이 무의미함.
    # wrist_cam renderer(정책 입력 경로)와는 완전히 분리된 별도 렌더러/카메라라
    # 이 스트림이 실패해도 추론 자체에는 영향이 없다.
    scene_renderer = mujoco.Renderer(model, height=SCENE_STREAM_HEIGHT, width=SCENE_STREAM_WIDTH)
    scene_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, scene_cam)

    mjpeg_server = None
    try:
        mjpeg_server = SceneStreamServer(port=INFERENCE_STREAM_PORT)
        mjpeg_server.start()
        print(
            f"[run_inference_mujoco] 브라우저 스트리밍: http://127.0.0.1:{INFERENCE_STREAM_PORT}/stream"
        )
    except Exception:
        print("[run_inference_mujoco] 스트리밍 서버 시작 실패(추론은 계속 진행):", flush=True)
        traceback.print_exc()
        mjpeg_server = None

    try:
        preview = CameraPreviewWindow(IMAGE_WIDTH, IMAGE_HEIGHT, title=WRIST_CAMERA)
    except Exception:
        print("[run_inference_mujoco] 카메라 미리보기 창 생성 실패(추론은 계속 진행):", flush=True)
        traceback.print_exc()
        preview = None

    control = EpisodeResetControl()

    def key_callback(keycode):
        try:
            if keycode == KEY_RESET_EPISODE:
                control.reset_requested = True
        except Exception:
            print("[run_inference_mujoco] key_callback 중 오류(무시):", flush=True)
            traceback.print_exc()

    print("=" * 60)
    print("SO101 ACT policy 자율 추론 (MuJoCo 시뮬레이션)")
    print(f"체크포인트: {args.checkpoint}")
    print(f"물리 스텝: {args.hz} Hz   policy 호출: {args.policy_hz} Hz")
    print("R 키: episode 리셋(로봇+물체를 초기 위치로, policy 액션 큐도 초기화)")
    print("=" * 60)

    loop_dt = 1.0 / args.hz
    frame_interval = 1.0 / args.policy_hz
    frame_accum = 0.0
    scene_frame_accum = 0.0
    scene_frame_interval = 1.0 / SCENE_STREAM_FPS

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if control.reset_requested:
                    control.reset_requested = False
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    policy.reset()
                    frame_accum = 0.0
                    print("[run_inference_mujoco] episode 리셋 (R)")

                try:
                    frame_accum += loop_dt
                    if frame_accum >= frame_interval:
                        frame_accum -= frame_interval
                        targets = _select_action(
                            model=model,
                            data=data,
                            renderer=renderer,
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            follower_qpos_adr=follower_qpos_adr,
                            task=args.task,
                            ctrl_lo=ctrl_lo,
                            ctrl_hi=ctrl_hi,
                            preview=preview,
                        )
                        for act_id, val in zip(follower_actuator_id, targets):
                            data.ctrl[act_id] = val
                except AssertionError:
                    # 해상도 불일치는 이번 프레임만 건너뛸 문제가 아니라 매 프레임
                    # 계속 재발할 설정 오류이므로, 아래 except Exception과 달리
                    # 삼키지 않고 세션 자체를 중단시킨다.
                    raise
                except Exception:
                    print("[run_inference_mujoco] policy 추론 중 오류(이번 프레임 건너뜀):", flush=True)
                    traceback.print_exc()

                # 브라우저 스트리밍용 씬 렌더 -- policy 입력 경로(위 _select_action의
                # wrist_cam 렌더)와는 독립된 accumulator/try-except를 쓴다. 순수
                # 모니터링용이라 실패해도 삼키기만 하고 추론 자체를 중단시키지 않는다.
                if mjpeg_server is not None:
                    try:
                        scene_frame_accum += loop_dt
                        if scene_frame_accum >= scene_frame_interval:
                            scene_frame_accum -= scene_frame_interval
                            scene_renderer.update_scene(data, camera=scene_cam)
                            mjpeg_server.publish_frame(scene_renderer.render())
                    except Exception:
                        print(
                            "[run_inference_mujoco] 스트리밍 프레임 렌더 중 오류(이번 프레임 건너뜀, 추론엔 영향 없음):",
                            flush=True,
                        )
                        traceback.print_exc()

                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                time_until_next_step = loop_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        renderer.close()
        scene_renderer.close()
        if mjpeg_server is not None:
            mjpeg_server.stop()
        if preview is not None:
            preview.close()


def _select_action(
    *,
    model,
    data,
    renderer,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    follower_qpos_adr,
    task: str,
    ctrl_lo: np.ndarray,
    ctrl_hi: np.ndarray,
    preview: "CameraPreviewWindow | None" = None,
) -> np.ndarray:
    """카메라 렌더 -> policy 추론 -> ctrl 범위로 clip까지 한 번에 수행하고
    float32 목표 관절각 배열을 반환한다.
    """
    renderer.update_scene(data, camera=WRIST_CAMERA)
    wrist_image = renderer.render()
    # 41절: run_teleop_real.py와 동일한 pipeline/camera_config.py 상수를 쓰지만,
    # 실제 렌더러 출력 셰이프는 각 스크립트의 mujoco.Renderer(height=..., width=...)
    # 호출 인자에 달려있어 상수 공유만으로는 보장되지 않는다 -- 학습 시 정책이
    # 본 해상도와 여기서 추론에 실제로 먹이는 해상도가 어긋나면 예외 없이 조용히
    # 틀린 크기의 이미지를 정책에 넣게 되므로, 여기서 즉시 크게 실패하게 한다
    # (뷰어 모드의 바깥 except Exception은 이 AssertionError를 삼키지 않고 다시
    # 던지도록 별도 처리됨. headless 모드는 이 함수를 감싸는 try/except가 아예
    # 없어 그대로 프로세스가 nonzero exit code로 죽는데, 이는 대시보드 트랙A
    # 헬스체크의 기존 실패 판정과 그대로 맞아떨어진다, 33-4절).
    assert wrist_image.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH), (
        f"wrist_cam 렌더 shape {wrist_image.shape[:2]}가 pipeline/camera_config.py의 "
        f"기대값 ({IMAGE_HEIGHT}, {IMAGE_WIDTH})과 다릅니다 -- 이 상태로 계속 추론하면 "
        "학습 때와 다른 해상도의 이미지가 policy에 들어갑니다."
    )
    if preview is not None:
        preview.update_image(wrist_image)

    qpos_state = data.qpos[follower_qpos_adr].astype(np.float32)
    raw_obs = build_observation(qpos_state, wrist_image, task)

    with torch.inference_mode():
        obs = preprocessor(raw_obs)
        action = policy.select_action(obs)
        action = postprocessor(action)

    action_np = action.squeeze(0).to("cpu").numpy()
    # ctrl_lo/ctrl_hi는 model.actuator_ctrlrange(MuJoCo mjtNum=double)에서 온
    # float64라, np.clip()의 numpy 타입 승격 규칙에 의해 action_np가 float32여도
    # 결과가 float64로 올라간다. run_teleop_real.py에서 이미 한 번 겪은 버그
    # 패턴(8절)이라 여기서도 clip 직후 명시적으로 다시 float32로 캐스팅한다.
    targets = np.clip(action_np, ctrl_lo, ctrl_hi).astype(np.float32)
    return targets


def _run_headless(
    *,
    args,
    model,
    data,
    renderer,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    follower_qpos_adr,
    follower_actuator_id,
    ctrl_lo: np.ndarray,
    ctrl_hi: np.ndarray,
    substeps: int,
) -> None:
    """뷰어 없이, 실시간 페이싱 없이 --duration초 분량의 시뮬레이션 시간만 최대한
    빨리 돌린다. 자동 검증(에러 유무, action 값 확인)용.
    """
    loop_dt = 1.0 / args.hz
    frame_interval = 1.0 / args.policy_hz
    frame_accum = 0.0
    sim_time = 0.0
    n_physics_steps = 0
    n_policy_calls = 0
    all_actions = []

    print(
        f"[run_inference_mujoco] 헤드리스 실행 시작: duration={args.duration}s "
        f"(물리 {args.hz}Hz / policy {args.policy_hz}Hz)"
    )
    while sim_time < args.duration:
        frame_accum += loop_dt
        if frame_accum >= frame_interval:
            frame_accum -= frame_interval
            targets = _select_action(
                model=model,
                data=data,
                renderer=renderer,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                follower_qpos_adr=follower_qpos_adr,
                task=args.task,
                ctrl_lo=ctrl_lo,
                ctrl_hi=ctrl_hi,
                preview=None,
            )
            n_policy_calls += 1
            all_actions.append(targets.copy())
            for act_id, val in zip(follower_actuator_id, targets):
                data.ctrl[act_id] = val

        for _ in range(substeps):
            mujoco.mj_step(model, data)
        n_physics_steps += substeps
        sim_time += loop_dt

    actions_arr = np.stack(all_actions) if all_actions else np.empty((0, len(follower_actuator_id)))
    print(
        f"[run_inference_mujoco] 헤드리스 실행 종료: 물리 스텝 {n_physics_steps}회, "
        f"policy 호출 {n_policy_calls}회"
    )
    print(f"[run_inference_mujoco] action dtype={actions_arr.dtype}")
    print(
        f"[run_inference_mujoco] action에 NaN 포함: {bool(np.isnan(actions_arr).any())}, "
        f"Inf 포함: {bool(np.isinf(actions_arr).any())}"
    )
    print(f"[run_inference_mujoco] action min/max per joint:")
    for i, name in enumerate(JOINT_NAMES):
        col = actions_arr[:, i]
        print(f"  {name:15s} min={col.min():+.4f} max={col.max():+.4f}")


if __name__ == "__main__":
    main()
