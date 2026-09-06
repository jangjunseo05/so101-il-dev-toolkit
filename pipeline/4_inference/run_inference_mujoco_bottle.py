"""학습된 ACT 체크포인트로 MuJoCo 시뮬레이션에서 병 파지·따르기 policy가
자율적으로 움직이는 걸 시각적으로 확인하기 위한 순수 추론 스크립트.

`run_inference_mujoco.py`(사각형 블록 pick&place용)를 복사해 만들었다 --
원본은 수정하지 않았고, 이 스크립트는 그것과 완전히 독립적으로 실행된다.
차이는 오직 카메라 구성뿐이다:
  - 이 스크립트: front_cam + wrist_cam, 320x240, 실제 웹캠으로 학습된 체크포인트
  - 원본: wrist_cam 1대, 640x480, MuJoCo 렌더 자체로 학습된 체크포인트

⚠️ 이 프로젝트(다른 저장소 so101-mobile-manipulation-main의 nl_pick_pour.py)가
쓰는 체크포인트는 **실제 웹캠 사진**으로 학습됐다. 여기서 주는 관측치는
MuJoCo 합성 렌더다 -- 학습/추론 이미지 도메인이 완전히 다르므로, 이 스크립트는
"policy가 병을 실제로 집고 따르는지"를 검증하지 않는다. 검증 범위는 오직
파이프라인이 에러 없이 돌고 action이 유한값(NaN/Inf 없음)으로 나오는지뿐이다
(스모크테스트, 사용자 결정 사항).

사용법:
  뷰어로 직접 관찰(기본):
    python run_inference_mujoco_bottle.py
    python run_inference_mujoco_bottle.py --checkpoint <다른 체크포인트 경로>

  R 키: 현재 episode를 버리고 초기 상태로 리셋 후 policy.reset()도 호출.

  헤드리스 검증(뷰어 없이 짧게 몇 초만 돌려서 에러 유무만 확인):
    python run_inference_mujoco_bottle.py --headless --duration 5
"""

import argparse
import ctypes
import sys
import time
import traceback
from pathlib import Path


def _setup_file_logging() -> str:
    import datetime
    import os
    import sys

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
    print(f"[run_inference_mujoco_bottle] 세션 로그 시작: {log_path}")
    return log_path


_LOG_PATH = _setup_file_logging()


def _import_mujoco_with_plugin_bypass():
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
            print(f"[run_inference_mujoco_bottle] 플러그인 DLL 로딩 실패, 건너뜀: {name} ({exc})")
            return None

    ctypes.CDLL = _tolerant_cdll
    try:
        import mujoco
        import mujoco.viewer
    finally:
        ctypes.CDLL = real_cdll
    return mujoco


def _ensure_xxhash_importable():
    try:
        import xxhash  # noqa: F401
        return
    except ImportError as exc:
        print(f"[run_inference_mujoco_bottle] xxhash 로딩 실패, hashlib 기반 대체 모듈 사용: {exc}")

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

# so101_web/pipeline/4_inference/run_inference_mujoco_bottle.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = str(PROJECT_ROOT / "robot_model" / "scene_bottle.xml")

sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
from camera_config import (  # noqa: E402
    BOTTLE_FRONT_IMAGE_HEIGHT,
    BOTTLE_FRONT_IMAGE_WIDTH,
    BOTTLE_WRIST_IMAGE_HEIGHT,
    BOTTLE_WRIST_IMAGE_WIDTH,
    INFERENCE_STREAM_PORT,
    SCENE_STREAM_FPS,
    SCENE_STREAM_HEIGHT,
    SCENE_STREAM_WIDTH,
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

# dataset feature 이름(observation.images.<key>) -> MuJoCo 카메라 이름.
# nl_pick_pour.py의 build_camera_config()가 쓴 키 이름과 그대로 맞춘 것 --
# 체크포인트가 이 정확한 feature 키를 기대한다(config.json의 input_features).
CAMERAS = {"front": "front_cam", "wrist": "wrist_cam"}
IMAGE_SIZES = {
    "front": (BOTTLE_FRONT_IMAGE_HEIGHT, BOTTLE_FRONT_IMAGE_WIDTH),
    "wrist": (BOTTLE_WRIST_IMAGE_HEIGHT, BOTTLE_WRIST_IMAGE_WIDTH),
}

CONTROL_HZ = 50  # 물리 스텝 주파수
POLICY_HZ = 30  # 카메라 렌더 + policy 호출 주파수 (학습 데이터셋 fps와 동일)

# 실물 로봇 쪽(nl_pick_pour.py)은 send_action()/ramp_to_pose() 둘 다 틱당 최대
# 이동각(기본 6도)을 강제한다 -- 이 클램프가 없으면(원본 run_inference_mujoco.py도
# 마찬가지로 없음) policy가 학습 때와 완전히 다른 이미지(합성 렌더)를 받아
# 관절 한계까지 튀는 값을 그대로 낼 때(실측: elbow_flex가 min==max==한계값으로
# 포화) MuJoCo 액추에이터가 그 목표를 향해 빠르게 움직여 병을 쳐서 날려버리는
# 문제가 실제로 관측됐다. 병 위치를 옮기는 것보다 이 클램프가 근본 원인(포화된
# 목표값 자체)을 막는 더 확실한 수정이라 실물 쪽과 동일한 방식(현재 실제
# qpos 기준 클로즈드루프 클램프)으로 추가한다.
MAX_STEP_RAD = np.radians(6.0)

# 다른 프로젝트(so101-mobile-manipulation-main)에서 이미 검증된 체크포인트를
# 절대경로로 재사용 -- so101_web 안에서 새로 학습한 체크포인트가 아직 없어도
# 바로 스모크테스트할 수 있는 폴백.
DEFAULT_CHECKPOINT = str(
    Path.home() / "so101_train" / "act_demo_slots_v2" / "checkpoints" / "020000" / "pretrained_model"
)
DEFAULT_TASK = "Grasping bottle at slot A and pouring water"

KEY_RESET_EPISODE = ord("R")


class CameraPreviewWindow:
    """front_cam+wrist_cam을 가로로 이어붙인 실시간 미리보기 창."""

    def __init__(self, width: int, height: int, title: str = "front | wrist"):
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
    def __init__(self):
        self.reset_requested = False


def build_observation(qpos_state: np.ndarray, images_hwc_uint8: dict, task: str) -> dict:
    """MuJoCo에서 읽은 raw 관측치를 policy 입력 dict로 변환.

    LeRobotDataset이 저장하는 이미지 포맷(float32, [0,1] 범위, channel-first)과
    동일하게 맞춘다. `images_hwc_uint8`는 {"front": array, "wrist": array}.
    """
    state_tensor = torch.from_numpy(qpos_state)
    obs = {"observation.state": state_tensor, "task": task}
    for key, image in images_hwc_uint8.items():
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        obs[f"observation.images.{key}"] = image_tensor
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help="ACTPolicy 체크포인트 디렉터리 (config.json + model.safetensors가 있는 pretrained_model 폴더)")
    parser.add_argument("--device", default="cuda", help="추론에 사용할 디바이스 (cuda/cpu)")
    parser.add_argument("--hz", type=float, default=CONTROL_HZ, help="물리 스텝 주파수(Hz)")
    parser.add_argument("--policy-hz", type=float, default=POLICY_HZ, help="카메라 렌더 + policy 호출 주파수(Hz)")
    parser.add_argument("--task", default=DEFAULT_TASK, help="policy에 전달할 task 설명 문자열")
    parser.add_argument("--headless", action="store_true",
                         help="뷰어/미리보기 창 없이, 실시간 페이싱 없이 --duration초 분량만 최대한 빨리 돌리고 종료(자동 검증용)")
    parser.add_argument("--duration", type=float, default=5.0,
                         help="--headless일 때 시뮬레이션할 시간(초). 뷰어 모드에서는 사용하지 않음")
    args = parser.parse_args()

    print(f"[run_inference_mujoco_bottle] 체크포인트 로딩: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint, device=args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    print(
        f"[run_inference_mujoco_bottle] policy 로딩 완료: device={policy.config.device} "
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
    renderers = {
        key: mujoco.Renderer(model, height=h, width=w)
        for key, (h, w) in IMAGE_SIZES.items()
    }

    policy.reset()  # episode 시작 -- action chunking 큐 초기화

    if args.headless:
        _run_headless(
            args=args, model=model, data=data, renderers=renderers, policy=policy,
            preprocessor=preprocessor, postprocessor=postprocessor,
            follower_qpos_adr=follower_qpos_adr, follower_actuator_id=follower_actuator_id,
            ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, substeps=substeps,
        )
        for r in renderers.values():
            r.close()
        return

    scene_renderer = mujoco.Renderer(model, height=SCENE_STREAM_HEIGHT, width=SCENE_STREAM_WIDTH)
    scene_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, scene_cam)

    mjpeg_server = None
    try:
        mjpeg_server = SceneStreamServer(port=INFERENCE_STREAM_PORT)
        mjpeg_server.start()
        print(f"[run_inference_mujoco_bottle] 브라우저 스트리밍: http://127.0.0.1:{INFERENCE_STREAM_PORT}/stream")
    except Exception:
        print("[run_inference_mujoco_bottle] 스트리밍 서버 시작 실패(추론은 계속 진행):", flush=True)
        traceback.print_exc()
        mjpeg_server = None

    try:
        preview_h = max(h for h, w in IMAGE_SIZES.values())
        preview_w = sum(w for h, w in IMAGE_SIZES.values())
        preview = CameraPreviewWindow(preview_w, preview_h)
    except Exception:
        print("[run_inference_mujoco_bottle] 카메라 미리보기 창 생성 실패(추론은 계속 진행):", flush=True)
        traceback.print_exc()
        preview = None

    control = EpisodeResetControl()

    def key_callback(keycode):
        try:
            if keycode == KEY_RESET_EPISODE:
                control.reset_requested = True
        except Exception:
            print("[run_inference_mujoco_bottle] key_callback 중 오류(무시):", flush=True)
            traceback.print_exc()

    print("=" * 60)
    print("SO101 ACT policy 자율 추론 (MuJoCo 시뮬레이션, 병 파지·따르기)")
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
                    print("[run_inference_mujoco_bottle] episode 리셋 (R)")

                try:
                    frame_accum += loop_dt
                    if frame_accum >= frame_interval:
                        frame_accum -= frame_interval
                        targets = _select_action(
                            model=model, data=data, renderers=renderers, policy=policy,
                            preprocessor=preprocessor, postprocessor=postprocessor,
                            follower_qpos_adr=follower_qpos_adr, task=args.task,
                            ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, preview=preview,
                        )
                        for act_id, val in zip(follower_actuator_id, targets):
                            data.ctrl[act_id] = val
                except AssertionError:
                    raise
                except Exception:
                    print("[run_inference_mujoco_bottle] policy 추론 중 오류(이번 프레임 건너뜀):", flush=True)
                    traceback.print_exc()

                if mjpeg_server is not None:
                    try:
                        scene_frame_accum += loop_dt
                        if scene_frame_accum >= scene_frame_interval:
                            scene_frame_accum -= scene_frame_interval
                            scene_renderer.update_scene(data, camera=scene_cam)
                            mjpeg_server.publish_frame(scene_renderer.render())
                    except Exception:
                        print("[run_inference_mujoco_bottle] 스트리밍 프레임 렌더 중 오류(이번 프레임 건너뜀, 추론엔 영향 없음):", flush=True)
                        traceback.print_exc()

                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                time_until_next_step = loop_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        for r in renderers.values():
            r.close()
        scene_renderer.close()
        if mjpeg_server is not None:
            mjpeg_server.stop()
        if preview is not None:
            preview.close()


def _render_all_cameras(model, data, renderers) -> dict:
    """모든 카메라를 렌더하고 {key: HWC uint8 array}로 반환, 해상도도 검증한다."""
    images = {}
    for key, cam_name in CAMERAS.items():
        renderers[key].update_scene(data, camera=cam_name)
        image = renderers[key].render()
        expected_h, expected_w = IMAGE_SIZES[key]
        assert image.shape[:2] == (expected_h, expected_w), (
            f"{cam_name} 렌더 shape {image.shape[:2]}가 pipeline/camera_config.py의 "
            f"기대값 ({expected_h}, {expected_w})과 다릅니다 -- 이 상태로 계속 추론하면 "
            "학습 때와 다른 해상도의 이미지가 policy에 들어갑니다."
        )
        images[key] = image
    return images


def _select_action(
    *, model, data, renderers, policy: ACTPolicy, preprocessor, postprocessor,
    follower_qpos_adr, task: str, ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
    preview: "CameraPreviewWindow | None" = None,
) -> np.ndarray:
    images = _render_all_cameras(model, data, renderers)

    if preview is not None:
        preview.update_image(np.concatenate([images["front"], images["wrist"]], axis=1))

    qpos_state = data.qpos[follower_qpos_adr].astype(np.float32)
    raw_obs = build_observation(qpos_state, images, task)

    with torch.inference_mode():
        obs = preprocessor(raw_obs)
        action = policy.select_action(obs)
        action = postprocessor(action)

    action_np = action.squeeze(0).to("cpu").numpy()
    targets = np.clip(action_np, ctrl_lo, ctrl_hi)
    # 실물 로봇과 동일한 틱당 최대 이동각 클램프 (모듈 docstring 상수 참고) --
    # 매번 실제 qpos를 다시 읽어 그 기준으로 제한하는 클로즈드루프 방식이라
    # (ramp_to_pose와 동일 설계) 목표가 아무리 극단적이어도 한 policy 호출당
    # MAX_STEP_RAD 이상은 움직이지 않는다.
    delta = np.clip(targets - qpos_state, -MAX_STEP_RAD, MAX_STEP_RAD)
    targets = (qpos_state + delta).astype(np.float32)
    return targets


def _run_headless(
    *, args, model, data, renderers, policy: ACTPolicy, preprocessor, postprocessor,
    follower_qpos_adr, follower_actuator_id, ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
    substeps: int,
) -> None:
    loop_dt = 1.0 / args.hz
    frame_interval = 1.0 / args.policy_hz
    frame_accum = 0.0
    sim_time = 0.0
    n_physics_steps = 0
    n_policy_calls = 0
    all_actions = []

    print(f"[run_inference_mujoco_bottle] 헤드리스 실행 시작: duration={args.duration}s "
          f"(물리 {args.hz}Hz / policy {args.policy_hz}Hz)")
    while sim_time < args.duration:
        frame_accum += loop_dt
        if frame_accum >= frame_interval:
            frame_accum -= frame_interval
            targets = _select_action(
                model=model, data=data, renderers=renderers, policy=policy,
                preprocessor=preprocessor, postprocessor=postprocessor,
                follower_qpos_adr=follower_qpos_adr, task=args.task,
                ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, preview=None,
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
    print(f"[run_inference_mujoco_bottle] 헤드리스 실행 종료: 물리 스텝 {n_physics_steps}회, "
          f"policy 호출 {n_policy_calls}회")
    print(f"[run_inference_mujoco_bottle] action dtype={actions_arr.dtype}")
    print(f"[run_inference_mujoco_bottle] action에 NaN 포함: {bool(np.isnan(actions_arr).any())}, "
          f"Inf 포함: {bool(np.isinf(actions_arr).any())}")
    print(f"[run_inference_mujoco_bottle] action min/max per joint:")
    for i, name in enumerate(JOINT_NAMES):
        col = actions_arr[:, i]
        print(f"  {name:15s} min={col.min():+.4f} max={col.max():+.4f}")


if __name__ == "__main__":
    main()
