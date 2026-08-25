"""
MuJoCo 시뮬레이션 팔(마우스로 조작) -> 실물 SO101 팔로워암 텔레오퍼레이션.

robot_model/scene_teleop.xml(액추에이터가 없는 leader 팔 + 액추에이터가 있는
follower 시각화 팔, 2대)을 MuJoCo에 불러온다. leader 팔은 모든 body가
gravcomp="1"이라 중력의 영향을 받지 않고 어떤 자세로 놓아도 그대로 유지되며,
액추에이터가 아예 없어 물리 엔진이 자세를 되돌리려 하지 않는다 -- 그래서 사람이
마우스로 직접 자세를 잡을 수 있다. 매 제어 주기마다 이 leader 팔의 관절각을
읽어:
  1) MuJoCo 안의 follower(시각화용, x=+0.3m) 팔 액추에이터에 그대로 복사해
     실시간으로 따라 움직이는 모습을 보여주고,
  2) 실물 SO101 팔로워암(Feetech 서보, 시리얼 포트로 연결)에 목표 관절각으로
     변환해 전송한다.

pipeline/1_data_collection/run_teleop_real.py(실물 리더암 -> MuJoCo 팔로워)와
정확히 반대 방향이다: 여기서는 사람이 MuJoCo 뷰어 안에서 팔을 움직이고, 그
결과가 실물 로봇을 움직인다.

조작법(README.md에 문서화된 마우스 리더암 조작법과 동일):
  1. Ctrl + 더블클릭으로 leader 팔의 링크(관절)를 선택
  2. Ctrl + 우클릭 드래그로 이동, Ctrl + Shift + 우클릭 드래그로 회전

사용법:
  python run_teleop_follower.py --port COM7

  실물 팔로워암 캘리브레이션 파일이 없으면 최초 연결 시 대화형 캘리브레이션이
  진행된다(팔을 중간 자세로 이동 후 각 관절을 가동 범위 끝까지 움직이라는 안내가
  콘솔에 출력됨 -- run_teleop_real.py의 리더암 캘리브레이션과 동일한 흐름). 이후
  실행부터는 --id로 지정한 이름의 캘리브레이션 파일을 자동으로 재사용한다.

안전:
  - 연결 직후 실물 팔로워암의 현재 자세를 읽어 MuJoCo leader 팔의 초기 자세로
    그대로 반영한다 -- 그렇지 않으면 MuJoCo 기본 자세(대개 관절각 0)로 첫
    프레임부터 순간이동 명령이 나가 실물 팔이 갑자기 튈 위험이 있다.
  - --max-relative-target으로 한 번의 전송에서 허용하는 목표 이동량(관절은 도,
    그리퍼는 %)을 제한한다(기본값 15 -- SOFollower.send_action() 내부의
    ensure_safe_goal_position()이 적용, 0 이하로 주면 비활성화).
  - 데이터셋 녹화 기능은 없음(순수 텔레오퍼레이션만) -- 필요해지면 별도로 추가.
"""

import argparse
import ctypes
import time
import traceback
from pathlib import Path


def _import_mujoco_with_plugin_bypass():
    """일부 Windows 환경(애플리케이션 제어 정책/백신)에서 mujoco 번들 플러그인
    DLL 로딩이 차단되어 import mujoco 자체가 실패하는 문제를 우회한다
    (pipeline/1_data_collection/run_teleop_real.py와 동일한 우회, 이 환경에서
    mujoco를 import하는 모든 스크립트에 공통으로 필요).
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
            print(f"[run_teleop_follower] 플러그인 DLL 로딩 실패, 건너뜀: {name} ({exc})")
            return None

    ctypes.CDLL = _tolerant_cdll
    try:
        import mujoco
        import mujoco.viewer
    finally:
        ctypes.CDLL = real_cdll
    return mujoco


def _ensure_xxhash_importable():
    """lerobot -> datasets가 쓰는 xxhash의 컴파일된 확장 모듈이 같은 애플리케이션
    제어 정책에 막히는 경우 hashlib 기반 대체 모듈로 바꿔치기한다
    (run_teleop_real.py와 동일한 우회).
    """
    try:
        import xxhash  # noqa: F401
        return
    except ImportError as exc:
        print(f"[run_teleop_follower] xxhash 로딩 실패, hashlib 기반 대체 모듈 사용: {exc}")

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
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_XML = str(PROJECT_ROOT / "robot_model" / "scene_teleop.xml")

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
GRIPPER_JOINT = "gripper"

CONTROL_HZ = 50
DEFAULT_FOLLOWER_ID = "full_arm_calibration_follower"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "config"
DEFAULT_MAX_RELATIVE_TARGET = 15.0


def compute_motor_degree_range(calibration, model_resolution: int) -> tuple[float, float]:
    """캘리브레이션의 range_min/range_max가 실제로 매핑되는 도(degree) 양 끝값을
    계산한다. lerobot FeetechMotorsBus._normalize()의 DEGREES 공식과 동일
    (run_teleop_real.py의 compute_leader_degree_range()와 같은 함수 -- 리더/
    팔로워 어느 쪽 캘리브레이션에도 그대로 적용되는 일반 공식이라 이름만
    generic하게 바꿔 이 스크립트에도 독립적으로 둔다).
    """
    mid = (calibration.range_min + calibration.range_max) / 2
    max_res = model_resolution - 1
    lo = (calibration.range_min - mid) * 360 / max_res
    hi = (calibration.range_max - mid) * 360 / max_res
    return lo, hi


def _linear_map(value: float, src_lo: float, src_hi: float, dst_lo: float, dst_hi: float) -> float:
    frac = (value - src_lo) / (src_hi - src_lo)
    return dst_lo + frac * (dst_hi - dst_lo)


def mujoco_qpos_to_action(
    qpos_values: np.ndarray,
    mj_ranges: dict[str, tuple[float, float]],
    motor_ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """MuJoCo leader 팔의 관절각(라디안)을 실물 팔로워의 send_action() 형식
    ({"관절명.pos": 값}, 5개 관절은 도 단위, gripper는 0~100%)으로 변환한다.
    MuJoCo 쪽 range(mj_ranges)와 실물 팔로워의 실측 캘리브레이션 range
    (motor_ranges)를 선형으로 대응시킨다 -- run_teleop_real.py의
    leader_action_to_radians()와 같은 방식(서로 다르게 캘리브레이션된 두
    range를 비례 스케일링)을, 방향만 반대로 적용한 것.
    """
    action: dict[str, float] = {}
    for i, name in enumerate(JOINT_NAMES):
        mj_lo, mj_hi = mj_ranges[name]
        motor_lo, motor_hi = motor_ranges[name]
        value = _linear_map(float(qpos_values[i]), mj_lo, mj_hi, motor_lo, motor_hi)
        value = float(np.clip(value, min(motor_lo, motor_hi), max(motor_lo, motor_hi)))
        action[f"{name}.pos"] = value
    return action


def action_to_mujoco_qpos(
    action: dict[str, float],
    mj_ranges: dict[str, tuple[float, float]],
    motor_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    """mujoco_qpos_to_action()의 역변환. 실물 팔로워의 현재 자세(action 형식)를
    MuJoCo leader 팔의 초기 관절각(라디안)으로 되돌리는 데 쓴다(연결 직후 1회,
    안전을 위한 초기 자세 동기화 용도).

    실물 팔로워의 현재 읽음값이 캘리브레이션 range(motor_lo~motor_hi)를 벗어나면
    (캘리브레이션 스윕이 실제 가동범위를 다 못 담았거나, 캘리브레이션 이후 관절이
    그 범위 밖으로 움직인 적이 있으면 발생 가능) mujoco_qpos_to_action()과 달리
    이 함수는 클램프가 없어 _linear_map()이 mj_ranges 밖으로 외삽될 수 있었다 --
    MuJoCo leader 관절은 limited="true"라 range 밖 qpos를 대입하면 물리 스텝 중
    엔진이 강제로 되돌리면서(constraint force) 사용자가 손대지 않아도 leader가
    스스로 움직이는 것처럼 보이고, 그 값이 그대로 실물 팔로워에도 전송되는 문제로
    이어졌다. mujoco_qpos_to_action()과 대칭으로 여기도 클램프를 건다.
    """
    qpos = np.empty(len(JOINT_NAMES), dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        mj_lo, mj_hi = mj_ranges[name]
        motor_lo, motor_hi = motor_ranges[name]
        value = action[f"{name}.pos"]
        value = float(np.clip(value, min(motor_lo, motor_hi), max(motor_lo, motor_hi)))
        qpos[i] = _linear_map(value, motor_lo, motor_hi, mj_lo, mj_hi)
    return qpos


def prepare_port(port: str, retries: int = 3, retry_delay: float = 1.0) -> None:
    """연결 전에 포트를 열고 바로 닫아, 이전 실행이 비정상 종료되면서 남아있을
    수 있는 스테일 상태를 정리하고 가용성을 미리 확인한다(run_teleop_real.py와
    동일). 실패해도 예외를 던지지 않고 경고만 출력한다 -- 실제 연결은 뒤이어
    follower.connect()가 시도하며, 거기서 나는 에러가 최종적으로 사용자에게
    전달된다.
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
                print(f"[run_teleop_follower] 포트 '{port}' 확보 성공 (시도 {attempt}/{retries})")
            return
        except (serial.SerialException, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)

    print(
        f"[run_teleop_follower] 경고: 포트 '{port}'를 열지 못했습니다 ({last_error}).\n"
        "  이전에 비정상 종료된 python.exe가 아직 포트를 점유 중일 수 있습니다. PowerShell에서:\n"
        "    Get-Process python* | Select-Object Id,ProcessName,Path\n"
        "    Stop-Process -Id <PID> -Force\n"
        "  로 확인/종료한 뒤 다시 실행해 보세요. (일단 연결은 계속 시도합니다)"
    )


def safe_disconnect(follower: SO101Follower) -> None:
    """어떤 상황에서도 예외를 밖으로 전파하지 않고 팔로워암 연결 해제를 시도한다."""
    try:
        if follower.is_connected:
            follower.disconnect()
            print("[run_teleop_follower] 팔로워암 연결 해제 완료.")
    except Exception as exc:  # noqa: BLE001 - 종료 경로이므로 모든 예외를 삼킨다
        print(f"[run_teleop_follower] 팔로워암 연결 해제 중 오류(무시): {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="실물 팔로워암 시리얼 포트 (예: COM7)")
    parser.add_argument("--id", default=DEFAULT_FOLLOWER_ID, help="팔로워암 캘리브레이션 식별자")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="팔로워암 캘리브레이션 JSON(<id>.json)이 있는 디렉터리. 기본값은 config/.",
    )
    parser.add_argument("--hz", type=float, default=CONTROL_HZ, help="제어 루프 주파수(Hz)")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=DEFAULT_MAX_RELATIVE_TARGET,
        help=(
            "한 번의 전송에서 허용하는 최대 목표 이동량(5개 관절은 도, gripper는 "
            "%%). 0 이하로 주면 비활성화(권장하지 않음 -- 안전 클램프가 완전히 "
            "꺼짐)."
        ),
    )
    args = parser.parse_args()

    max_relative_target = args.max_relative_target if args.max_relative_target > 0 else None

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    leader_qpos_adr = [model.jnt_qposadr[model.joint(f"leader_{name}").id] for name in JOINT_NAMES]
    follower_actuator_id = [model.actuator(f"follower_{name}").id for name in JOINT_NAMES]
    mj_ranges = {
        name: tuple(model.jnt_range[model.joint(f"leader_{name}").id]) for name in JOINT_NAMES
    }

    substeps = max(1, round(1.0 / args.hz / model.opt.timestep))

    prepare_port(args.port)
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            calibration_dir=args.calibration_dir,
            max_relative_target=max_relative_target,
            use_degrees=True,
        )
    )
    calibration_fpath = args.calibration_dir / f"{args.id}.json"
    print(
        f"[run_teleop_follower] 팔로워암 캘리브레이션 파일: {calibration_fpath}"
        f" ({'존재함' if calibration_fpath.is_file() else '없음 -- 최초 연결 시 새로 캘리브레이션 진행됨'})"
    )

    try:
        follower.connect()

        # gripper를 제외한 5개 관절: 실물 팔로워의 실측 캘리브레이션 range를
        # 도(degree) 단위 양 끝값으로 변환해(motor_ranges) MuJoCo leader 팔의
        # 관절 range(mj_ranges, 라디안)와 짝짓는다 -- run_teleop_real.py가 리더의
        # 실측 range를 팔로워 MuJoCo 모델과 짝짓던 것과 정확히 같은 방식이다.
        degree_joint_names = [n for n in JOINT_NAMES if n != GRIPPER_JOINT]
        motor_ranges: dict[str, tuple[float, float]] = {GRIPPER_JOINT: (0.0, 100.0)}
        for name in degree_joint_names:
            calibration = follower.bus.calibration[name]
            model_name = follower.bus.motors[name].model
            resolution = follower.bus.model_resolution_table[model_name]
            motor_ranges[name] = compute_motor_degree_range(calibration, resolution)

        print("[run_teleop_follower] MuJoCo leader range -> 실물 팔로워 calibrated range 스케일링:")
        for name in JOINT_NAMES:
            mj_lo, mj_hi = mj_ranges[name]
            m_lo, m_hi = motor_ranges[name]
            unit = "%" if name == GRIPPER_JOINT else "deg"
            print(
                f"  {name:15s} mujoco=[{mj_lo:+.4f}, {mj_hi:+.4f}]rad -> "
                f"follower=[{m_lo:+.2f}, {m_hi:+.2f}]{unit}"
            )

        # 연결 직후 실물 팔로워의 현재 자세를 읽어 MuJoCo leader 팔의 초기 자세로
        # 반영한다 -- 하지 않으면 MuJoCo 기본 자세(관절각 0)를 향해 첫 프레임부터
        # 순간이동 명령이 나가 실물 팔이 갑자기 튈 위험이 있다.
        initial_obs = follower.get_observation()
        initial_action = {f"{name}.pos": initial_obs[f"{name}.pos"] for name in JOINT_NAMES}

        # 실물 팔로워의 현재 읽음값이 캘리브레이션 range를 벗어나 있으면(=캘리브레이션
        # 스윕이 실제 가동범위를 다 못 담았거나, 관절이 캘리브레이션 이후 범위 밖으로
        # 움직인 적이 있다는 신호), action_to_mujoco_qpos()의 클램프가 조용히 값을
        # 잘라내기 전에 사용자가 원인을 바로 알 수 있도록 경고를 남긴다.
        out_of_range = []
        for name in JOINT_NAMES:
            m_lo, m_hi = motor_ranges[name]
            value = initial_action[f"{name}.pos"]
            lo, hi = min(m_lo, m_hi), max(m_lo, m_hi)
            if value < lo or value > hi:
                out_of_range.append((name, value, lo, hi))
        if out_of_range:
            print(
                "[run_teleop_follower] 경고: 아래 관절의 현재 실물 위치가 캘리브레이션 "
                "range를 벗어나 있습니다 -- 캘리브레이션이 이 관절의 실제 가동범위를 "
                "다 못 담았을 가능성이 높습니다. lerobot-calibrate로 재캘리브레이션을 "
                "권장합니다:"
            )
            for name, value, lo, hi in out_of_range:
                print(f"  {name:15s} 현재={value:+.2f}  캘리브레이션 range=[{lo:+.2f}, {hi:+.2f}]")

        initial_qpos = action_to_mujoco_qpos(initial_action, mj_ranges, motor_ranges)
        for adr, val in zip(leader_qpos_adr, initial_qpos):
            data.qpos[adr] = val
        for act_id, val in zip(follower_actuator_id, initial_qpos):
            data.ctrl[act_id] = val
        mujoco.mj_forward(model, data)
        print(f"[run_teleop_follower] 실물 팔로워 현재 자세로 MuJoCo leader 초기화 완료: {initial_action}")

        print("=" * 60)
        print("MuJoCo 시뮬레이션 팔(마우스 조작) -> 실물 SO101 팔로워암 텔레오퍼레이션")
        print(f"팔로워암 포트: {args.port}  (id={args.id})")
        print(f"제어 주기: {args.hz} Hz   max_relative_target={max_relative_target}")
        print("Ctrl+더블클릭으로 leader 팔(왼쪽, x=-0.3m) 링크 선택 후")
        print("Ctrl+우클릭 드래그로 이동, Ctrl+Shift+우클릭 드래그로 회전하세요.")
        print("=" * 60)

        loop_dt = 1.0 / args.hz

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                try:
                    leader_qpos = data.qpos[leader_qpos_adr].copy()

                    # MuJoCo 안의 follower(시각화용) 팔은 leader와 동일한 소스
                    # 파일(so101_new_calib.xml)에서 만들어져 관절 range가
                    # 정확히 같으므로, 스케일링 없이 그대로 복사하면 된다.
                    for act_id, val in zip(follower_actuator_id, leader_qpos):
                        data.ctrl[act_id] = val

                    action = mujoco_qpos_to_action(leader_qpos, mj_ranges, motor_ranges)
                    follower.send_action(action)
                except Exception:
                    print("[run_teleop_follower] 제어 스텝 중 오류(이번 프레임 건너뜀):", flush=True)
                    traceback.print_exc()

                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                time_until_next_step = loop_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
