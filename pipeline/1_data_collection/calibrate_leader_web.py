"""SO-101 리더암 캘리브레이션 -- 대시보드(① 데이터 수집)에서 버튼으로 진행하는 버전.

`lerobot-calibrate`(= `SO101Leader.calibrate()`)는 단계 전환을 Enter 키로 받는데,
Windows에서는 `lerobot.utils.utils.enter_pressed()`가 `msvcrt.kbhit()`로 콘솔
키보드를 직접 읽어서 웹 버튼이나 stdin 파이프로는 넘길 수 없다. 그래서 이
스크립트가 `SO101Leader.calibrate()`와 **같은 순서로 같은 lerobot 공개 메서드**를
호출하되, Enter 대신 대시보드가 만드는 신호 파일을 기다린다. lerobot
site-packages는 수정하지 않는다(CLAUDE.md 3절).

순서 (lerobot 0.4.4 `so_leader.py` `calibrate()`와 동일):
  1. bus.disable_torque(), 모든 모터 Operating_Mode=POSITION
  2. [신호 middle] 모든 관절을 가동 범위 중앙에 둔 뒤 -> bus.set_half_turn_homings()
  3. wrist_roll을 뺀 5개 관절을 양 끝까지 움직이는 동안 Present_Position(raw)의
     min/max를 계속 기록 (`MotorsBus.record_ranges_of_motion()`과 같은 계산)
  4. [신호 stop] 기록 종료 -> wrist_roll은 0/4095 (lerobot과 동일)
  5. bus.write_calibration() 후 캘리브레이션 JSON 저장(draccus, lerobot과 같은 형식)

자체 캘리브레이션 스크립트를 폐기한 이유(CLAUDE.md 23절: homing offset 직접 계산,
Enter 순간 1회 스냅샷)는 여기 해당하지 않는다 -- homing은 lerobot의
set_half_turn_homings()가, range는 계속 polling하며 극값을 추적한다.

대시보드와의 통신 (state/calibration/, git 제외 폴더):
  status.json   이 스크립트가 쓰는 현재 단계·관절 값
  cmd_middle    대시보드가 만드는 "중앙 자세 완료" 신호
  cmd_stop      대시보드가 만드는 "범위 기록 완료" 신호
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "state" / "calibration"
FULL_TURN_MOTOR = "wrist_roll"  # lerobot과 동일: range를 기록하지 않고 0~4095로 둠

_cancel = {"flag": False}


def _on_signal(signum, frame):  # 대시보드의 "취소"(CTRL_BREAK) 또는 Ctrl+C
    _cancel["flag"] = True


def write_status(**fields) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fields["updated_at"] = time.time()
    tmp = STATE_DIR / "status.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False)
    # Windows에서는 대시보드가 status.json을 읽는 순간 교체가 막힌다(WinError 5) -- 잠깐 뒤 재시도
    for attempt in range(20):
        try:
            tmp.replace(STATE_DIR / "status.json")
            return
        except PermissionError:
            time.sleep(0.01 * (attempt + 1))
    # 끝내 못 바꾸면 이번 갱신만 건너뛴다(다음 tick에 다시 씀). 최종 단계(done/error)는 아래에서 재시도됨


def wait_for_signal(name: str, on_tick=None, tick_s: float = 0.1) -> None:
    flag = STATE_DIR / name
    while not flag.exists():
        if _cancel["flag"]:
            raise KeyboardInterrupt
        if on_tick is not None:
            on_tick()
        time.sleep(tick_s)
    flag.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="full_arm_calibration_leader")
    parser.add_argument("--calibration-dir", type=Path, default=PROJECT_ROOT / "config")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_signal)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ("cmd_middle", "cmd_stop"):
        (STATE_DIR / stale).unlink(missing_ok=True)

    import draccus
    from lerobot.motors.feetech import OperatingMode
    from lerobot.motors.motors_bus import MotorCalibration
    from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

    calib_path = args.calibration_dir / f"{args.id}.json"
    base = {"port": args.port, "id": args.id, "calibration_path": str(calib_path)}
    write_status(step="connecting", message=f"{args.port} 연결 중", **base)

    leader = SO101Leader(SOLeaderTeleopConfig(port=args.port, id=args.id, calibration_dir=args.calibration_dir))
    bus = leader.bus
    try:
        # leader.connect()는 캘리브레이션이 없으면 콘솔 대화형 calibrate()로 들어가므로
        # 버스만 연다(모터 응답 확인 handshake 포함).
        bus.connect()
        motors = list(bus.motors)
        print(f"[calibrate_leader_web] connected {args.port}, motors={motors}", flush=True)

        bus.disable_torque()
        for motor in motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        def show_positions(step: str, message: str):
            pos = bus.sync_read("Present_Position", motors, normalize=False)
            write_status(step=step, message=message, positions={m: int(v) for m, v in pos.items()}, **base)

        print("[calibrate_leader_web] waiting: move all joints to the middle of their range", flush=True)
        wait_for_signal(
            "cmd_middle",
            on_tick=lambda: show_positions("wait_middle", "모든 관절을 가동 범위 가운데에 두고 '중앙 자세 완료'를 누르세요"),
        )
        homing_offsets = bus.set_half_turn_homings()
        print(f"[calibrate_leader_web] homing offsets: {homing_offsets}", flush=True)

        range_motors = [m for m in motors if m != FULL_TURN_MOTOR]
        start = bus.sync_read("Present_Position", range_motors, normalize=False)
        mins, maxes = dict(start), dict(start)
        print("[calibrate_leader_web] recording ranges of motion", flush=True)

        def record_tick():
            pos = bus.sync_read("Present_Position", range_motors, normalize=False)
            for m in range_motors:
                mins[m] = min(mins[m], pos[m])
                maxes[m] = max(maxes[m], pos[m])
            write_status(
                step="recording",
                message=f"{FULL_TURN_MOTOR}를 뺀 관절을 하나씩 양 끝까지 움직인 뒤 '범위 기록 완료'를 누르세요",
                positions={m: int(v) for m, v in pos.items()},
                mins={m: int(v) for m, v in mins.items()},
                maxes={m: int(v) for m, v in maxes.items()},
                **base,
            )

        wait_for_signal("cmd_stop", on_tick=record_tick, tick_s=0.02)

        same = [m for m in range_motors if mins[m] == maxes[m]]
        if same:
            # lerobot record_ranges_of_motion()도 같은 조건에서 ValueError로 멈춘다 -- 저장하지 않음
            raise ValueError(f"움직이지 않은 관절이 있습니다(min == max): {same}")

        mins[FULL_TURN_MOTOR], maxes[FULL_TURN_MOTOR] = 0, 4095
        calibration = {
            m: MotorCalibration(
                id=bus.motors[m].id,
                drive_mode=0,
                homing_offset=homing_offsets[m],
                range_min=mins[m],
                range_max=maxes[m],
            )
            for m in motors
        }
        write_status(step="saving", message="모터에 쓰고 파일로 저장 중", **base)
        bus.write_calibration(calibration)

        backup = None
        if calib_path.is_file():
            backup_dir = STATE_DIR / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"{args.id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(calib_path, backup)
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        with open(calib_path, "w") as f, draccus.config_type("json"):
            draccus.dump(calibration, f, indent=4)
        print(f"Calibration saved to {calib_path}", flush=True)
        write_status(
            step="done",
            message="캘리브레이션 저장 완료",
            backup=str(backup) if backup else None,
            calibration={
                m: {"homing_offset": int(c.homing_offset), "range_min": int(c.range_min), "range_max": int(c.range_max)}
                for m, c in calibration.items()
            },
            **base,
        )
        return 0
    except KeyboardInterrupt:
        print("[calibrate_leader_web] cancelled -- nothing saved", flush=True)
        write_status(step="cancelled", message="취소됨 (파일은 바뀌지 않았습니다)", **base)
        return 1
    except Exception as e:  # 연결 실패·min==max 등. 로그에는 traceback도 남긴다
        import traceback

        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        if "Missing motor IDs" in str(e):
            # 포트는 열렸는데 모터가 하나도 응답하지 않음 -- 대부분 모터 전원(어댑터) 미연결
            msg = (
                f"{args.port}는 열렸지만 모터가 응답하지 않습니다. 리더암 전원 어댑터와 "
                "보드-모터 케이블 연결을 확인하세요.\n\n" + msg
            )
        write_status(step="error", message=msg, **base)
        return 1
    finally:
        try:
            if bus.is_connected:
                bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
