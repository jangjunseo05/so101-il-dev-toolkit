"""리더암 포트 찾기와 모터 응답 시험 (홈 화면 단계별 가이드에서 사용).

모터 시험은 lerobot `SO101Leader.bus.connect(handshake=True)`만 호출한다 --
6개 모터 ID가 모두 응답하는지 확인하고 바로 닫는다. 캘리브레이션이나 모터
설정 쓰기는 하지 않는다(`leader.connect()`는 캘리브레이션 파일이 없으면 콘솔
대화형 캘리브레이션으로 들어가므로 쓰지 않음).
"""

from __future__ import annotations

from pathlib import Path

# 자주 쓰이는 USB-시리얼 칩 (설명 표시용 추정일 뿐, 판정에는 쓰지 않음)
_KNOWN_VID = {"1A86": "WCH CH34x", "0403": "FTDI", "10C4": "Silicon Labs CP210x", "2E8A": "RP2040"}


def list_serial_ports() -> list[dict]:
    from serial.tools import list_ports

    ports = []
    for p in sorted(list_ports.comports(), key=lambda p: p.device):
        chip = _KNOWN_VID.get(f"{p.vid:04X}") if p.vid is not None else None
        ports.append({"device": p.device, "description": p.description or "", "chip": chip, "usb": p.vid is not None})
    return ports


def ping_leader(port: str, leader_id: str, calibration_dir: str) -> tuple[bool, str]:
    """(성공 여부, 설명). 포트를 열고 6개 모터 응답을 확인한 뒤 닫는다."""
    from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

    leader = SO101Leader(SOLeaderTeleopConfig(port=port, id=leader_id, calibration_dir=Path(calibration_dir)))
    bus = leader.bus
    try:
        bus.connect(handshake=True)
        return True, f"{port}: 모터 {len(bus.motors)}개가 모두 응답했습니다."
    except Exception as e:
        text = str(e)
        if "Missing motor IDs" in text:
            return False, (
                f"{port}는 열렸지만 모터가 응답하지 않습니다. 리더암 **전원 어댑터**와 보드-모터 케이블을 확인하세요."
                f"\n\n```\n{text[:600]}\n```"
            )
        if "could not open port" in text.lower() or "permission" in text.lower():
            return False, (
                f"{port}를 열 수 없습니다. 포트 번호가 맞는지, 다른 프로그램(녹화 세션·시리얼 모니터)이 "
                f"쓰고 있지 않은지 확인하세요.\n\n```\n{text[:400]}\n```"
            )
        return False, f"{type(e).__name__}: {text[:600]}"
    finally:
        try:
            if bus.is_connected:
                bus.disconnect()
        except Exception:
            pass
