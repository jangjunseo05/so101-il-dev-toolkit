# 트러블슈팅 기록 (Windows + Anaconda `mujoco_env`)

`REAL_LEADER_SETUP.md`를 따라 실물 SO101 리더암 ↔ MuJoCo 팔로워암 텔레오퍼레이션을
구성하는 과정에서 발생한 문제들과 해결 방법을 정리한다. 같은 환경을 다시 세팅하거나
비슷한 에러가 재발했을 때 참고용.

---

## 1. `import mujoco` 자체가 실패함 (WinError 4551 / "잘못된 이미지" 팝업)

### 증상

```
OSError: [WinError 4551] 애플리케이션 제어 정책에서 이 파일을 차단했습니다
```
과 동시에 Windows GUI 팝업:
```
python.exe - 잘못된 이미지
...\mujoco\plugin\sensor.dll ... 오류 상태 0xc0e90002
```
`mujoco.__init__` 내부의 `_load_all_bundled_plugins()`에서 발생. `run_teleop_real.py`,
`run_teleop_simul.py`, `build_teleop_scene.py` 모두 `import mujoco` 시점에 동일하게
재현됨.

### 원인

- Windows 11의 **Smart App Control** 또는 백신/EDR의 **애플리케이션 제어 정책**이
  서명 검증에 실패한 DLL(`mujoco/plugin/sensor.dll`)의 로딩을 커널 레벨에서 차단.
- 실제로는 `actuator.dll`, `elasticity.dll`, `sdf_plugin.dll`은 정상 로드되고
  **`sensor.dll`만** 차단됨 (해당 파일에 대한 특정 오탐으로 보임).
- Windows 로더가 하드 에러(팝업)를 먼저 띄우려 시도하는데, 이게 `python.exe - 잘못된
  이미지` 팝업. `SetErrorMode`로 억제되지 않으면 프로세스가 팝업 앞에서 멈춘다.
- `os.environ["MUJOCO_PLUGIN_DIR"] = ""` 로 우회를 시도했으나 효과 없었음 — mujoco
  3.11의 `mujoco/__init__.py`를 직접 확인해보면 이 환경변수를 아예 참조하지 않고,
  `PLUGINS_DIR = os.path.join(os.path.dirname(__file__), 'plugin')` 로 경로를
  하드코딩해서 무조건 순회 로딩한다. 즉 이 env var로는 애초에 우회가 불가능한 구조.

### 해결

`import mujoco` 직전에 다음을 적용 (`run_teleop_real.py`, `run_teleop_simul.py`,
`build_teleop_scene.py` 세 파일 모두 `_import_mujoco_with_plugin_bypass()` 함수로
동일하게 처리):

1. `ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
   | SEM_NOOPENFILEERRORBOX)` — 로더가 차단된 DLL을 만나도 GUI 팝업을 띄우지 않고
   즉시 예외로만 반환하게 함.
2. `ctypes.CDLL`을 일시적으로 감싸서, 호출이 `OSError`를 던지면 경고만 출력하고
   `None`을 반환 (import 자체는 계속 진행). import가 끝나면 원본 `CDLL`로 복원.

이 프로젝트의 XML(`scene.xml`, `so101_new_calib.xml`)은 MuJoCo `<plugin>` 태그를
전혀 쓰지 않으므로, 차단된 플러그인을 건너뛰어도 시뮬레이션 동작에 영향 없음. 실제로
모델 로드 → COM 포트 연결 단계까지 정상 진행되는 것을 확인함.

### (선택) 시스템 레벨 근본 해결

`Unblock-File`은 효과 없음 (Mark-of-the-Web 차단이 아니라 정책/EDR 차단이라서).
필요하면:
- 회사/학교 PC: IT 부서에 해당 DLL 해시 예외 처리 요청
- 개인 PC: 설정 → 개인 정보 보호 및 보안 → Windows 보안 → 앱 및 브라우저 컨트롤 →
  **Smart App Control 끄기** (끄면 재활성화는 OS 재설치 필요 — 되돌리기 어려우므로
  위 코드 패치만으로 충분하면 건드리지 않는 게 낫다)

---

## 2. 비정상 종료 시 COM 포트가 계속 잠김

### 증상

`run_teleop_real.py` 실행 중 예외/크래시가 나면, 재실행 시 해당 COM 포트에서
`SerialException` / `PermissionError`가 발생하며 연결이 안 됨.

### 원인

기존 코드 구조상 `leader.connect()`와 `dataset = load_or_create_dataset(...)`가
`try` 블록 **바깥**에 있었음. 연결 도중 핸드셰이크 실패나 데이터셋 생성 오류가 나면
`finally`의 `leader.disconnect()`가 아예 실행되지 않고 포트가 열린 채로 예외가
전파되는 구조적 문제.

추가로, lerobot의 `SOLeader.disconnect()`는 `@check_if_not_connected` 데코레이터가
붙어 있어 **연결 안 된 상태에서 호출하면 `DeviceNotConnectedError`를 새로 던진다.**
방어 없이 `finally`에서 그냥 호출하면, 원래 예외가 정리 과정에서 발생한 다른 예외로
뒤덮이는 2차 문제도 생길 수 있음.

### 해결 (`run_teleop_real.py`)

1. **`try...finally` 재구성**: `leader.connect()` 이후 전체(데이터셋 로드, 뷰어 루프
   전부)를 `try`로 감싸고, 바깥쪽 `finally`에서 항상 `safe_disconnect(leader)`를
   호출. `KeyboardInterrupt`, `OSError`, MuJoCo 관련 예외 등 무엇이 나든 이 경로를
   탄다.
2. **`safe_disconnect(leader)`**: `leader.is_connected`를 먼저 확인하고, 연결된
   경우에만 `disconnect()` 호출. 그 자체도 `try/except`로 감싸 어떤 예외도 밖으로
   전파하지 않음.
3. **`prepare_port(port)`**: 리더암 연결 전에 `pyserial`로 포트를 열었다 바로
   닫아(reset_input_buffer/reset_output_buffer 포함) 스테일 상태를 정리. 실패하면
   최대 3회, 1초 간격 재시도 후 원인이 될 만한 조치를 안내(좀비 `python.exe` 프로세스
   확인/종료)하고, 실제 연결은 계속 시도(치명적 실패로 만들지 않음).

`--port COM_NONEXISTENT`로 실행해서 `prepare_port` 경고 → `safe_disconnect`가 조용히
넘어가고 원래의 `ConnectionError`만 깔끔히 표시되는 것을 확인함.

### PowerShell로 잠긴 COM 포트 강제 해제

**① 포트를 물고 있는 프로세스 확인 후 종료** (가장 흔한 원인 — 죽지 않고 남은
`python.exe`)
```powershell
Get-Process python* | Select-Object Id,ProcessName,Path,StartTime
Stop-Process -Id <위에서_확인한_PID> -Force
```

**② 그래도 안 풀리면 해당 COM 포트만 재시작** (관리자 권한 PowerShell 필요)
```powershell
$portName = "COM5"
$device = Get-PnpDevice | Where-Object { $_.FriendlyName -like "*($portName)*" }
$device | Disable-PnpDevice -Confirm:$false
Start-Sleep -Seconds 2
$device | Enable-PnpDevice -Confirm:$false
```

**③ 그래도 실패하면 USB 자체를 순환** (물리적으로 케이블을 뽑았다 꽂는 것과 동일
효과, 명령은 ②와 같음 — USB-시리얼 칩(CH340 등) 드라이버 노드를 다시 초기화하는
목적으로 별도 안내)

---

## 3. 손상된 LeRobotDataset 폴더로 인한 401 / RepositoryNotFound 에러

### 증상

```
FileNotFoundError: ... datasets\so101_teleop_real\meta\tasks.parquet
...
requests.exceptions.HTTPError: 401 Client Error: Unauthorized ...
huggingface_hub.errors.RepositoryNotFoundError: Repository Not Found for url:
https://huggingface.co/api/datasets/local/so101_teleop_real/refs
```

리더암 연결/해제는 정상 동작했는데(`[run_teleop_real] 리더암 연결 해제 완료.` 출력),
`load_or_create_dataset()`에서 위 에러로 크래시.

### 원인

`datasets/so101_teleop_real/` 폴더를 직접 확인해보니 `meta/info.json`만 있고
`tasks.parquet`, `stats.json`, `episodes/`, `data/` 등은 전혀 없었음
(`total_episodes: 0`). 이전 실행이 `dataset.finalize()` 전에 비정상 종료되면서
`LeRobotDataset.create()`가 끝까지 완료되지 못하고 반쪽짜리 상태로 남은 것.

이 반쪽짜리 로컬 폴더를 "기존 데이터셋"으로 인식해 `LeRobotDataset(repo_id,
root=root)`로 로드를 시도 → `tasks.parquet`이 없어 실패 → 그 폴백으로 `repo_id
="local/so101_teleop_real"`을 **실제 HuggingFace Hub 저장소**로 착각하고 API를 조회
→ 그런 저장소가 없으니 401/RepositoryNotFound로 죽음. (`local/...`이라는 이름 때문에
원격 리포로 취급됨)

### 즉시 조치

손상되어 있던 `datasets/so101_teleop_real`을
`datasets/so101_teleop_real_corrupted_<timestamp>`로 백업 이동 (episode 0개라 데이터
유실 없음).

### 재발 방지 코드 수정 (`run_teleop_real.py`, `run_teleop_simul.py` 양쪽
`load_or_create_dataset()`)

- 기존 데이터셋 로드가 실패하면 `info.json`의 `total_episodes`를 확인.
  - **0개면**: 자동으로 `<root>_corrupted`로 백업 이동 후 새로 생성 (HF Hub 조회로
    빠지지 않고 조용히 복구).
  - **1개 이상이면**: 자동으로 건드리지 않고, 무엇이 문제인지와 수동으로
    확인/복구하는 방법을 안내하는 `RuntimeError`를 던짐 (실제 녹화 데이터를 실수로
    삭제하는 일이 없도록).

가짜 손상 폴더(episode 0개)를 만들어 자동 백업+재생성이 실제로 동작하는 것을
테스트로 확인함.

---

## 4. (참고) pick-and-place 연습용 물체 추가

트러블슈팅은 아니지만 같은 세션에서 진행한 작업이라 기록:

`scene.xml`에 로봇이 실제로 집었다 놓을 수 있는 물체가 전혀 없었음(팔로워암 +
바닥뿐). 코드 변경 없이 `scene.xml`에만 추가:

- `pick_object`: 2.5cm 정육면체, `freejoint`로 붙어 자유롭게 들 수 있음. 그리퍼
  도달 반경을 실제로 순전파(forward kinematics)로 계산해 로봇 앞 x=0.20m 지점에
  배치. 그리퍼 사이에서 미끄러지지 않도록 `friction`/`condim=4` 조정.
- `place_target`: y=0.15m 위치에 초록색 원판 표시(충돌 없는 시각 전용 site).
- 짧은 물리 시뮬레이션(2000 스텝)으로 큐브가 폭발하거나 바닥을 뚫지 않고 안정적으로
  놓이는 것을 검증함.

---

## 수정된 파일 요약

| 파일 | 주요 변경 |
|---|---|
| `run_teleop_real.py` | mujoco 플러그인 우회 로딩, try/finally 재구성, `safe_disconnect`, `prepare_port`, `load_or_create_dataset` 손상 감지/자동 복구 |
| `run_teleop_simul.py` | mujoco 플러그인 우회 로딩, `load_or_create_dataset` 손상 감지/자동 복구 |
| `build_teleop_scene.py` | mujoco 플러그인 우회 로딩 |
| `scene.xml` | pick-and-place용 `pick_object`, `place_target` 추가 |
