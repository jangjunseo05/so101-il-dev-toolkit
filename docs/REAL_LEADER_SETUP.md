# 실물 SO101 리더암 연결 & 테스트 가이드

> 환경 설정이 아직 안 되어 있다면 `docs/NEW_TEAMMATE_SETUP.md`를 먼저 따라가세요.

`run_teleop_real.py`를 사용해 **실물 리더암 → MuJoCo 팔로워암** 텔레오퍼레이션을
처음 실행하기 위한 순서다. 위에서 아래로 그대로 따라 하면 된다.

전제: `mujoco_env` conda 환경이 이미 구성되어 있고(`conda activate mujoco_env`,
환경이 없다면 `docs/NEW_TEAMMATE_SETUP.md` 참고), 이 문서의 모든 명령은
**이 저장소를 clone/복사한 프로젝트 루트 폴더**에서 실행한다.

```powershell
conda activate mujoco_env
cd <이 저장소를 clone/복사한 경로>   # 예: C:\Users\USER\Desktop\so101_web
```

---

## 1단계. USB 연결 및 드라이버 확인

1. 실물 SO101 **리더암**의 서보 드라이버 보드(웨이브쉐어 서보 드라이버 보드 등)를
   USB 케이블로 PC에 연결한다.
2. 로봇 전원(서보 전원, 별도 5V/6V 어댑터)도 함께 연결한다. USB만 꽂고 서보
   전원이 안 들어온 상태면 모터가 응답하지 않는다.
3. Windows 장치관리자(`Win + X` → "장치 관리자")를 열어 **포트(COM & LPT)** 항목에
   새 COM 포트가 나타나는지 확인한다 (보통 `USB-SERIAL CH340` 또는
   `CP210x`류 이름). 목록에 안 뜨면 해당 USB-시리얼 칩 드라이버를 설치한 뒤
   재연결한다.

## 2단계. COM 포트 번호 확인

lerobot에 내장된 포트 탐색 도구로 정확한 COM 번호를 찾는다(장치관리자에서 봤어도
한 번 더 교차 확인 권장).

```powershell
lerobot-find-port
```

- 먼저 현재 연결된 포트 목록이 출력된다.
- 안내에 따라 **리더암 USB 케이블을 뽑고** Enter를 누른다.
- 사라진 포트가 화면에 `The port of this MotorsBus is 'COMx'` 형태로 출력된다.
- **다시 USB 케이블을 꽂는다.** (이후 단계에서 필요)
- 출력된 `COMx` 값을 기억해 둔다 (아래 단계에서 `--port COMx`로 사용).

## 3단계. 단독 연결 테스트 (모터 응답 확인)

본격적으로 캘리브레이션/텔레오퍼레이션을 하기 전에, 리더암과 통신이 되는지만
가볍게 확인한다. `COMx`는 2단계에서 찾은 값으로 바꾼다.

```powershell
python -c "from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig; leader = SO101Leader(SOLeaderTeleopConfig(port='COMx', id='so101_leader')); leader.connect(); print('connected:', leader.is_connected, '| calibrated:', leader.is_calibrated); leader.disconnect()"
```

- 캘리브레이션 파일이 아직 없으면 여기서 대화형 캘리브레이션이 자동으로
  시작된다 → 그대로 진행해도 되고(4단계와 동일한 절차), 지금은 `Ctrl+C`로
  중단하고 4단계에서 정식으로 진행해도 된다.
- `Could not open port` / `PermissionError` 등의 에러가 나면:
  - 포트 번호가 맞는지 다시 확인 (2단계 재실행)
  - 다른 프로그램(Arduino IDE 시리얼 모니터 등)이 같은 포트를 점유하고 있지
    않은지 확인
  - USB 케이블/포트를 바꿔서 재시도

## 4단계. 리더암 캘리브레이션

`run_teleop_real.py`는 최초 실행 시 캘리브레이션 파일이 없으면 자동으로
대화형 캘리브레이션을 시작하지만, 처음에는 아래처럼 **별도로 먼저** 캘리브레이션만
끝내 놓는 것을 권장한다(실패해도 텔레오퍼레이션 루프와 섞이지 않아 디버깅이 쉬움).

```powershell
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COMx --teleop.id=so101_leader
```

콘솔 안내를 그대로 따라간다:

1. `Move ... to the middle of its range of motion and press ENTER...`
   → 리더암을 손으로 움직여 **모든 관절이 가동 범위의 중앙(대략 중립 자세)**
     쪽에 오도록 맞춘 뒤 Enter.
2. `Move all joints except 'wrist_roll' sequentially through their entire
   ranges of motion. Recording positions. Press ENTER to stop...`
   → `wrist_roll`을 제외한 나머지 5개 관절(shoulder_pan, shoulder_lift,
     elbow_flex, wrist_flex, gripper)을 **하나씩 순서대로 최대 가동 범위까지**
     천천히 움직여준다(양 끝까지). 그리퍼도 완전히 열고 완전히 닫아본다.
   → 다 끝나면 Enter.
3. `Calibration saved to ...` 메시지가 뜨면 완료. 저장 경로는 보통
   `C:\Users\USER\.cache\huggingface\lerobot\calibration\teleoperators\so_leader\so101_leader.json`
   이다.

**확인**

```powershell
type C:\Users\USER\.cache\huggingface\lerobot\calibration\teleoperators\so_leader\so101_leader.json
```

파일이 존재하고 6개 모터(`shoulder_pan` ~ `gripper`)의 `homing_offset`,
`range_min`, `range_max` 값이 채워져 있으면 정상이다.

> 캘리브레이션을 다시 하고 싶으면 이 `.json` 파일을 지우고 4단계를 재실행하거나,
> `run_teleop_real.py --id <다른이름>`처럼 다른 id를 줘서 별도 캘리브레이션을
> 새로 만들면 된다.

## 5단계. MuJoCo 팔로워 씬만 단독으로 확인 (선택)

리더암 문제와 MuJoCo/씬 문제를 분리해서 진단하고 싶다면, 팔로워 씬만 먼저
띄워서 뷰어가 정상적으로 열리는지 확인해도 좋다.

```powershell
python -c "import mujoco, mujoco.viewer; m = mujoco.MjModel.from_xml_path('robot_model/scene.xml'); d = mujoco.MjData(m); mujoco.viewer.launch(m, d)"
```

뷰어 창에 SO101 팔로워암 1대가 보이면 정상. 창을 닫아 종료한다.

## 6단계. 실제 텔레오퍼레이션 실행

```powershell
python pipeline/1_data_collection/run_teleop_real.py --port COMx --id so101_leader
```

- 이미 4단계에서 캘리브레이션을 끝냈다면 대화형 캘리브레이션 없이 바로
  MuJoCo 뷰어가 열린다.
- 콘솔에 다음과 같은 배너가 뜨면 정상 시작된 것이다.
  ```
  ============================================================
  SO101 실물 리더암 -> MuJoCo 팔로워암 텔레오퍼레이션
  리더암 포트: COMx  (id=so101_leader)
  제어 주기: 50.0 Hz
  S 키: episode 녹화 시작/종료   X 키: 녹화 중인 episode 버리기 (task='...')
  ============================================================
  ```

### 동작 확인 체크리스트

1. **기본 추종**: 실물 리더암의 `shoulder_pan`(베이스 회전)을 천천히 좌우로
   움직여 본다. → MuJoCo 뷰어 속 팔로워암 베이스가 같은 방향으로, 거의
   실시간으로 따라 움직여야 한다.
2. **관절별 확인**: 나머지 관절(shoulder_lift, elbow_flex, wrist_flex,
   wrist_roll)도 하나씩 움직여 보며 MuJoCo 쪽 해당 관절만 반응하는지 확인한다.
   (엉뚱한 관절이 움직이면 4단계 캘리브레이션을 다시 확인)
3. **그리퍼**: 실물 그리퍼를 완전히 벌렸다 오므렸다 해본다. → MuJoCo
   그리퍼도 거의 끝까지 벌어지고 닫혀야 한다. 중간까지만 움직이거나 방향이
   반대면 캘리브레이션 시 그리퍼 양 끝까지 충분히 움직이지 않은 것이므로
   4단계를 다시 수행한다.
4. **지연/끊김**: 손을 빠르게 움직였을 때 MuJoCo 쪽이 심하게 끊기거나
   지연이 크면 `--hz` 값을 낮춰본다 (예: `--hz 30`).
5. **녹화 테스트**: 뷰어에 포커스가 있는 상태에서 `S` 키를 눌러 녹화를
   시작하고, 리더암을 몇 초 움직인 뒤 다시 `S`를 눌러 종료한다. 콘솔에
   `[REC] 저장 완료: N frames (총 1 episodes)`가 뜨면 정상. `X` 키는 녹화 중
   버리기 테스트용으로 한 번 눌러 `[REC] episode 버림`이 뜨는지 확인한다.
6. 종료는 MuJoCo 뷰어 창을 닫거나 터미널에서 `Ctrl+C`. 콘솔 마지막에
   `[run_teleop_real] 총 N episodes 저장됨: ./data/so101_teleop_real`이
   출력되면 정상 종료된 것이다.

### 저장된 데이터 확인 (녹화했다면)

```powershell
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/so101_teleop_real', root='./data/so101_teleop_real')
print(ds.meta.total_episodes, len(ds))
print(ds[0]['action'], ds[0]['observation.state'])
"
```

---

## 문제 해결(Troubleshooting)

| 증상 | 원인/해결 |
|---|---|
| `lerobot-find-port` 실행 시 포트가 하나도 안 사라짐 | 케이블을 잘못 뽑았거나 인식 자체가 안 된 것. 1단계 드라이버 설치부터 재확인 |
| `PermissionError` / `Access is denied` (포트 열기 실패) | 다른 프로그램이 포트 점유 중이거나 관리자 권한 문제. 다른 시리얼 앱 종료 후 재시도 |
| 캘리브레이션 후에도 관절이 반대로 움직이거나 범위가 이상함 | 4단계에서 특정 관절을 끝까지 움직이지 않은 경우. 캘리브레이션 파일 삭제 후 재실행 |
| MuJoCo 팔로워가 리더 대비 심하게 느리거나 끊김 | `--hz`를 낮추거나(예: 20~30), USB 케이블/허브 품질 확인 (긴 케이블·저가 허브는 시리얼 지연 유발) |
| 그리퍼만 유독 이상하게 움직임 | 그리퍼는 각도가 아니라 0~100 정규화 값으로 오므로, 캘리브레이션 시 완전히 열고 완전히 닫는 동작을 빠뜨리지 않았는지 확인 |
| `dataset.finalize()` 관련 에러로 재실행 안 됨 | 이전 실행이 비정상 종료(`Ctrl+C` 여러 번 강제 종료 등)되어 데이터셋이 깨진 경우. `./data/so101_teleop_real` 폴더를 사용자가 직접 확인 후 필요시 백업/삭제 |
