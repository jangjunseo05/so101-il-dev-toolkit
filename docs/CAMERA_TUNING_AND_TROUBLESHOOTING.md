# 손목 카메라 조정 가이드 & 트러블슈팅 정리

이 문서는 두 부분으로 구성된다.

1. **손목 카메라(wrist_cam) 위치/각도를 직접 조정하는 방법** — 어느 파일의 어느
   위치를 어떻게 고치면 되는지.
2. **이번 세션에서 다룬 트러블슈팅 정리** — 포트폴리오 등에 정리해서 쓸 수 있도록
   증상 → 원인 조사 → 해결/결정 → 검증 순으로 기록.

---

## 1부. 손목 카메라(wrist_cam) 위치/각도 조정하기

### 어느 파일, 어느 위치

카메라는 로봇 자체를 정의하는 **`so101_new_calib.xml`** 파일에, 그리퍼 body
안에 자식으로 붙어 있다.

```xml
<body name="gripper" ...>
  ...
  <site group="3" name="gripperframe" .../>
  <camera name="wrist_cam" pos="-0.02290001 -0.00021812 -0.0631274"
      quat="0.57363481 0.41345266 -0.41345266 -0.57363481" fovy="70"/>
  ...
```

`scene.xml`(실물 리더암용)과 `scene_teleop.xml`(마우스 리더암 시뮬레이션용,
`build_teleop_scene.py`가 이 파일을 읽어 자동 생성)이 전부 이 파일을 기반으로
하므로, **이 카메라 태그 하나만 고치면 두 시나리오 모두에 반영된다.**
`scene_teleop.xml`은 스크립트 실행 시 자동으로 재생성되므로 직접 손댈 필요 없음.

### 파라미터가 뜻하는 것

- **`pos`**: 카메라 위치. `gripper` body의 **로컬 좌표계** 기준 오프셋(미터
  단위)이다. 월드 좌표가 아니라 그리퍼가 어떤 자세든 그리퍼와 함께 움직이는
  좌표계라는 점이 중요하다.
- **`quat`**: 카메라 방향. `(w, x, y, z)` 순서의 쿼터니언, 역시 `gripper` body
  로컬 좌표계 기준.
- **`fovy`**: 수직 시야각(도 단위). 클수록 광각(더 넓게, 어안렌즈처럼), 작을수록
  좁게(줌인).
- **MuJoCo 카메라 렌더링 규칙(고정)**: 카메라는 **자기 자신의 로컬 -Z축 방향을
  바라보고, +Y축이 이미지의 "위쪽"**이다. 이건 이 씬만의 규칙이 아니라 MuJoCo
  전체의 카메라 규약이라, 어디에 붙이든 항상 성립한다.

### 상황별 조정 가이드

| 보이는 문제 | 조정할 파라미터 | 방법 |
|---|---|---|
| 화면이 너무 좁다/넓다 (줌 인/아웃만 하고 싶다) | `fovy` | 숫자만 바꾸면 됨. 좁게(줌인) → 값 감소(예: 70→50), 넓게 → 값 증가(예: 70→90). 가장 안전하고 쉬운 조정. |
| 그리퍼 자기 몸체(하우징)가 화면을 가득 채운다 | `pos` | 카메라를 몸체에서 좀 더 "물러나게" 이동. 아래 스니펫으로 카메라 시점 기준 뒤로/옆으로/위로 이동시킨 새 `pos`를 계산할 수 있음. |
| 보고 싶은 방향(위/아래/좌우)과 다른 곳을 보고 있다 | `quat` | 원시 쿼터니언 숫자를 손으로 계산하는 건 비현실적. 아래 "회전 조정" 스니펫으로 "현재 방향에서 O도만큼 위/아래/좌우로 더 기울이기" 방식으로 계산해서 얻은 새 `quat` 값을 붙여넣는 걸 권장. |
| 이미지가 기울어져(롤) 보인다 | `quat` (카메라 로컬 Z축 기준 회전) | 아래 회전 스니펫에서 axis='z'로 사용. |

카메라 로컬 축의 의미(회전 스니펫에서 `axis` 인자로 사용):
- `x` 축 회전 = 위/아래로 기울이기(pitch)
- `y` 축 회전 = 좌우로 돌리기(pan/yaw)
- `z` 축 회전 = 이미지 기울이기(roll)

### 빠르게 확인하는 방법 (실물 리더암 연결 없이)

XML을 고칠 때마다 실제 리더암을 연결해서 확인할 필요 없다. 아래 스크립트를
`SO101` 폴더에 저장해두고(`preview_wrist_cam.py` 등의 이름으로), 값을 바꿀
때마다 실행해서 결과 이미지를 보면 된다. `python-mujoco` 환경(`mujoco_env`)에서
실행.

```python
import ctypes
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
import mujoco
from PIL import Image

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# 카메라가 뭘 보는지 확인하고 싶은 팔 자세를 여기서 지정 (라디안).
# 전부 0으로 두면 홈 포지션(팔을 편 기본 자세).
pose = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.121,
    "elbow_flex": 0.368,
    "wrist_flex": 1.235,
    "wrist_roll": 0.0,
    "gripper": 1.0,
}
for name, value in pose.items():
    model_joint = model.joint(name)
    data.qpos[model.jnt_qposadr[model_joint.id]] = value
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="wrist_cam")
Image.fromarray(renderer.render()).save("wrist_cam_preview.png")
renderer.close()
print("wrist_cam_preview.png 저장됨 - 열어서 확인하세요.")
```

또는 하드웨어 없이 **마우스로 팔을 움직여보면서** 실시간으로 확인하고 싶다면
`run_teleop_simul.py`를 실행한 뒤(리더암을 마우스로 조작), MuJoCo 뷰어 창에서
`]` 키를 눌러 메인 시점을 `follower_wrist_cam`으로 전환하면 실제 카메라가
보는 화면을 그대로 볼 수 있다(`[` 키로 다시 순환). 다만 이 방법은 XML을
고친 뒤에는 스크립트를 껐다 다시 켜야 반영된다(뷰어를 켠 채로 XML만 바꾼다고
자동 반영되지는 않음).

### 회전(quat) 조정 스니펫

원하는 방향으로 "몇 도만큼 더 기울이기"를 계산해서 새 `quat` 값을 얻는다.

```python
import numpy as np
import mujoco

CURRENT_QUAT = [0.57363481, 0.41345266, -0.41345266, -0.57363481]  # 현재 so101_new_calib.xml 값


def tilt_camera(current_quat_wxyz, axis: str, degrees: float):
    """카메라 자신의 로컬 축(axis='x'|'y'|'z') 기준으로 degrees만큼 회전을 추가한
    새 quat을 반환한다. x=위아래(pitch), y=좌우(pan), z=기울임(roll)."""
    axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    delta_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(delta_quat, axis_vec, np.radians(degrees))
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, np.array(current_quat_wxyz), delta_quat)
    return new_quat


# 예: 카메라를 자기 기준으로 15도 더 아래로 숙이고 싶다면
print(tilt_camera(CURRENT_QUAT, axis="x", degrees=15))
# 출력된 4개 숫자를 so101_new_calib.xml의 quat="..." 자리에 그대로 붙여넣기
```

### 위치(pos) 조정 스니펫 — 카메라 시점 기준으로 이동

`pos`는 `gripper` body 좌표계 기준이라 "카메라 기준 앞/뒤/좌우/위아래"와 축이
일치하지 않는다. 아래 함수는 "카메라 시점에서 얼마나 이동할지"를 입력하면
`gripper` 좌표계 기준의 새 `pos`를 계산해준다.

```python
import numpy as np
import mujoco

CURRENT_POS = [-0.02290001, -0.00021812, -0.0631274]
CURRENT_QUAT = [0.57363481, 0.41345266, -0.41345266, -0.57363481]


def move_camera(pos_parent_frame, quat_wxyz, forward=0.0, right=0.0, up=0.0):
    """카메라 시점 기준(전진 forward, 오른쪽 right, 위 up, 단위: 미터)으로
    이동시킨 새 pos(gripper body 좌표계 기준)를 반환한다."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.array(quat_wxyz))
    rot = rot.reshape(3, 3)
    right_axis, up_axis, back_axis = rot[:, 0], rot[:, 1], rot[:, 2]
    delta = right_axis * right + up_axis * up + back_axis * (-forward)
    return np.array(pos_parent_frame) + delta


# 예: 카메라를 몸체에서 3cm 더 뒤로 빼고 싶다면
print(move_camera(CURRENT_POS, CURRENT_QUAT, forward=-0.03))
# 예: 카메라를 오른쪽으로 1cm, 위로 0.5cm 옮기고 싶다면
print(move_camera(CURRENT_POS, CURRENT_QUAT, right=0.01, up=0.005))
```

값을 계산했으면 `so101_new_calib.xml`의 `<camera name="wrist_cam" .../>` 줄의
`pos`/`quat`를 새 값으로 바꾸고, 위 미리보기 스크립트로 확인 → 만족스러울
때까지 반복하면 된다.

---

## 2부. 트러블슈팅 정리 (포트폴리오용)

### 문제 1: 텔레오퍼레이션 중 'S' 키를 누르면 MuJoCo 창이 이유 없이 꺼짐

**증상**: `run_teleop_real.py` 실행 중 녹화 시작/종료용 'S' 키를 누르면 뷰어
창이 아무 오류 메시지 없이 즉시 닫힘.

**조사 과정**:
- MuJoCo 뷰어의 내장 키 바인딩을 확인(공식 문서/이슈 검색) → 'S'는 원래
  "그림자(shadow) 렌더링 토글" 기능이라 충돌 가능성은 낮음.
- `key_callback` 로직, 녹화 시작/종료 로직 자체를 코드 리뷰 → 명백한 버그를
  찾지 못함.
- `_import_mujoco_with_plugin_bypass()`가 `SetErrorMode()`로 Windows의 하드
  크래시 팝업을 **프로세스 전체 수명 동안** 억제하고 있다는 걸 발견. 이 상태에서
  실제 크래시(access violation 등)가 나면 아무 표시 없이 프로세스가 조용히
  죽는 것과 정확히 일치하는 증상.

**시도했다가 되돌린 방향**: import 직후 `SetErrorMode`를 원래 상태로 복원해
크래시가 보이게 하려고 했으나, 검증 중 **다른 회귀 위험을 실측**함 — 이 복원을
적용하면 뒤이어 import되는 `lerobot → datasets → xxhash` 체인이 같은 Windows
애플리케이션 제어 정책에 걸릴 경우, 억제가 풀려 있어서 오히려 하드 에러 팝업이
뜨며 스크립트가 멈출 수 있음을 확인(→ 문제 4와 동일 원인). "보이지 않는 크래시"
보다 "팝업으로 인한 행(hang)"이 더 나쁜 실패 모드라 판단해 억제는 그대로 두기로
결정.

**최종 해결**: `SetErrorMode`는 건드리지 않는 대신, 실패 가능성이 있는 지점을
전부 `try/except`로 감싸 예외를 삼키지 않고 traceback을 콘솔에 남기도록 함.
- `key_callback` 내부 전체
- 매 제어 스텝의 리더암 통신/제어값 계산 부분 (통신 순간 오류가 나도 마지막
  ctrl 값으로 물리 스텝은 계속 진행 → 뷰어가 멈추거나 죽지 않음)
- 녹화 프레임 처리(카메라 렌더링 + `dataset.add_frame`) 부분

**검증**: 코드 컴파일 확인, 정상 동작 경로(모델 로드/렌더/데이터셋 생성)에 대한
자동화 스모크 테스트로 회귀 없음을 확인. 사용자 실기기 테스트에서 'S' 키가
정상적으로 녹화 토글로 동작함을 확인.

---

### 문제 2: 팔로워 그리퍼에 시각(카메라) 데이터 수집 기능이 없었음

**요구사항**: 관절 값뿐 아니라 그리퍼에 달린 카메라의 이미지도 함께 기록하고,
실시간으로도 볼 수 있어야 함.

**설계 결정**:
- 카메라를 `so101_new_calib.xml`의 `gripper` body에 자식으로 추가(손목/eye-in-
  hand 카메라). 이렇게 하면 `scene.xml`, `scene_teleop.xml` 양쪽에 자동으로
  반영됨(둘 다 이 파일을 기반으로 만들어지므로).
- 카메라 pose(pos/quat)는 감으로 정하지 않고, **순기구학(forward kinematics)
  + look-at 벡터 계산**으로 구한 뒤 실제로 오프스크린 렌더링해 이미지를 눈으로
  확인하며 반복 조정함(1부 문서가 이 과정에서 만든 도구/노하우를 정리한 것).
  - 첫 시도: 그리퍼 body의 특정 기하학적 방향(모터/부품 오프셋 벡터)을 "전진
    방향"으로 가정 → 렌더링해보니 그리퍼 자신의 몸체만 비추는 잘못된 각도였음.
  - 재시도: 기존에 정의돼 있던 `gripperframe` site(그리퍼의 TCP 프레임으로
    추정)의 Z축을 전진 방향으로 사용 → 훨씬 합리적인 화각을 얻음.
  - 그리퍼 하우징이 화면을 가리는 문제 → 카메라를 진행축에서 살짝 옆/위로
    오프셋하고 목표점(look-at target)을 전방으로 밀어 재계산 → 렌더링으로
    최종 확인.
- 데이터셋 스키마: `observation.images.wrist_cam` 피처를 `dtype="image"`
  (video 아님, ffmpeg 미설치 환경이라 PNG 저장 방식 사용), `shape=(H,W,3)`,
  `names=["height","width","channels"]`로 등록. `lerobot`의 `validate_frame`/
  `hw_to_dataset_features` 관례에 맞춰 결정.
- 녹화 시 `mujoco.Renderer`로 매 기록 프레임마다 `wrist_cam` 이미지를
  오프스크린 렌더링해 `dataset.add_frame()`에 함께 전달.
- 실시간 미리보기: 애초엔 `cv2.imshow`를 쓰려 했으나, 이 환경엔 numpy 버전
  호환성 문제로 GUI 기능이 빠진 `opencv-python-headless`가 고정돼 있어 사용
  불가(확인 시 `cvShowImage` 관련 OpenCV 예외 발생). 기존에 고정해둔 패키지
  조합을 건드리지 않기 위해, 표준 라이브러리 `tkinter` + `Pillow`로 논블로킹
  미리보기 창을 직접 구현(`mainloop()` 대신 메인 루프에서 매번
  `update_idletasks()`/`update()` 호출).

**검증**: 렌더러 출력 shape/dtype 확인, `LeRobotDataset.create()` →
`add_frame()`(이미지 포함) → `save_episode()` → `finalize()` → 재로드 →
샘플 읽기까지 전 과정을 실제로 실행해 정상 동작 확인. 미리보기 창도 실제 렌더
이미지로 여러 번 갱신 후 정상 종료되는 것을 확인.

---

### 문제 3: 기존 pick 물체가 실제로 잘 집히지 않음

**증상**: 씬에 있던 2.5cm 정육면체가 그리퍼로 잘 집히지 않음. 온전한
pick-and-place(집어서 접시에 놓기)를 구현해달라는 요청.

**조사 과정**:
- 그리퍼 구조를 오프스크린 렌더링으로 직접 확인한 결과, **일반적인 평행
  집게(parallel jaw gripper)가 아니라 "스쿱(고정 턱) + 후크(가동 턱)" 구조**
  라는 걸 발견(집게가 서로 마주보며 닫히는 게 아니라, 가동 턱이 위에서 아래로
  스쿱 위를 누르는 방식).
- 이 구조에 맞춰 물체를 스쿱 안에 놓고 가동 턱을 닫는 시나리오를 순기구학
  최적화(그리드 서치)로 여러 팔 자세를 찾아 스크립트로 자동 검증 시도(물체를
  집고 → 들어올려서 → 바닥에서 떨어졌는지로 성공 판정).
- 마찰/충돌 파라미터(`condim`, `solref`, `solimp`)를 표준적인 값으로 강화하고,
  가동 턱을 천천히 닫는(순간적으로 확 닫지 않는) 시나리오까지 시도했으나,
  여러 자세/파라미터 조합(수십 건)에서 스크립트 기준으로는 100% 성공을 재현하지
  못함 — 손으로 짠 근사 자세로 6D 그랩 플래닝을 정확히 맞추는 게 근본적으로
  어려운 문제였음.

**결정(사용자와 상의 후)**: 실제 사용 환경은 사람이 카메라 피드백을 보며
실시간으로 리더암을 조작하는 텔레오퍼레이션이므로, 제가 짠 고정된 스크립트
자세보다 사람이 시각/촉각 피드백으로 훨씬 잘 맞출 수 있다고 판단. 표준적인
물리 개선(물체 크기 축소, 마찰/접촉 파라미터 강화)과 실제 접시(충돌체 있는
평판) 추가까지만 반영하고, 최종 파지 성공 여부 검증은 사용자의 실제
텔레오퍼레이션 테스트로 넘기기로 함.

**적용한 변경**(`scene.xml`):
- `pick_object`: 한 변 2.5cm → 2cm로 축소, `friction="1.8 0.03 0.03"`,
  `condim="6"`, `solref="0.004 1"`, `solimp="0.95 0.99 0.001"`로 강화.
- `place_plate`: 기존의 충돌 없는 표시용 site를 제거하고, 실제로 물체를
  올려놓을 수 있는 충돌체 있는 얇은 원판(지름 9cm)을 추가.

**검증**: 물리 안정성 확인(수천 스텝 동안 물체/접시가 폭발하거나 NaN 없이
안정적으로 자리 잡는지), 전체 씬 오프스크린 렌더링으로 배치 확인. 자동
파지 성공 자체는 증명하지 못했음을 투명하게 공유.

---

### 문제 4: `import lerobot` 단계에서 xxhash DLL 로딩 실패로 스크립트 자체가 죽음

**증상**:
```
ImportError: DLL load failed while importing _xxhash: 애플리케이션 제어
정책에서 이 파일을 차단했습니다.
```
`from lerobot.datasets.lerobot_dataset import LeRobotDataset` 시점에 발생,
즉 텔레오퍼레이션 로직 실행 전에 스크립트가 아예 시작도 못 함.

**원인**: 문제 1에서 다룬 mujoco 플러그인 DLL 차단과 **동일한 원인**(Windows
애플리케이션 제어 정책이 특정 서명/해시의 DLL을 차단) — 이번엔 `lerobot`이
의존하는 HuggingFace `datasets` 패키지가 내부적으로 쓰는 `xxhash`의 컴파일된
확장 모듈(`_xxhash`)이 차단 대상이었음.

**해결**: `datasets` 소스(`fingerprint.py`)를 확인해 `xxhash.xxh64()`가
`.update(bytes)` / `.hexdigest()` 두 메서드만 쓰는 걸 확인. 이 해시는 로컬
캐시 fingerprint 생성에만 쓰이고(허깅페이스 허브와 동기화되는 값이 아니라 이
프로세스 안에서만 일관되면 됨) 이 프로젝트는 로컬 데이터셋만 다루므로, 실제
알고리즘이 xxhash와 달라도 문제없음 → `import xxhash`가 실패하면
`hashlib.blake2b` 기반의 호환 shim을 `sys.modules["xxhash"]`에 등록해 이후
import 체인이 계속 진행되게 함(`run_teleop_real.py`, `run_teleop_simul.py`
양쪽에 동일하게 적용).

**검증**: 동일 환경에서 재현되는 것을 직접 확인 후 수정, 이후 앞서 언급한
전체 데이터셋 생성→저장→재로드 스모크 테스트가 정상 통과하는 것으로 검증.

---

## 변경된 파일 요약

| 파일 | 주요 변경 |
|---|---|
| `so101_new_calib.xml` | `gripper` body에 `wrist_cam` 카메라 추가 |
| `scene.xml` | `pick_object` 크기/마찰/접촉 파라미터 조정, `place_plate`(접시) 추가 |
| `run_teleop_real.py` | key_callback/제어 루프 예외 처리 강화, xxhash 임포트 우회, 손목 카메라 렌더링·녹화(`observation.images.wrist_cam`)·실시간 미리보기 창(Tkinter) 추가 |
| `run_teleop_simul.py` | xxhash 임포트 우회만 동일 적용(카메라/미리보기 기능은 추가하지 않음) |
| `scene_teleop.xml` | `build_teleop_scene.py` 재실행으로 자동 재생성(카메라 반영) |
