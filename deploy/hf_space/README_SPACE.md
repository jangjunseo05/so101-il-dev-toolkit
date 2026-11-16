---
title: SO-101 Imitation Learning Dashboard (Demo)
emoji: 🦾
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Teleop data → QA → ACT training → MuJoCo inference
---

# SO-101 모방학습 파이프라인 대시보드 — 체험용 데모

'데이터 수집 – QA – 학습 – 추론' 4단계를 하나의 웹 대시보드에서 실행하는 모방학습 개발 키트의 공개 데모입니다.

- 이 데모는 실제 리더암·GPU와 연결되어 있지 않습니다. 녹화·학습·시뮬레이션 실행 버튼은 잠겨 있고,
  실제로 수집·학습·평가한 결과를 보여 줍니다.
- ② QA 검증(아웃라이어·무결성 검사)은 이 서버에서 실제로 계산됩니다.
- 코드: https://github.com/jangjunseo05/so101-il-dev-toolkit
