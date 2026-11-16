# SO-101 모방학습 대시보드 -- 공개 체험 데모 (Hugging Face Spaces, Docker SDK, CPU)
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# SO101_DEMO=1: 장비가 필요한 버튼을 잠그고 demo_assets/의 실제 결과를 보여 준다 (dashboard/lib/demo_mode.py)
ENV SO101_DEMO=1 \
    PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 \
    HOME=/home/user

EXPOSE 7860
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
