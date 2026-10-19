FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends nodejs fonts-noto-cjk \
    && pip install --no-cache-dir pillow numpy pandas matplotlib openpyxl playwright==1.58.0 \
    && PLAYWRIGHT_BROWSERS_PATH=/opt/browsers python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

ENV PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
CMD ["sleep", "infinity"]
