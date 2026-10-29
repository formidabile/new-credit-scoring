FROM python:3.12-slim

# LightGBM требует библиотеку OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-bot.txt .
RUN pip install --no-cache-dir -r requirements-bot.txt

COPY src ./src
COPY bot ./bot
COPY models/lgbm_bot.joblib models/features_bot.json models/decision_config_bot.json models/bot_input_ranges.json ./models/

# токен передаётся переменной окружения при запуске: docker run -e TELEGRAM_BOT_TOKEN=... образ
CMD ["python", "-m", "bot.main"]
