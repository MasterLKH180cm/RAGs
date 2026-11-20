FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAGS_ENV=production
WORKDIR /app
COPY pyproject.toml README.md ./
COPY rags ./rags
RUN pip install --no-cache-dir ".[pdf,mcp]" \
    && useradd --uid 10001 --create-home appuser \
    && mkdir -p /app/data && chown appuser:appuser /app/data
USER appuser
EXPOSE 8000
CMD ["uvicorn", "rags.api:app", "--host", "0.0.0.0", "--port", "8000"]
