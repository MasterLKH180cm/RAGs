# syntax=docker/dockerfile:1
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAGS_ENV=production \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
ARG RAGS_EXTRAS=postgres,pdf,mcp,vision
RUN --mount=type=cache,target=/root/.cache/uv \
    set --; for extra in $(printf '%s' "$RAGS_EXTRAS" | tr ',' ' '); do set -- "$@" --extra "$extra"; done; \
    uv sync --locked --no-dev --no-install-project "$@"
COPY README.md ./
COPY rags ./rags
RUN --mount=type=cache,target=/root/.cache/uv \
    set --; for extra in $(printf '%s' "$RAGS_EXTRAS" | tr ',' ' '); do set -- "$@" --extra "$extra"; done; \
    uv sync --locked --no-dev --no-editable "$@" \
    && useradd --uid 10001 --create-home appuser \
    && mkdir -p /app/data && chown appuser:appuser /app/data
USER appuser
EXPOSE 8000
CMD ["uvicorn", "rags.api:app", "--host", "0.0.0.0", "--port", "8000"]
