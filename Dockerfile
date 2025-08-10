# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.lock requirements.lock
RUN pip install --no-cache-dir --upgrade uv uvicorn
RUN uv venv .venv
RUN uv pip sync requirements.lock

COPY --chown=user . /app
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
