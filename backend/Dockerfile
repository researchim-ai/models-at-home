# Backend container — extends the base models-at-home image with FastAPI.
# Electron starts this final image, not the raw Streamlit/ML base image.
FROM models-at-home-base:latest

ENV MODELS_AT_HOME_ROOT=/app

# Install the API and supervisor dependencies used by this source tree.
COPY backend/requirements.txt /tmp/mah-requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r /tmp/mah-requirements.txt

# Copy backend API code
COPY backend/ /app/backend/
COPY examples/ /app/examples/
COPY requirements-world-foundation.txt /app/requirements-world-foundation.txt
COPY requirements-world-local.txt /app/requirements-world-local.txt

EXPOSE 8000 8888

CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
