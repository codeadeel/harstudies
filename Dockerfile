# This file is responsible for building the one image that runs JupyterLab for every study
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Install The System Libraries ( libgomp1 for lightgbm, git for the ActRecTut download, procps so joblib can stop its worker processes after an error )
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install The Pinned Python Packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Start JupyterLab On Port 8888, The Repository Folder Is Mounted At /app By Docker Compose
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=", "--ServerApp.root_dir=/app"]
