# One image for the harness, the API and the dashboard.
#
# They share every dependency except Streamlit, and maintaining three images to
# save ~40 MB would mean three places for a version to drift. The command
# differs per service in docker-compose.yml; the image does not.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /work

# Dependencies first so editing source does not re-resolve the whole tree.
COPY pyproject.toml /work/pyproject.toml
COPY src/toolsmith/__init__.py /work/src/toolsmith/__init__.py

RUN pip install --no-cache-dir -e ".[dev]"

COPY . /work

# The source tree is bind-mounted over this at run time, which is what makes
# an edit visible without a rebuild. The copy above exists so the image is
# usable on its own.
ENV PYTHONPATH=/work/src

CMD ["python", "-c", "import toolsmith; print(toolsmith.__version__)"]
