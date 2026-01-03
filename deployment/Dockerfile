# -------------------------------------------------
# Base image: official SageMaker inference base
# -------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------
# Environment variables required by SageMaker
# -------------------------------------------------
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# -------------------------------------------------
# Set working directory
# -------------------------------------------------
WORKDIR /opt/program

# -------------------------------------------------
# Copy inference code
# -------------------------------------------------
COPY inference.py /opt/program/inference.py

# -------------------------------------------------
# Copy requirements
# -------------------------------------------------
COPY requirements.txt /opt/program/requirements.txt

# -------------------------------------------------
# Install dependencies
# -------------------------------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=600 -r requirements.txt


# -------------------------------------------------
# Expose SageMaker default port
# -------------------------------------------------
EXPOSE 8080

# -------------------------------------------------
# SageMaker entry point
# -------------------------------------------------
ENTRYPOINT ["python"]
CMD ["-m", "sagemaker_inference"]
