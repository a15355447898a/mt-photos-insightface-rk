# Use the official Python 3.10 image.
FROM python:3.10-slim

# Set the working directory in the container.
WORKDIR /app

# Copy and install wheel dependencies first to leverage Docker layer caching.
COPY rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl ./
COPY inspireface-0.0.0-cp310-cp310-linux_aarch64.whl ./
RUN pip install --no-cache-dir rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl inspireface-0.0.0-cp310-cp310-linux_aarch64.whl

# Copy the rest of the application code.
COPY requirements.txt server_rknn.py ./

# Install other dependencies from requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the application.
CMD ["python", "server_rknn.py"]
