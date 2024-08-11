# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /Users/beingrampopuri/Downloads/cell_segmentation/U-net-segmentation

# Copy the requirements file and install the dependencies
COPY requirements.txt .

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
