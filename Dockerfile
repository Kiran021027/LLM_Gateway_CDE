# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and configuration into the container
COPY ./src ./src
COPY config.yaml .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables for API keys (placeholders)
# In a real deployment, these would be injected securely (e.g., via Kubernetes Secrets)
# ENV OPENAI_API_KEY=""
# ENV ANTHROPIC_API_KEY=""

# Run the application when the container launches
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
