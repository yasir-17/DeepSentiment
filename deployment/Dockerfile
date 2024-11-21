# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY src /app


# Expose the port Gradio will use
EXPOSE 7860
EXPOSE 8000

# Command to run your Gradio app
CMD ["python", "main.py"]
