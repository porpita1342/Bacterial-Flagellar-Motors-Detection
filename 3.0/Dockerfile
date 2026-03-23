FROM nvcr.io/nvidia/pytorch:25.01-py3

# 2. Set the internal working directory
WORKDIR /workspace

# 3. Copy only the requirements file (the recipe) to utilize build caching
# If you haven't made requirements.txt yet, see the note below
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
