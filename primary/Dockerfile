FROM python:3.13.5-slim

WORKDIR /validation
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . ./

CMD ["python", "verify_certificate.py", "--recompute"]
