FROM bitnami/minideb:latest

ARG OPENAI_API_KEY=apikey
ENV OPENAI_API_KEY=$OPENAI_API_KEY

RUN install_packages ca-certificates build-essential python3-full python3-pip \
	&& useradd -m -s /bin/bash appuser \
	&& mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /home/appuser/.cache \
	&& chown -R appuser:appuser /home/appuser \
	&& chmod -R 777 /home/appuser/.cache \
	&& chown -R appuser:appuser /app/data \
	&& chmod -R 777 /app/data \
	&& chown -R appuser:appuser /app/processed_data \
	&& chmod -R 777 /app/processed_data

USER appuser

EXPOSE 5052

CMD ["python3", "main.py"]

