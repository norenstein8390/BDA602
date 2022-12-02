FROM python:3.10.0

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY ./src/homework6/homework6.sql .
COPY ./src/homework6/main_bash.sh .
COPY baseball.sql .

# Run app (added executable priviledge)
RUN chmod +x ./main_bash.sh
CMD ./main_bash.sh
