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
COPY ./src/final/final.sh .
COPY ./src/final/final.sql .
COPY ./src/final/make_constants_table.sql .
COPY ./src/homework5/final.py .
COPY ./src/homework5/homework4_main.py .
COPY ./src/homework5/homework4_plots.py .
COPY ./src/homework5/homework4_scores.py .
COPY ./src/homework5/midterm_bruteforce.py .
COPY ./src/homework5/midterm_main.py .
COPY baseball.sql .

# Run app (added executable priviledge)
RUN chmod +x ./final.sh
CMD ./final.sh
