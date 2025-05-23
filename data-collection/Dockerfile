FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* is only to reduce image size

RUN type -p curl >/dev/null || (apt update && apt install curl -y)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

WORKDIR /app

COPY fetch.sh .
COPY fetch.py .

RUN chmod +x fetch.sh

RUN pip install requests

ENTRYPOINT ["/app/fetch.sh"]

# By default, collect 1000 repositories, between 50 and 2000 stars, 2 repositories at each step
CMD [ "1000", "50", "2000", "2" ]