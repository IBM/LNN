FROM python:3.9.10-slim-buster

WORKDIR /home/LNN

RUN apt-get update && apt-get install --assume-yes gcc wget python3-dev git graphviz graphviz-dev libgraphviz-dev pkg-config libgmp-dev
RUN /usr/local/bin/python -m pip install --upgrade pip

EXPOSE 8888

CMD ["./docker/docker-entrypoint.sh"]