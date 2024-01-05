FROM rust:1.74-alpine as builder

RUN apk add --no-cache bash
RUN apk add --no-cache g++ cmake make
RUN apk add --no-cache python3 py3-pip

WORKDIR /md_exp

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./algorithms ./algorithms
COPY ./common ./common
COPY ./hippodrome ./hippodrome
COPY ./playground ./playground
COPY Cargo.toml .
COPY ./scripts ./scripts

# RUN cargo build  --release --bin md

ENTRYPOINT tail -f /dev/null