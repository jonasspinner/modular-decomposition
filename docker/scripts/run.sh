#!/usr/bin/env bash

docker run --name md_exp --rm\
       --user "$(id -u):$(id -g)"\
       -v $PWD/data:/md_exp/data\
       md_exp &