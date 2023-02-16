#!/bin/sh

cd "$(dirname "$0")"

pip3 install -r requirements.txt
make github-pages

