#!/bin/sh
echo To run tests by module name, use:
echo $0 -k 'module_name'
PYTHONPATH=./lnn:$PYTHONPATH pytest -s $*
