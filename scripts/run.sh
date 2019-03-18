#!/usr/bin/env bash

rm -rf "build"
rm -rf "dist"
rm -rf "clus.egg-info"
rm -rf "~/.local/lib/python3.7/site-packages/clus-0.0.1-py3.7.egg"
reset
python3 setup.py install --user
