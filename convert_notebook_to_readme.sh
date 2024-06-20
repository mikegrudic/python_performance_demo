#!/usr/bin bash
export PYDEVD_DISABLE_FILE_VALIDATION=1
jupyter nbconvert --execute --to markdown nbody_demo.ipynb
mv nbody_demo.md README.md