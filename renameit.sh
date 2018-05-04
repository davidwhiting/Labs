#!/bin/bash
git mv ../${1} ../${2}-${1}
git mv ${1}.py ${2}-${1}.py
git mv ${1}_problems.tex ${2}-${1}_problems.tex
