#!/bin/bash

python2 ../Python/phate/test.py || python ../Python/phate/test.py
python3 ../Python/phate/test.py || python ../Python/phate/test.py
python2 phate_examples.py || python phate_examples.py
python3 phate_examples.py || python phate_examples.py
Rscript phate_examples.R
