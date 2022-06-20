#! /bin/bash

#load python 3.9
module load python/3.9

#activate virtual environment
cd && source projects/def-uofavis-ab/jkiechle/CoMA/env/bin/activate

#start mlflow ui
mlflow ui -h 0.0.0.0 -p 5000 &
