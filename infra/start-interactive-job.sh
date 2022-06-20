#! /bin/bash

salloc --gres=gpu:v100l:1 --time=0-1:0:0 --account=def-uofavis-ab --cpus-per-task=8 --mem-per-cpu=6GB