#! /bin/bash

salloc --gres=gpu:1 --time=0-1:0:0 --account=def-uofavis-ab --cpus-per-task=8 --mem-per-cpu=6GB
