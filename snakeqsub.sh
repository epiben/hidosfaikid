#! /usr/bin/env bash

spec=("qsub -l nodes=1:ppn={threads},"
	  "mem={resources.vmem}mb,"
	  "walltime={resources.tmin}:00"
	  " -j eo -e ~/.qsub_logs/")

call=$(printf "%s" "${spec[@]}")

snakemake $@ -p --jobs 30 --notemp --verbose \
	--cluster "$call" --latency-wait 120
