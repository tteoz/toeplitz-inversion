#!/bin/bash

for SIZE in 18 20 22
do
   for NPROC in 3 6 12 24 48 96
   do
      llsubmit "run_${NPROC}_${SIZE}.job"
   done
done

