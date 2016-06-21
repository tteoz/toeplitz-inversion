#!/bin/bash

for SIZE in 12 14 16 18 20 22 24
do
   for NPROC in 3 6 12 24 48 96
   do
      touch "run_${NPROC}_${SIZE}.job"

      echo "#!/bin/bash
#@ job_type = parallel
#@ initialdir = /home/gruppo1/git/polydiv
#@ input = /dev/null

#@ output = /home/gruppo1/git/polydiv/results/out_NPROC-${NPROC}_SIZE-${SIZE}.txt
#@ error = /home/gruppo1/git/polydiv/results/err_NPROC-${NPROC}_SIZE-${SIZE}.txt
#@ notification = error
#@ class = medium 
#@ blocking = UNLIMITED
#@ total_tasks = ${NPROC} 
#@ queue

./src/polydiv ${SIZE} 

#---ENDS HERE---" > run_${NPROC}_${SIZE}.job


   done
done

