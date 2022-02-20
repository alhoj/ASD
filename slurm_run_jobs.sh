#!/bin/sh

chmod 755 /autofs/cluster/transcend/jussi/jobs/*

# This is the part where we submit the jobs that we cooked

for j in $(ls -1 "/autofs/cluster/transcend/jussi/jobs/");do
sbatch "/autofs/cluster/transcend/jussi/jobs/"$j
sleep 0.01
done
echo "All jobs submitted!"

