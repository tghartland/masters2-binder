#! /bin/bash -f

# args
# 1 - Cluster
# 2 - Cluster.Process
# 3 - Mass
# 4 - Process
# 5 - Description
# 6 - Simulated data file name
# 7 - Simulated data histogram name

echo "Job $2"
export DISPLAY="localhost0.0"

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

WORKDIR=/home/atlas/thartland/masters/week17

cd ${WORKDIR}

lsetup root

source /home/atlas/thartland/venvs/root-py2/bin/activate

sleeptime=$(($4*10 - 60*($4/15)))
# sleeptime=$((($4%15 + 4*$4/15)*10))
echo "Sleeping for $sleeptime"
sleep "$sleeptime"
echo "Resuming"

mkdir "results/$1"

if [ "$4" -eq "0" ]; then
    echo "37" >> "results/$1/fb.txt"
    echo "$5" >> "results/$1/desc.txt"
    echo "$6" >> "results/$1/desc.txt"
    echo "$7" >> "results/$1/desc.txt"
fi

python limit_dist_37fb.py "$1" "$2" "$3" "$6" "$7"
