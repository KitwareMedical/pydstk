#!/bin/bash

#-------------------------------------------------------------------------------
PYTHON="/opt/local/bin/python2.7"
#-------------------------------------------------------------------------------

if [ $# -lt 4 ] ; then
      echo "Usage: ./gendb.sh <VideoDir> <ModelDir> <DSType [dt|kdt]> <#States>"  
      exit 0
fi

VIDEOS=$1
MODELS=$2
DSTYPE=$3
STATES=$4

for file in `find ${VIDEOS} -name 'ks*.avi'`; do
  BASENAME=`basename ${file} .avi`
  case ${DSTYPE} in
   "dt"  )  
     CMD="${PYTHON} dt.py -i ${file} -n ${STATES} -t vFile -e -o ${MODELS}/${BASENAME}.pkl";;
   "kdt" )  
     CMD="${PYTHON} kdt.py -i ${file} -n ${STATES} -t vFile -o ${MODELS}/${BASENAME}.pkl";;
  esac
  echo $CMD
  ${CMD}
done
