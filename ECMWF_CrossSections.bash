#!/bin/bash

date=`date`
echo "STARTING AT "$date

lag=6
yyyy=`date -d '-'$lag' hours' -u +%Y`
mm=`date -d '-'$lag' hours' -u +%m`
dd=`date -d '-'$lag' hours' -u +%d`
hh=`date -d '-'$lag' hours' -u +%H`

fmm=`date -u -d $yyyy$mm$dd' '$hh' +120 hours' +%m`
fdd=`date -u -d $yyyy$mm$dd' '$hh' +120 hours' +%d`
fhh=`date -u -d $yyyy$mm$dd' '$hh' +120 hours' +%H`

filename="/data/downloaded/Forecasts/ECMWF/NRT_data/"$yyyy$mm$dd$hh"/S2D"$mm$dd$hh"00"$fmm$fdd$fhh"001.grb"
echo $filename
 while true; do
 if [[ -e "$filename" ]]; then         # Check if the file exists
  if ! lsof "$filename" > /dev/null 2>&1; then   # Check if the file is being written to using lsof
   break                    # Exit the loop for this file
  else
   echo $yyyy$mm$dd$hh "F-120 is being written."
   sleep 5
  fi
 else
  echo $yyyy$mm$dd$hh "F-120 does not exist."
  sleep 30
 fi
done

echo $yyyy$mm$dd$hh "F120 ready for processing"
date=`date`
echo "STARTING PLOTS AT "$date


cd /data/projects/operations/ivt_cross_sections

/bin/apptainer exec --bind /data:/data,/home:/home,/work:/work,/common:/common -e /data/projects/operations/ivt_cross_sections/envs/ivt_cross_section.sif /opt/conda/envs/container/bin/python /data/projects/operations/ivt_cross_sections/run_tool_parallel.py "ECMWF" > /data/projects/operations/ivt_cross_sections/ECMWF_CrossSections.log 

try=1
while [ $try -le 5 ]; do
 timeout 240 rsync --ignore-missing-args -avih /data/projects/operations/ivt_cross_sections/figs/ECMWF/*.png /data/projects/website/mirror/htdocs/images/ECMWF/Cross_Sections/new/
 if [ $? -eq 0 ]; then
  break
 fi
 attempt=$((attempt + 1))
 sleep 5
done

date=`date`
echo "FINISHED AT "$date




exit
