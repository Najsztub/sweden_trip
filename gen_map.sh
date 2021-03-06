#!/bin/bash

gen=false
merge=false

for i in "$@"
do
case $i in
    -g|--gen)
    gen=true
    shift
    ;;
    -m|--merge)
    merge=true
    shift 
    ;;
    *)
    ;;
esac
done

if [ $gen = true ]
then
    ~/.venv/gis_env/bin/python print.py ./OSM_tiles -z 13 --gray --gpx ./routes/cycle_travel2.gpx -o ./OSM_map --water 170 211 223
fi
if [ $merge = true ]
then
    ~/.venv/gis_env/bin/python merge_png.py ./OSM_map -o ./OSM_map/out.tiff
fi