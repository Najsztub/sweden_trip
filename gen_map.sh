#!/bin/bash

gen=false
merge=false

for i in "$@"
do
case $i in
    -g|--gen)
    gen=true
    shift # past argument=value
    ;;
    -m|--merge)
    merge=true
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [ $gen = true ]
then
    ~/.venv/gis_env/bin/python print.py ./OSM_tiles -z 13 --gray --gpx ./routes/cycle_travel2.gpx -o ./OSM_map
fi
if [ $merge = true ]
then
    ~/.venv/gis_env/bin/python merge_png.py ./OSM_map -o ./OSM_map/out.tiff
fi