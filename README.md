# Swedish trip
This is a small project that I developed to print [OpenStreetMap](https://www.openstreetmap.org) maps
for a bike trip around Sweden, but it should work for general GPX tracks as well. 
The script selects and stitches offlie tiles from OSM along a GPX track for print.
It automatically creates overlaping maps with a selected size in OSM tiles.

I plan to travel around a month. The route is avaliable [here](https://www.bikemap.net/en/r/4733311/).

## Usage
### Getting GPX tracks
There are many different cycling track design websites avaliable. The three that I've been using and that I can reccomend are:
* [GPSies](https://www.gpsies.com)
* [BikeMap](https://www.bikemap.net/)
* [cycle travel](https://cycle.travel/map)

All allow to create routes by simply selecting start and beginging. Routes can be adjusted by moving 
each part and recalculating automatically the new route. Personally I find the third one the fastest and 
most robust. Each website allows for GPX track exporting.

### Getting OSM tiles
To download OSM tiles to have them offline I used a 2.7 Python GUI application [GMapCatcher](https://github.com/heldersepu/GMapCatcher).
The program has the possibility to download tiles alongside a GPX track. Then I used the GUI to download the missing tiles for
corners of the future map. The command to download the tiles is the following:

```
python2 download.py --gpx=/home/mateusz/projects/gis/sweden_trip/cycle_travel.gpx --min-zoom=5 --max-zoom=5 --width=5
```

The script allows to download missing (all) tiles from a tile server after specifying the PNG tiles path.
In case of OSM this will be `https://a.tile.openstreetmap.org/{z}/{x}/{y}.png`. More tile servers are avaliable on the
[OpenStreetMap Wiki](https://wiki.openstreetmap.org/wiki/Tile_servers). Remember to remove the `$` sign from the URL.

**NOTE:** Depending on the zoom level and length of the track the number of tiles can easily go into thousands and millions.
OSM is run non-commercially by contributors, so caution and understanding is required when downloading many tiles simultaneously.
Detailed instructions are provided in the [Tile usage policy](https://wiki.openstreetmap.org/wiki/Tile_usage_policy).

Tiles are downloaded into `GMapCatcher` folder which is by default `~/.GMapCatcher`. The tiles also need to be exported using 
the GUI to OSM tiles that which I'll be using as tile source for my script.

### Prerequisites
I wrote the script under Python 3.7. I used the following dependencies: requests, Numpy, Matplotlib, Pillow and [gpxpy](https://github.com/tkrajina/gpxpy).
All the modules should be available through `pip`.

Details on how the script works can be found on [my personal page](https://maten.pl/post/sweden_trip/).

### Syntax
Usage: 
```
usage: print.py [-h] [-z ZOOM] --gpx GPX [-o OUTPUT_DIR] [-p MAP_PREFIX]
                [--gray] [--water WATER WATER WATER] [-x NX] [-y NY]
                [--dl TILE_DL]
                tile_path
```
positional arguments:
* `tile_path` Directory with OSM PNG tiles: ./{zoom}/{x}/{y}.png

optional arguments:
* `-h, --help` show this help message and exit
* `-z ZOOM` OSM zoom level
* `--gpx GPX` GPX trace to produce map
* `-o OUTPUT_DIR` output directory
* `-p MAP_PREFIX` output map prefix, 'stitch' by default
* `--gray` output as grayscale
* `--water R G B` removes water color given as R G B byte values; e.g. `--water 170 211 223` for OSM
* `-x NX` number of tiles in X dimension to load per chart; 8 by default
* `-y NY` number of tiles in Y dimension to load per chart; 11 by default
* `--dl TILE_DL` URL for downloading missing tiles, e.g.: `--dl https://a.tile.openstreetmap.org/{z}/{x}/{y}.png` for OSM

I also added a small script to merge and rotate generated maps into one TIFF 
or PDF document, because my local print shop does not print double-sided pages
from separate images...

## License

This project is provided as it is. You use it on your own responsibility.
This project is licensed under the MIT License - see the LICENSE.md file for details.
