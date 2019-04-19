import os
import re
import numpy as np
import math
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFont
import argparse
import gpxpy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return (xtile, ytile)


class Tiles:
    def __init__(self, path, zoom, tile_size='256x256'):
        self.path = path
        self.zoom = zoom
        self.tile_px = [int(px) for px in tile_size.split('x')]
        self.tiles = self.get_tiles(os.path.join(path, str(zoom)))
        self.tile_array, self.tile_range = self.gen_tile_array(self.tiles)

    def get_tiles(self, path):
        tiles = []
        for files in os.walk(path):
            if len(files[2]) == 0:
                continue
            x = re.search(r'/([0-9]+)$', files[0])[1]
            for f in files[2]:
                y = re.search('[0-9]+', f)[0]
                tiles.append((int(x), int(y)))
        print("Found %s tiles." % len(tiles))
        return tiles

    def gen_tile_array(self, tiles):
        x = [xy[0] for xy in tiles]
        y = [xy[1] for xy in tiles]
        tile_range = {
            'x': [min(x), max(x)],
            'y': [min(y), max(y)],
            'dx': max(x) - min(x),
            'dy': max(y) - min(y)
        }
        print("X: %s - %s" % (tile_range['x'][0], tile_range['x'][1]))
        print("Y: %s - %s" % (tile_range['y'][0], tile_range['y'][1]))

        tile_array = np.zeros((tile_range['y'][1] - tile_range['y'][0] + 1, tile_range['x'][1] - tile_range['x'][0] + 1))
        for c in tiles:
            tile_array[c[1] - tile_range['y'][0], c[0] - tile_range['x'][0]] = 1
        return tile_array, tile_range

    def import_gpx(self, gpx):
        lat = []
        lon = []
        gpx_track = gpxpy.parse(open(gpx, 'r'))
        for track in gpx_track.tracks:
            for segment in track.segments:
                for point in segment.points:
                    x, y = deg2num(point.latitude, point.longitude, zoom=self.zoom)
                    x -= self.tile_range['x'][0]
                    y -= self.tile_range['y'][0]
                    # Add middle points in case abs diff >= 1
                    if len(lat) > 1:
                        dx = x - lon[-1]
                        dy = y - lat[-1]
                        if abs(dx) >= 1 or abs(dy) >= 1:
                            nx = int(abs(dx))
                            ny = int(abs(dy))
                            n = max(nx, ny)
                            for _ in range(n):
                                lon.append(lon[-1] + dx / (n+1))
                                lat.append(lat[-1] + dy / (n+1))
                    lon.append(x)
                    lat.append(y)
        return (lon, lat)

    def generate_maps(self, rectangles, path, track=None, prefix='stitch', water=False, gray=False):
        # Import track
        lat = []
        lon = []
        if track is not None:
            (lon, lat) = track
        # save Image
        for idx, r in enumerate(rectangles):
            # Create canvas
            img = r.stitch_array(self)
            # Add page number
            font = ImageFont.truetype("DejaVuSans.ttf", 48)
            ImageDraw.Draw(img).text((10, 10), str(idx), (0, 0, 0), font=font)
            # Add scale
            r.add_scale(img, self)
            # Add track
            if track is not None:
                cut_track = r.cut_path(lon, lat)
                if len(cut_track) > 1:
                    print("Track length: %s segments" % len(cut_track))
                    draw = ImageDraw.Draw(img)
                    draw.line(cut_track, fill=(0, 0, 0), width=2)
                    del draw
            img_file = os.path.join(path, '{}_{}.png'.format(prefix, idx))
            print("Saving ", img_file)
            # Remove water; color as white
            if not water:
                # Replace water with white colour
                data = np.array(img)   # "data" is a height x width x 3 numpy array
                red, green, blue = data.T # Temporarily unpack the bands for readability
                # Replace colors
                water_px = (red == 170) & (green == 211) & (blue == 223) 
                data[:, :][water_px.T] = (255, 255, 255) # Transpose back needed
                img = Image.fromarray(data)
            # Enhance contrast and convert to grayscale
            if gray:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.25).convert('L')
            # Save as PNG
            img.save(img_file, 'PNG')


class Box:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.x1 = x + dx
        self.y1 = y + dy

    def add_plot(self, ax):
        rectangle = Rectangle((self.x, self.y), self.dx, self.dy, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rectangle)

    def stitch_array(self, tiles):
        px_size = tiles.tile_px
        x_size = px_size[0] * math.ceil(self.dx)
        y_size = px_size[1] * math.ceil(self.dy)
        # Create an empty image
        result = Image.new('RGB', (x_size, y_size), color=(255, 255, 255))
        for x in range(int(self.x), int(self.x1)):
            for y in range(int(self.y), int(self.y1)):
                if tiles.tile_array[y, x] == 0:
                    continue
                x_tile = x + tiles.tile_range['x'][0]
                y_tile = y + tiles.tile_range['y'][0]
                tile_file = os.path.join(tiles.path, str(tiles.zoom), str(x_tile), str(y_tile) + '.png') 
                # print('Loading ', tile_file)
                tile_image = Image.open(tile_file)
                result.paste(im=tile_image, box=((x-self.x) * px_size[0], (y-self.y) * px_size[1]))
        crop = False
        for dim in [self.x, self.y, self.dx, self.dy]:
            crop |= (int(dim) - dim) != 0
        if crop:
            x0 = int((self.x - int(self.x)) * px_size[0])
            y0 = int((self.y - int(self.y)) * px_size[1])
            x1 = int((self.x1 - int(self.x1)) * px_size[0])
            y1 = int((self.y1 - int(self.y1)) * px_size[1])
            result = result.crop((x0, y0, x1, y1))
        return result 

    def cut_path(self, lon, lat, size='256x256'):
        px_size = [int(px) for px in size.split('x')]
        result = []
        for coord in zip(lon, lat):
            if coord[0] >= self.x and coord[0] <= self.x1 and coord[1] >= self.y and coord[1] <= self.y1:
                result.append(((coord[0] - self.x) * px_size[0], (coord[1] - self.y) * px_size[1]))
        return result

    def add_scale(self, img, tiles, pos=(0.03, 0.97)):
        px_size = tiles.tile_px
        x = pos[0] * self.dx * px_size[0]
        y = pos[1] * self.dy * px_size[1]
        # Get lat for px y coordinate
        # From: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Tile_numbers_to_lon..2Flat._2
        n = 2.0 ** tiles.zoom
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (tiles.tile_range['y'][0] + self.y + pos[1] * self.dy) / n)))
        # Get m / px 
        # https://wiki.openstreetmap.org/wiki/Zoom_levels
        C = 2 * math.pi * 6378137.0
        m_per_px = C * math.cos(lat_rad) / 2 ** (tiles.zoom + math.log2(px_size[0]))
        km1 = 1000 / m_per_px
        km5 = 5000 / m_per_px
        km10 = 10000 / m_per_px
        # Draw lines
        draw = ImageDraw.Draw(img)
        # Horiz 10 km line
        draw.line((x, y) + (x + km10, y), fill=(0, 0, 0), width=2)
        # Add ticks
        draw.line((x + km5, y) + (x + km5, y-5), fill=(0, 0, 0), width=2)
        draw.line((x + km1, y-5) + (x + km1, y+5), fill=(0, 0, 0), width=2)
        draw.line((x, y-7) + (x, y+7), fill=(0, 0, 0), width=2)
        # Add annotations
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        draw.text(((x, y + 10) ), "0 km", (0, 0, 0), font=font)
        draw.text(((x + km1, y + 10) ), "1 km", (0, 0, 0), font=font)
        draw.text(((x + km5, y + 10) ), "5 km", (0, 0, 0), font=font)
        draw.text(((x + km10, y + 10) ), "10 km", (0, 0, 0), font=font)
        del draw



def gen_rects(tiles, dx=10, dy=10, minpx=0):
    box_list = []
    (size_y, size_x) = tiles.tile_array.shape
    for n_x in range(int(np.ceil(size_x/dx))):
        for n_y in range(int(np.ceil(size_y/dy))):
            d_x = min((n_x+1)*dx, size_x) - n_x * dx
            d_y = min((n_y+1)*dy, size_y) - n_y * dy
            box = Box(n_x*dx, n_y*dy, d_x, d_y)
            subarr = tiles.tile_array[box.y:box.y1, box.x:box.x1]
            if subarr.sum((0, 1)) > minpx:
                box_list.append(box)
    return box_list

def gen_rects_from_track(track, dx=10, dy=10, border=1):
    # Range calculating class
    class Range:
        @classmethod
        def reset(cls):
            cls.range = {
                'x': [None, None],
                'y': [None, None],
                'dx': 0,
                'dy': 0
            }
            cls.n = 0
        @classmethod
        def get_extent(cls, point):
            if cls.n == 0:
                cls.range['x'] = [point[0], point[0]]
                cls.range['y'] = [point[1], point[1]]
            else:
                min_x = int(min(cls.range['x'][0], point[0]))
                min_y = int(min(cls.range['y'][0], point[1]))
                max_x = int(max(cls.range['x'][1], point[0]))
                max_y = int(max(cls.range['y'][1], point[1]))
                cls.range = {
                    'x': [min_x, max_x],
                    'y': [min_y, max_y],
                    'dx': max_x - min_x + 1,
                    'dy': max_y - min_y + 1
                }
            cls.n += 1
            return cls.range
    # Initialize Range with empty range
    Range.reset()
    number_points = len(track[0])
    boxes = []
    prev_P = False
    prev_extent = {}
    for idx, point in enumerate(zip(track[0], track[1])):
        extent = Range.get_extent(point)
        # TODO: Allow for float coords in boxes
        P = (extent['dx'] <= (dx - border)) and (extent['dy'] <= (dy - border))
        L = (extent['dx'] <= (dy - border)) and (extent['dy'] <= (dx - border))
        # If neither Portrait nor Landscape view fits, create a rectangle 
        # from the previous fitting view
        if len(prev_extent) > 0 and (((P or L) == False) or idx == number_points - 1):
            if prev_P:
                rot_dx = dx
                rot_dy = dy
                x = prev_extent['x'][0] - (rot_dx - prev_extent['dx']) / 2
                y = prev_extent['y'][0]
            else:
                rot_dx = dy
                rot_dy = dx
                x = prev_extent['x'][0]
                y = prev_extent['y'][0] - (rot_dy - prev_extent['dy']) / 2
            boxes.append(Box(int(x), int(y), rot_dx, rot_dy))
            prev_P = False
            prev_extent = {}
            Range.reset()
            continue
        prev_extent = extent
        prev_P = P
    return boxes


if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description='Create printable maps from downloaded OSM tiles.')
    parser.add_argument("tile_path", help=r"Directory with OSM PNG tiles: /{zoom}/{x}/{y}.png")
    parser.add_argument(
        '-z',
        action="store",
        dest="zoom",
        type=int,
        default=13,
        help="OSM zoom level"
    )
    parser.add_argument(
        '--gpx',
        action="store",
        dest="gpx",
        type=str,
        help="GPX trace to produce map",
        required=True
    )
    parser.add_argument(
        '-o',
        action="store",
        dest="output_dir",
        type=str,
        default=".",
        help="output directory"
    )
    parser.add_argument(
        '-p',
        action="store",
        dest="map_prefix",
        type=str,
        default="stitch",
        help="output map prefix, 'stitch' by default"
    )
    parser.add_argument(
        '--gray',
        action="store_true",
        dest="gray",
        default=False,
        help="output as grayscale"
    )
    parser.add_argument(
        '--water',
        action="store_true",
        dest="water",
        default=False,
        help="include water; removes water color (170, 211, 223) by default"
    )
    parser.add_argument(
        '-x',
        action="store",
        dest="nx",
        type=int,
        default=8,
        help="number of tiles in X dimension to load per chart; 8 by default"
    )
    parser.add_argument(
        '-y',
        action="store",
        dest="ny",
        type=int,
        default=11,
        help="number of tiles in Y dimension to load per chart; 11 by default"
    )
    args = parser.parse_args()

    # Start plotting
    # Generate tile canvas
    tiles = Tiles(args.tile_path, zoom=args.zoom)

    # Get rectangles
    gpx_trace = tiles.import_gpx(args.gpx)
    # TODO: Create maps with all tiles if no GPX is given
    #rects = gen_rects(tile_array, dx=11, dy=15, minpx=1)
    rects = gen_rects_from_track(gpx_trace, dx=args.nx, dy=args.ny)
    print("Number of charts: ", len(rects))

    # Plot legend with rectangles and track
    # Import track
    lat = []
    lon = []
    if gpx_trace is not None:
        (lon, lat) = gpx_trace
    # Get the current reference
    plt.figure(figsize=(10,10)) 
    plt.matshow(tiles.tile_array)
    ax = plt.gca()
    # Create a Rectangle patch for each rectangle
    for idx, r in enumerate(rects):
        r.add_plot(ax)
        # Add rectangle number
        plt.text(r.x + r.dx * 0.25, r.y + r.dy * 0.75, str(idx), fontsize=9, color='white')
    plt.plot(lon, lat)
    plt.savefig(os.path.join(args.output_dir, 'legend.png'))

    # Save separate maps
    tiles.generate_maps(rects, track=gpx_trace, path=args.output_dir, gray=args.gray, water=args.water)
