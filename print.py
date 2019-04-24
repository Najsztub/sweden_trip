import os
import re
import numpy as np
import math
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFont
import argparse
import gpxpy
import utils

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Tiles:
    """Class for tiles"""
    def __init__(self, path, zoom, tile_size='256x256', web_source='', overwrite=False):
        self.path = path
        self.zoom = zoom
        self.downloader = None
        if not os.path.exists(path):
            raise NotADirectoryError("Path %s does not exist!" % path)
        if web_source != '':
            self.downloader = utils.TileSource(source=web_source, destination=path, overwrite=overwrite)
        self.tile_px = [int(px) for px in tile_size.split('x')]
        self.tiles = self.get_tiles(os.path.join(path, str(zoom)))
        self.tile_array, self.tile_range = self.gen_tile_array(self.tiles)

    def get_tiles(self, path):
        """Read avaliable tiles into a set"""
        tiles = set({})
        for files in os.walk(path):
            if len(files[2]) == 0:
                continue
            x = re.search(r'/([0-9]+)$', files[0])[1]
            for f in files[2]:
                y = re.search('[0-9]+', f)[0]
                tiles.add((int(x), int(y)))
        print("Found %s tiles." % len(tiles))
        return tiles

    def update(self):
        """Read again tiles from the given path"""
        self.tiles = self.get_tiles(os.path.join(self.path, str(self.zoom)))

    def update_array(self, pt1, pt2):
        """Update the tile array and extent with new extent from 2 points"""
        x = [xy[0] for xy in [pt1, pt2]]
        y = [xy[1] for xy in [pt1, pt2]]
        self.tile_range = {
            'x': [min(x), max(x)],
            'y': [min(y), max(y)],
            'dx': max(x) - min(x),
            'dy': max(y) - min(y)
        }
        self.tile_array = np.zeros((self.tile_range['dy'], self.tile_range['dx']))
        for c in self.tiles:
            if c[1] - self.tile_range['y'][0] < 0 or c[0] - self.tile_range['x'][0] < 0:
                continue
            if c[1] >= self.tile_range['y'][1] or c[0] >= self.tile_range['x'][1]:
                continue
            self.tile_array[c[1] - self.tile_range['y'][0], c[0] - self.tile_range['x'][0]] = 1


    def gen_tile_array(self, tiles):
        """Generate array with avaliable tiles for legend"""
        if len(tiles) == 0:
            tile_array = np.zeros((1,1))
            tile_range = {'x': [0,0], 'y':[0,0], 'dx': 0, 'dy': 0}
            return tile_array, tile_range
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
        """Import GPX track"""
        def add_middle_points():
            """Add middle points for sections skipping tiles"""
            dx = x - lon[-1]
            dy = y - lat[-1]
            if abs(dx) >= 1 or abs(dy) >= 1:
                nx = int(abs(dx))
                ny = int(abs(dy))
                n = max(nx, ny)
                for _ in range(n):
                    lon.append(lon[-1] + dx / (n+1))
                    lat.append(lat[-1] + dy / (n+1))

        lat = []
        lon = []
        gpx_track = gpxpy.parse(open(gpx, 'r'))
        for track in gpx_track.tracks:
            for segment in track.segments:
                for point in segment.points:
                    x, y = utils.deg2num(point.latitude, point.longitude, zoom=self.zoom)
                    # Add middle points in case abs diff >= 1
                    if len(lat) > 1:
                        add_middle_points()
                    lon.append(x)
                    lat.append(y)
        return (lon, lat)

    def generate_maps(self, rectangles, path, track=None, prefix='stitch', water=[None, None, None], gray=False):
        """Generate maps from a list of Boxes."""
        # Import track
        lat = []
        lon = []
        if track is not None:
            (lon, lat) = track
        # save Image for each Box
        for idx, r in enumerate(rectangles):
            img_file = os.path.join(path, '{}_{}.png'.format(prefix, idx))
            print("Saving %s/%s: %s" % (idx+1, len(rectangles), img_file))
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
            # Remove water; color as white
            if water[0] is not None:
                # Replace water with white colour
                data = np.array(img)   # "data" is a height x width x 3 numpy array
                red, green, blue = data.T # Temporarily unpack the bands for readability
                # Replace colors
                water_px = (red == water[0]) & (green == water[1]) & (blue == water[2]) 
                data[:, :][water_px.T] = (255, 255, 255) # Transpose back needed
                img = Image.fromarray(data)
            # Enhance contrast and convert to grayscale
            if gray:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.25).convert('L')
            # Save as PNG
            img.save(img_file, 'PNG')


class Box:
    """ Define boundary Box."""
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.x1 = x + dx
        self.y1 = y + dy

    def add_plot(self, ax, tiles):
        """Add Rectangle to legend plot."""
        rectangle = Rectangle(
            (self.x - tiles.tile_range['x'][0], self.y - tiles.tile_range['y'][0]), 
            self.dx, 
            self.dy, 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        # Add the patch to the Axes
        ax.add_patch(rectangle)

    def stitch_array(self, tiles):
        """Generate image from Box and tiles."""
        px_size = tiles.tile_px
        x_size = px_size[0] * math.ceil(self.dx)
        y_size = px_size[1] * math.ceil(self.dy)
        # Create an empty image
        result = Image.new('RGB', (x_size, y_size), color=(255, 255, 255))
        # Create tiles for box
        range_x = range(int(self.x), int(self.x1))
        range_y = range(int(self.y), int(self.y1))
        box_tiles = set([(x, y) for x in range_x for y in range_y])
        missing_tiles = box_tiles - tiles.tiles
        if len(missing_tiles) > 0:
            print("Missing tiles: ", len(missing_tiles))
            if tiles.downloader is not None:
                print("Downloading missing tiles...")
                tiles.downloader.download_tiles(missing_tiles, tiles.zoom)
                tiles.update()
                missing_tiles = box_tiles - tiles.tiles
        # Read tiles if present and paste into canvas
        for tile in box_tiles:
            if tile in missing_tiles:
                continue
            tile_file = os.path.join(tiles.path, str(tiles.zoom), str(tile[0]), str(tile[1]) + '.png') 
            tile_image = Image.open(tile_file)
            result.paste(im=tile_image, box=((tile[0]-self.x) * px_size[0], (tile[1]-self.y) * px_size[1]))
        # FUTURE: crop if box coords are float
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

    def _is_in_range(self, pt, pt1, pt2):
        """Check if a point is within a boundary box defined by two points."""
        max_x = max(pt1[0], pt2[0])
        min_x = min(pt1[0], pt2[0])
        max_y = max(pt1[1], pt2[1])
        min_y = min(pt1[1], pt2[1])
        within_x = (pt[0] >= min_x) and (pt[0] <= max_x)
        within_y = (pt[1] >= min_y) and (pt[1] <= max_y)
        return within_x and within_y

    def cut_path(self, lon, lat, size='256x256'):
        """ Cut path fragment using the rectangle."""
        def add_intersection():
            """ Add intersection with the Box border for begining or end."""
            # Generate line equations Ax + By = C from two points and edges
            line = utils.line(prev_pt, pt)
            top = {'a': 0, 'b': 1, 'c': self.y1}
            bottom = {'a': 0, 'b': 1, 'c': self.y}
            left = {'a': 1, 'b': 0, 'c': self.x}
            right = {'a': 1, 'b': 0, 'c': self.x1}
            # Check intersection of segment with edges
            pt_begin = None
            for edge in [top, bottom, left, right]:
                intersection = utils.line_intersection(line, edge)
                if intersection is None:
                    continue
                if self._is_in_range(intersection, pt, prev_pt):
                    pt_begin = intersection
            if pt_begin is not None:
                result.append(((pt_begin[0] - self.x) * px_size[0], (pt_begin[1] - self.y) * px_size[1]))    

        px_size = [int(px) for px in size.split('x')]
        result = []
        path_len = len(lon)
        prev_within = False
        for pt_idx, pt in enumerate(zip(lon, lat)):
            within_box = pt[0] >= self.x and pt[0] <= self.x1 and pt[1] >= self.y and pt[1] <= self.y1
            if within_box:
                # Add previous point if idx != 0
                if len(result) == 0 and pt_idx > 0:
                    prev_pt = (lon[pt_idx - 1], lat[pt_idx - 1])
                    add_intersection()
                # Add current point
                result.append(((pt[0] - self.x) * px_size[0], (pt[1] - self.y) * px_size[1]))
            # Add new point if previous was within and current is not  and is not the last point
            elif prev_within and pt_idx < path_len:
                prev_pt = (lon[pt_idx - 1], lat[pt_idx - 1])
                add_intersection()
            prev_within = within_box
        return result

    def add_scale(self, img, tiles, pos=(0.03, 0.97)):
        """ Add distance scale to image. """
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
        # Horizontal 10 km line
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


def  gen_boxes(tiles, dx=10, dy=10, minpx=0):
    """ Generate boxes by dividing the canvas. """
    box_list = []
    (size_y, size_x) = tiles.tile_array.shape
    for n_x in range(int(np.ceil(size_x/dx))):
        for n_y in range(int(np.ceil(size_y/dy))):
            d_x = min((n_x+1)*dx, size_x) - n_x * dx
            d_y = min((n_y+1)*dy, size_y) - n_y * dy
            box = Box(tiles.tile_range['x'][0]+ n_x*dx, tiles.tile_range['y'][0] + n_y*dy, d_x, d_y)
            subarr = tiles.tile_array[
                box.y - tiles.tile_range['y'][0]:box.y1 - tiles.tile_range['y'][0], 
                box.x - tiles.tile_range['x'][0]:box.x1 - tiles.tile_range['x'][0]
            ]
            if subarr.sum((0, 1)) > minpx:
                box_list.append(box)
    return box_list


def  gen_boxes_from_track(track, dx=10, dy=10, border=1):
    """ Generate boxes from track."""
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
    # Loop over each point of track
    for idx, point in enumerate(zip(track[0], track[1])):
        extent = Range.get_extent(point)
        # TODO: Allow for float coords in boxes
        P = (extent['dx'] <= (dx - border)) and (extent['dy'] <= (dy - border))
        L = (extent['dx'] <= (dy - border)) and (extent['dy'] <= (dx - border))
        # If neither Portrait nor Landscape view fits, create end the rectangle 
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
        action="store",
        dest="water",
        type=int,
        nargs=3,
        default=[None, None, None],
        help="removes water color given as R G B; 170 211 223 for OSM"
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
    parser.add_argument(
        '--dl',
        action="store",
        dest="tile_dl",
        type=str,
        default='',
        help=r"URL for downloading missing tiles, e.g.: https://a.tile.openstreetmap.org/{z}/{x}/{y}.png for OSM"
    )
    args = parser.parse_args()

    # Start plotting
    # Generate tile canvas
    tiles = Tiles(args.tile_path, zoom=args.zoom, web_source=args.tile_dl)

    # Load GPX trace
    gpx_trace = tiles.import_gpx(args.gpx)
    # TODO: Create maps with all tiles if no GPX is given
    # boxes =  gen_boxes(tiles, dx=11, dy=15, minpx=1)
    boxes =  gen_boxes_from_track(gpx_trace, dx=args.nx, dy=args.ny)
    print("Number of charts: ", len(boxes))
    # Update Tiles with new extent from boxes
    old_extent = tiles.tile_range
    p1 = [None, None]
    p2 = [None, None]
    for idx, box in enumerate(boxes):
        if idx == 0:
            p1 = [box.x, box.y]
            p2 = [box.x1, box.y1]
        else:
            min_x = int(min(p1[0], box.x))
            min_y = int(min(p1[1], box.y))
            max_x = int(max(p2[0], box.x1))
            max_y = int(max(p2[1], box.y1))
            p1 = [min_x, min_y]
            p2 = [max_x, max_y]
    tiles.update_array(p1, p2)

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
    for idx, r in enumerate(boxes):
        r.add_plot(ax, tiles)
        # Add rectangle number
        plt.text(r.x + r.dx * 0.25, r.y + r.dy * 0.75, str(idx), fontsize=9, color='white')
    plt.plot([x - tiles.tile_range['x'][0] for x in lon], [y - tiles.tile_range['y'][0] for y in lat])
    plt.savefig(os.path.join(args.output_dir, 'legend.png'))

    # Save separate maps
    tiles.generate_maps(boxes, track=gpx_trace, path=args.output_dir, gray=args.gray, water=args.water)
