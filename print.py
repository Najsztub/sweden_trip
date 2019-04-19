import os
import re
import numpy as np
import math
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFont
import gpxpy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return (xtile, ytile)


class Rect:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.x1 = x + dx
        self.y1 = y + dy

    def add_plot(self, ax):
        rect = Rectangle((self.x, self.y), self.dx, self.dy, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    def stitch_array(self, tile_array, tile_range, path, size='256x256'):
        px_size = [int(px) for px in size.split('x')]
        x_size = px_size[0] * self.dx
        y_size = px_size[1] * self.dy
        # Create an empty image
        result = Image.new('RGB', (x_size, y_size), color=(255, 255, 255))
        for x in range(self.x, self.x1):
            for y in range(self.y, self.y1):
                if tile_array[y, x] == 0:
                    continue
                x_tile = x + tile_range['x'][0]
                y_tile = y + tile_range['y'][0]
                tile_file = '/'.join([path, str(x_tile), str(y_tile) + '.png'])
                # print('Loading ', tile_file)
                tile_image = Image.open(tile_file)
                result.paste(im=tile_image, box=((x-self.x) * px_size[0], (y-self.y) * px_size[1]))
        return result

    def cut_path(self, lon, lat, size='256x256'):
        px_size = [int(px) for px in size.split('x')]
        result = []
        for coord in zip(lon, lat):
            if coord[0] >= self.x and coord[0] <= self.x1 and coord[1] >= self.y and coord[1] <= self.y1:
                result.append(((coord[0] - self.x) * px_size[0], (coord[1] - self.y) * px_size[1]))
        return result

    def add_scale(self, img, tile_range, zoom, pos=(0.03, 0.97), size='256x256'):
        px_size = [int(px) for px in size.split('x')]
        x = pos[0] * self.dx * px_size[0]
        y = pos[1] * self.dy * px_size[1]
        # Get lat for px y coordinate
        # From: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Tile_numbers_to_lon..2Flat._2
        n = 2.0 ** zoom
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (tile_range['y'][0] + self.y + pos[1] * self.dy) / n)))
        # Get m / px 
        # https://wiki.openstreetmap.org/wiki/Zoom_levels
        C = 2 * math.pi * 6378137.0
        m_per_px = C * math.cos(lat_rad) / 2 ** (zoom + math.log2(px_size[0]))
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


def get_tiles(path):
    tiles = []
    for files in os.walk(path):
        if len(files[2]) == 0:
            continue
        x = re.search(r'/([0-9]+)$', files[0])[1]
        for f in files[2]:
            y = re.search('[0-9]+', f)[0]
            tiles.append((int(x), int(y)))
    return tiles

def gen_tile_array(tiles):
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

def gen_rects(tile_array, dx=10, dy=10, minpx=0):
    rects = []
    (size_y, size_x) = tile_array.shape
    for n_x in range(int(np.ceil(size_x/dx))):
        for n_y in range(int(np.ceil(size_y/dy))):
            d_x = min((n_x+1)*dx, size_x) - n_x * dx
            d_y = min((n_y+1)*dy, size_y) - n_y * dy
            rect = Rect(n_x*dx, n_y*dy, d_x, d_y)
            subarr = tile_array[rect.y:rect.y1, rect.x:rect.x1]
            if subarr.sum((0, 1)) > minpx:
                rects.append(rect)
    return rects

def gen_rects_from_track(track, dims, dx=10, dy=10, border=1):
    # TODO: Change into an iterator over new points
    def get_extent(current_points):
        x = [xy[0] for xy in current_points]
        y = [xy[1] for xy in current_points]
        tile_range = {
            'x': [int(min(x)), int(max(x))],
            'y': [int(min(y)), int(max(y))],
            'dx': int(max(x)) - int(min(x)) + 1,
            'dy': int(max(y)) - int(min(y)) + 1
        }
        return tile_range
    number_points = len(track[0])
    rects = []
    current_points = []
    prev_P = False
    prev_extent = {}
    for idx, point in enumerate(zip(track[0], track[1])):
        current_points.append(point)
        extent = get_extent(current_points)
        P = (extent['dx'] <= (dx - border)) and (extent['dy'] <= (dy - border))
        L = (extent['dx'] <= (dy - border)) and (extent['dy'] <= (dx - border))
        if len(prev_extent) > 0 and (((P or L) == False) or idx == number_points - 1):
            if prev_P:
                rot_dx = dx
                rot_dy = dy
            else:
                rot_dx = dy
                rot_dy = dx
            x = prev_extent['x'][0] - (rot_dx - prev_extent['dx']) / 2
            y = prev_extent['y'][0] - (rot_dy - prev_extent['dy']) / 2
            rects.append(Rect(int(x), int(y), rot_dx, rot_dy))
            current_points = [point]
            prev_P = False
            prev_extent = {}
            continue
        prev_extent = extent
        prev_P = P
    return rects

def import_gpx(gpx, zoom):
    lat = []
    lon = []

    gpx_track = gpxpy.parse(open(gpx, 'r'))
    for track in gpx_track.tracks:
        for segment in track.segments:
            for point in segment.points:
                x, y = deg2num(point.latitude, point.longitude, zoom=zoom)
                x -= tile_dims['x'][0]
                y -= tile_dims['y'][0]
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


if __name__ == "__main__":
    # Start plotting
    zoom = 13
    path = '/home/mateusz/projects/gis/sweden_trip/OSM_tiles/{}/'.format(zoom)
    tiles = get_tiles(path)
    tile_array, tile_dims = gen_tile_array(tiles)

    # Get rectangles
    gpx_trace = import_gpx('/home/mateusz/projects/gis/sweden_trip/routes/cycle_travel2.gpx', zoom)
    #rects = gen_rects(tile_array, dx=11, dy=15, minpx=1)
    rects = gen_rects_from_track(gpx_trace, tile_dims, dx=8, dy=11)
    print("Number of rects: ", len(rects))

    # Plot legend
    # Import track
    lat = []
    lon = []
    if gpx_trace is not None:
        (lon, lat) = gpx_trace
    #plt.plot(lat, lon)
    # Get the current reference
    plt.figure(figsize=(10,10)) 
    plt.matshow(tile_array)
    ax = plt.gca()
    # Create a Rectangle patch for each rectangle
    for idx, r in enumerate(rects):
        r.add_plot(ax)
        plt.text(r.x + r.dx * 0.25, r.y + r.dy * 0.75, str(idx), fontsize=9, color='white')
    plt.plot(lon, lat)
    plt.savefig('/home/mateusz/projects/gis/sweden_trip/OSM_map/legend.png')


    def gen_map(track=None):
        # Import track
        lat = []
        lon = []
        if track is not None:
            (lon, lat) = track
        # save Image
        for idx, r in enumerate(rects):
            # Create canvas
            img = r.stitch_array(tile_array, tile_dims, path)
            # Add scale
            r.add_scale(img, tile_dims, zoom)
            # Add track
            if track is not None:
                cut_track = r.cut_path(lon, lat)
                if len(cut_track) > 1:
                    print("Track length: %s segments" % len(cut_track))
                    draw = ImageDraw.Draw(img)
                    draw.line(cut_track, fill=(0, 0, 0), width=2)
                    del draw
            img_file = '/home/mateusz/projects/gis/sweden_trip/OSM_map/stitch_{}.png'.format(idx)
            print("Saving ", img_file)

            data = np.array(img)   # "data" is a height x width x 3 numpy array
            red, green, blue = data.T # Temporarily unpack the bands for readability

            # Replace water with white colour
            water = (red == 170) & (green == 211) & (blue == 223) 
            data[:, :][water.T] = (255, 255, 255) # Transpose back needed
            img = Image.fromarray(data)
            # Add page number
            font = ImageFont.truetype("DejaVuSans.ttf", 48)
            ImageDraw.Draw(img).text((10, 10), str(idx), (0, 0, 0), font=font)
            # Enhance contrast and convert to grayscale
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.25).convert('L')
            # Add scale - km measure
            # Save as PNG
            img.save(img_file, 'PNG')

    gen_map(track=gpx_trace)
