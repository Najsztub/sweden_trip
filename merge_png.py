from PIL import Image
import os
import re
import argparse

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Merge PNG images into one TIFF/PDF document.')
    parser.add_argument("img_dir", help="Directory with PNG images")
    parser.add_argument(
        '-p',
        action="store",
        dest="prefix",
        type=str,
        default="stitch",
        help="PNG files prefix; 'stitch' by default"
    )
    parser.add_argument(
        '-o',
        action="store",
        dest="output",
        type=str,
        default="out.tiff",
        help="output TIFF/PDF file. 'out.tiff' by defult"
    )
    args = parser.parse_args()

    # Read files
    files = [os.path.join(args.img_dir, i) for i in os.listdir(args.img_dir) if i.startswith(args.prefix)]
    files.sort(key=lambda x: int(re.search(r'([0-9]+)\.png', x)[1]))

    images = []
    for idx, f in enumerate(files):
        print("Loading %s/%s" % (idx+1, len(files)))
        img = Image.open(f).convert('L')
        if img.size[0] > img.size[1]:
            img = img.transpose(Image.ROTATE_90)
        images.append(img)
    images[0].save(args.output, save_all=True, append_images=images[1:])