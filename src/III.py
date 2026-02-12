'''III... coding: runs a 2D image codec for each image of a sequence.'''

import sys
import io
import os
from skimage import io as skimage_io # pip install scikit-image
import main
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_video_coding as EVC
#from entropy_video_coding import Video
import av  # pip install av
from PIL import Image
import importlib
import re

# Encoder parser
parser.parser_encode.add_argument("-T", "--transform", type=str, 
    help=f"2D-transform, default: {EVC.DEFAULT_TRANSFORM}", 
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_encode.add_argument("-N", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to encode (default: {EVC.N_FRAMES})", default=f"{EVC.N_FRAMES}")

# Decoder parser
parser.parser_decode.add_argument("-T", "--transform", type=str,
    help=f"2D-transform, default: {EVC.DEFAULT_TRANSFORM}", 
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_decode.add_argument("-N", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to decode (default: {EVC.N_FRAMES})", default=f"{EVC.N_FRAMES}")

args = parser.parser.parse_known_args()[0]

if __debug__:
    if args.debug:
        print(f"III: Importing {args.transform}")

try:
    transform = importlib.import_module(args.transform)
except ImportError as e:
    print(f"Error: Could not find {args.transform} module ({e})")
    print(f"Make sure '2D-{args.transform}.py' is in the same directory as III.py")
    sys.exit(1)

def is_valid_name(name):
        pattern = r'^encoded_\d{4}\.png$'
        return bool(re.match(pattern, name))

class CoDec(EVC.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        #codec_args: any
        # codec_args = args
        self.transform_codec = transform.CoDec(args)
        logging.info(f"Using {args.transform} codec")

    def bye(self):
        super().bye()

    def encode(self):
        '''Input a file recognized by av (that can be also a single
        image) and output one or more files depending on the selected
        2D image encoder.

        '''
        logging.debug("trace")
        fn = self.args.original
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        exit = False
        for packet in container.demux():
            if __debug__:
                self.total_input_size += packet.size
            for frame in packet.decode():
                print(len(packet.decode()))
                img = frame.to_image()
                #img_fn = f"{EVC.ENCODE_OUTPUT_PREFIX}_%04d.png" % img_counter
                img_fn = f"/tmp/original_%04d.png" % img_counter
                img_fnNOPNG = f"{EVC.ENCODE_OUTPUT_PREFIX}_%04d" % img_counter
                #img_fnNOPNG = f"/tmp/original_%04d" % img_counter
                img.save(img_fn)
                if __debug__:
                    O_bytes = os.path.getsize(img_fn)
                    #self.total_output_size += O_bytes
                    logging.info(f"Extracted frame {img_fn} {img.size} {img.mode} in={packet.size} out={O_bytes}")
                else:
                    logging.info(f"Extracted frame {img_fn} {img.size} {img.mode} in={packet.size}")
                #self.transform_codec.args.input = img_fn
                #self.transform_codec.args.output = img_fnNOPNG
                #self.output = img_fnNOPNG
                #self.transform_codec.encode_javi(img_array)
                #logging.info(f"Generated {}")
                #O_bytes = self.transform_codec.encode()
                O_bytes = self.transform_codec.encode_fn(img_fn, img_fnNOPNG)
                #logging.info(f"O_bytes={O_bytes}")
                #O_bytes = os.path.getsize(img_fnNOPNG + ".TIFF") # Esto no debería estar aqui!
                self.total_output_size += O_bytes
                img_counter += 1
                #print("--------------->", img_counter, args.number_of_frames)
                logging.info(f"img_counter = {img_counter} / {args.number_of_frames}")
                if img_counter >= args.number_of_frames:
                    print("Uf")
                    exit = True
                img_fn = ""
                img_fnNOPNG = ""
            if exit:
                break
        self.N_frames = img_counter
        self.width, self.height = img.size
        self.N_channels = len(img.mode)

    def decode(self):
        '''
        img_fns = []
        for fn in os.listdir("/tmp/"):
            if is_valid_name(fn):
                img_fns.append(fn)
        sorted_img_fns = sorted(img_fns)
        '''
        logging.debug("trace")
        img_counter = 0
        #for img in imgs:
        #for i in range(len(sorted_img_fns)):
        for i in range(self.args.number_of_frames):
            img_fn = f"{EVC.DECODE_OUTPUT_PREFIX}_%04d.png" % img_counter
            img_fnNOPNG = f"{EVC.ENCODE_OUTPUT_PREFIX}_%04d" % img_counter
            #img_fn = f"/tmp/original_%04d.png" % img_counter
            #img_fnNOPNG = f"/tmp/original_%04d" % img_counter
            #logging.info(img_fn)
            #self.transform_codec.args.input = img_fn[:-4]
            #self.input = img_fn[:-4]
            #self.transform_codec.args.output= img_fn
            #logging.info(f"Decoding frame {self.transform_codec.args.input} into {self.transform_codec.args.output}")
            logging.info(f"Decoding frame {img_fnNOPNG} into {img_fn}")
            self.transform_codec.decode_fn(img_fnNOPNG, img_fn)
            img_counter += 1

            #img.save(self.args.output)
        # Open the output file container for the resulting video

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
