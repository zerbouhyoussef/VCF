'''Shared code among the video entropy codecs.'''

import os
import io
from skimage import io as skimage_io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
#import subprocess
import cv2
import main
import urllib
from urllib.parse import urlparse
#import requests
import av
import math

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

#import entropy_image_coding as EIC

# Default IOs
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.mp4"
ENCODE_OUTPUT_PREFIX = "/tmp/encoded"  # File extension decided in run-time
DECODE_INPUT_PREFIX = ENCODE_OUTPUT_PREFIX
DECODE_OUTPUT_PREFIX = "/tmp/decoded"
DECODE_OUTPUT = "/tmp/decoded.mp4"

N_FRAMES = 3

DEFAULT_TRANSFORM = "2D-DCT"

class UNUSED_Video:
    '''A video is a sequence of files stored in "prefix".'''

    def __init__(self, N_frames, height, width, fn):
        logging.debug("trace")
        self.N_frames = N_frames
        self.height = height
        self.width = width
        self.fn = fn

    def get_shape(self):
        logging.debug("trace")
        return self.N_frames, self.height, self.width

#class CoDec(EIC.CoDec):
class CoDec:

    def __init__(self, args):
        logging.debug("trace")
        self.args = args
        #super().__init__(args)
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"self.encoding = {self.encoding}")
        self.total_input_size = 0
        self.total_output_size = 0
        self.framerate = 30

    def bye(self):
        return
        logging.debug("trace")
        #logging.info(f"Total {self.total_input_size} bytes read")
        #logging.info(f"Total {self.total_output_size} bytes written")
        #logging.info(f"Number of color components = {self.N_channels}")
        if __debug__:
            if self.encoding:
                BPP = (self.total_output_size*8)/(self.N_frames*self.width*self.height)
                #BPP = self.N_frames*self.width*self.height*self.N_channels/self.total_output_size
                #BPP = self.N_frames*self.width*self.height*self.N_channels/self.get_output_bytes()
                logging.info(f"Output bit-rate = {BPP} bits/pixel")
                # Deberíamos usar un fichero distinto del que usa entropy_image_coding
                with open(f"{self.args.output}.txt", 'w') as f:
                    f.write(f"{self.args.input}\n")
                    f.write(f"{self.N_frames}\n")
                    f.write(f"{self.height}\n")
                    f.write(f"{self.width}\n")
                    f.write(f"{BPP}\n")
            else:
                with open(f"{self.args.input}.txt", 'r') as f:
                    original_file = f.readline().strip()
                    logging.info(f"original_file = {original_file}")
                    N_frames = int(f.readline().strip())
                    logging.info(f"N_frames = {N_frames}")
                    height = f.readline().strip()
                    logging.info(f"video height = {height} pixels")
                    width = f.readline().strip()
                    logging.info(f"video width = {width} pixels")
                    BPP = float(f.readline().strip())
                    logging.info(f"BPP = {BPP}")
                logging.info(f"Number of encoded frames = {N_frames}")
                logging.info(f"Video height (rows) = {height}")
                logging.info(f"Video width (columns) = {width}")
                total_RMSE = 0
                for i in range(N_frames):
                    x = self.encode_read_fn(f"file:///tmp/original_{i:04d}.png")
                    y = self.encode_read_fn(f"file:///tmp/decoded_{i:04d}.png")
                    img_RMSE = distortion.RMSE(x, y)
                    logging.debug(f"image RMSE = {img_RMSE}")
                    total_RMSE += img_RMSE
                RMSE = total_RMSE / N_frames
                logging.info(f"Mean RMSE = {RMSE}")

                J = BPP + RMSE
                logging.info(f"J = R + D = {J}")

                logging.info(f"Output: {self.args.output}")

                self.args.output = DECODE_OUTPUT
                container = av.open(self.args.output, 'w', format='avi')
                video_stream = container.add_stream('libx264', rate=self.framerate)

                # Set lossless encoding options
                #video_stream.options = {'crf': '0', 'preset': 'veryslow'}
                video_stream.options = {'crf': '0', 'preset': 'ultrafast'}

                # Optionally set pixel format to ensure no color space conversion happens
                video_stream.pix_fmt = 'yuv444p'  # Working but lossy because the YCrCb is floating point-based
                #video_stream.pix_fmt = 'rgb24'  # Does not work :-/
                imgs = []
                for i in range(N_frames):
                    #print("FILE: " + file + " " + str(len(file)))
                    imgs.append(f"{DECODE_OUTPUT_PREFIX}_%04d.png" % i)
                #img_0 = Image.open("/tmp/encoded_0000.png").convert('RGB')
                #img_0 = Image.open(imgs[0]).convert('RGB')
                #width, height = img_0.size
                video_stream.width = int(width)
                video_stream.height = int(height)
                #self.width, self.height = img_0.size
                #self.N_channels = len(img_0.mode)

                img_counter = 0
                for i in imgs:
                    img = Image.open(i).convert('RGB')
                    logging.info(f"Re-encoding frame {img_counter} into {self.args.output}")

                    # Convert the image to a VideoFrame
                    frame = av.VideoFrame.from_image(img)

                    # Encode the frame and write it to the container
                    packet = video_stream.encode(frame)
                    container.mux(packet)
                    img_counter += 1

                # Ensure all frames are written
                container.mux(video_stream.encode())
                container.close()
                self.N_frames = img_counter
                #vid = compressed_vid
                #vid.prefix = DECODE_OUTPUT
                #return vid
        
                '''
                container_x = av.open(original_file)
                container_y = av.open(self.args.output)
                img_counter = 0
                total_RMSE = 0
                logging.info("Computing RD performance ...")
                for frame_x, frame_y in zip(container_x.decode(video=0), container_y.decode(video=0)):
                    img_x = np.array(frame_x.to_image())
                    img_y = np.array(frame_y.to_image())
                    #img_RMSE = distortion.RMSE(img_x, img_y)
                    #print(img_RMSE)
                    #total_RMSE += img_RMSE
                    #print(f"{img_counter}/{self.N_frames}", end='\r', flush=True)
                    img_counter += 1
                RMSE = total_RMSE / self.N_frames
                logging.info(f"RMSE = {RMSE}")
                '''

    def encode_read_fn(self, fn):
        logging.debug("trace")
        #img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        #img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        try:
            input_size = os.path.getsize(fn)
            self.total_input_size += input_size 
            img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(fn, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.total_input_size += input_size
            img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        logging.debug(f"Read {fn} with shape {img.shape} and type={img.dtype}")
        self.img_shape = img.shape
        return img
                
    def UNUSED_encode(self, in_fn, out_fn):
        logging.debug("trace")
        vid = self.encode_read(in_fn)
        #self.vid_shape = vid.get_shape()
        compressed_vid = self.compress(vid, out_fn)
        logging.info(f"output_bytes={self.get_output_bytes()}")

        #self.shape = compressed_vid.get_shape()
        output_size = self.encode_write(compressed_vid, out_fn)
        return output_size

    def UNUSED_decode(self, in_fn, out_fn):
        logging.debug("trace")
        compressed_vid = self.decode_read(in_fn)
        vid = self.decompress(compressed_vid, in_fn)
        output_size = self.decode_write(vid, out_fn)
        return output_size

    def UNUSED_encode_read(self, fn):
        vid = self.encode_read_fn(self.args.input)
        if __debug__:
            self.decode_write_fn(vid, "/tmp/original.avi") # Save a copy for comparing later
            self.total_output_size = 0
        return vid

    def is_http_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme.lower() in ['http', 'https']
        except ValueError:
            return False

    def UNUSED_read_video(self, fn):
        '''"Read" the video <fn>, which can be a URL. The video is
        saved in "/tmp/<fn>".'''
    
        if self.is_http_url(fn):
            response = requests.get(fn, stream=True)
            if response.status_code == 200: # If the download was successful
                input_size = 0
                #file_path = os.path.join("/tmp", fn)
                file_path = "/tmp/original.avi"
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            input_size += 8192
                            print('.', end='', flush=True)
                print("\nVideo downloaded")
                #fn = io.BytesIO(response.content) # Open the downloaded video as a byte stream
                #input_size = len(fn)
                #req = urllib.request.Request(fn, method='HEAD')
                #f = urllib.request.urlopen(req)
                #input_size = int(f.headers['Content-Length'])
        else:
            input_size = os.path.getsize(fn)
        self.total_input_size += input_size

        cap = cv2.VideoCapture(fn)
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(fn, N_frames)
        #digits = len(str(N_frames))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if __debug__:
            self.N_frames = N_frames
            self.shape = (N_frames, height, width)

        vid = Video(N_frames, height, width, fn)
        logging.info(f"Read {input_size} bytes from {fn} with shape {vid.get_shape()[1:]}")

        return vid

    def UNUSED_encode_write(self, compressed_vid):
        '''Save to disk the video specified in the class attribute args.output.'''
        self.encode_write_fn(compressed_vid, self.args.output)

    def UNUSED_encode_write_fn(self, data, fn_without_extention):
        #data.seek(0)
        fn = fn_without_extention + self.file_extension
        with open(fn, "wb") as output_file:
            output_file.write(data.read())
        self.total_output_size += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn}")

    def UNUSED_decode_read(self):
        compressed_vid = self.decode_read_fn(self.args.input)
        return compressed_vid

    def UNUSED_decode_write(self, vid):
        return self.decode_write_fn(vid, self.args.output)

    def UNUSED_decode_read(self, fn_without_extention):
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.total_input_size += input_size
        logging.info(f"Read {os.path.getsize(fn)} bytes from {fn}")
        data = open(fn, "rb").read()
        return data

    def UNUNSED_decode_write(self, vid, fn):
        frames = [e for e in os.listdir(vid.prefix)]
        for i in frames:
            skimage_io.imsave(fn, img)
        self.total_output_size += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

###################################################

    def UNUSED_encode_read_fn(self, fn):
        '''Read the video <fn>.'''

    
        if __is_http_url(fn):
            response = requests.get(fn) # Download the video file (in memory)
            if response.status_code == 200: # If the download was successful
                fn = BytesIO(response.content) # Open the downloaded video as a byte stream
                input_size = len(fn)
        else:
            input_size = os.path.getsize(fn)
        self.total_input_size += input_size 
        cap = cv2.VideoCapture(fn)
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        digits = len(str(N_frames))
        img_counter = 0
        while True:
            ret, img = cap.read()

            if not ret:
                break # Break the loop if the video has ended

            # Write the frame in /tmp/VCF_input
            img_fn = os.path.join("/tmp", f"img_{img_counter:0{digits}d}.png")
            img_counter += 1
            cv2.imwrite(img_fn, img)
        return Video(N_frames, img.shape[0], img.shape[1], "/tmp/img_")

    def UNUSED_encode_read_fn(self, fn):
        '''Read the video <fn>.'''

        from urllib.parse import urlparse
        import imageio_ffmpeg as ffmpeg
    
        if __is_http_url(fn):
            response = requests.get(fn) # Download the video file (in memory)
            if response.status_code == 200: # If the download was successful
                fn = BytesIO(response.content) # Open the downloaded video as a byte stream
                input_size = len(fn)
        else:
            input_size = os.path.getsize(fn)
        self.total_input_size += input_size

        cap = cv2.VideoCapture(fn)
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        digits = len(str(N_frames))

        with ffmpeg.get_reader(fn) as reader:
            for i, img in enumerate(reader):
                frame_array = np.array(img)        

                # Write the frame in /tmp/img_
                img_fn = os.path.join("/tmp", f"img_{img_counter:0{digits}d}.png")
                img_counter += 1
                cv2.imwrite(img_fn, img)
            N_frames = len(reader)
            logging.info(f"")
        return Video(N_frames, img.shape[0], img.shape[1], "/tmp/frame_")
