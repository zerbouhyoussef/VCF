'''Entropy Encoding of images non-adaptive Huffman Coding'''

import io
import numpy as np
import main
import logging
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC
import heapq
from collections import defaultdict, Counter
import gzip
import pickle
from bitarray import bitarray
import os
import math
from huffman_coding import huffman_coding # pip install --ignore-installed "huffman_coding @ git+https://github.com/vicente-gonzalez-ruiz/huffman_coding"

class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".huf"

    def compress_fn(self, img, fn):
        logging.debug(f"trace img={img}")
        tree_fn = f"{fn}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO()

        # Flatten the array and convert to a list
        flattened_img = img.flatten().tolist()

        # Build Huffman Tree and generate the Huffman codes
        root = huffman_coding.build_huffman_tree(flattened_img)
        codes = huffman_coding.generate_huffman_codes(root)
        huffman_coding.print_huffman_tree(root)

        # Encode the flattened array
        encoded_img = huffman_coding.encode_data(flattened_img, codes)

        # Write encoded image and original shape to compressed_img
        compressed_img.write(encoded_img.tobytes())  # Save encoded data as bytes

        # Compress and save shape and the Huffman Tree
        logging.debug(f"Saving {tree_fn}")
        with gzip.open(tree_fn, 'wb') as f:
            np.save(f, img.shape)
            pickle.dump(root, f)  # `gzip.open` compresses the pickle data

        tree_length = os.path.getsize(tree_fn)
        logging.info(f"Length of the file \"{tree_fn}\" (Huffman tree + image shape) = {tree_length} bytes")
        #self.total_output_size += tree_length

        return compressed_img

    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)
    
    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img[:10]}")
        tree_fn = f"{fn}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)
        
        # Load the shape and the Huffman Tree from the compressed file
        with gzip.open(tree_fn, 'rb') as f:
            shape = np.load(f)
            root = pickle.load(f)
    
        huffman_coding.print_huffman_tree(root)
        # Read encoded image data as binary
        encoded_data = bitarray()
        encoded_data.frombytes(compressed_img.read())
    
        # Decode the image
        decoded_data = huffman_coding.decode_data(encoded_data, root)
        if math.prod(shape) < len(decoded_data):
            decoded_data = decoded_data[:math.prod(shape) - len(decoded_data)] # Sometimes, when the alphabet size is small, some extra symbols are decoded :-/


        # Reshape decoded data to original shape
        img = np.array(decoded_data).reshape(shape).astype(np.uint8)
        return img

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)





