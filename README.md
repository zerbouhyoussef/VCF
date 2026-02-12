# Visual Coding Framework
A programming environment to develop and test image and video compression algorithms.

## Install and configuration

Supposing that a Python interpreter and Git are available:

      python -m venv ~/envs/VCF
      git clone git@github.com:Sistemas-Multimedia/VCF.git
      cd VCF
      source ~/envs/VCF/bin/activate
      pip install -r requirements

## Usage

### Image coding (example)

      cd src
      python PNG.py encode
      display /tmp/encoded.png
      python PNG.py decode
      display /tmp/decoded.png

### Video coding (example)

      cd src
      python III.py encode
      ffplay /tmp/encoded_%04d.tif
      python III.py decode
      ffplay /tmp/decoded_%04d.png

## Codecs organization

	+---------------------+        +----+
	| temporal transforms |    III |-T,N|, [IPP] (9), [IBP] (10), [MCTF] (10).
	+---------------------+--+     +---++-------+
	| spatial transforms  |-T| 2D-DCT* |-B,p,L,x|, 2D-DWT, [LBT] (10), no_spatial_transform.
	+---------------------+--+         +--------+
	|  color transforms   |-t| YCoCg*, YCrCb, color-DCT, no_color_transform.
	+---------------------+--+           +--+           +------+     +----+           +--+
	|     quantizers      |-a| deadzone* |-q|, LloydMax |-q,m,n|, VQ |-q,b|, color-VQ |-q|.
	+---------------------+--+           +--+           ++--+--+     +----+           +--+
	|  decoding filters   |-f| no_filter*, gaussian_blur |-s|, [NLM] (1), [BM3D] (3)
	+---------------------+--+                           +--+
	|   entropy codecs    |-c| TIFF*, PNG, Huffman, PNM, [adaptive_Huffman] (4), [arith] (4), [adaptive_arith] (5).
	+---------------------+--+

	...* = default option
	[...] = to be implemented
	(.) = points for the evaluation of the subject
