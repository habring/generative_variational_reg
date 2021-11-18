## A Generative Variational Model for Inverse Problems in Imaging
##
## Copyright (C) 2021 Andreas Habring, Martin Holler
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.
##
##
## libjpeg.py:
## Utility functionals for handling of JPEG images. This piece of code 
## was taken from previous work by Kristian Bredies (kristian.bredies@uni-graz.at).
##
## -------------------------
## Andreas Habring (andreas.habring@uni-graz.at)
## Martin Holler (martin.holler@uni-graz.at)
## 
## 18.11.2021
## -------------------------
## If you consider this code to be useful, please cite:
## 
## [1] @misc{habring2021generative,
##          title={A Generative Variational Model for Inverse Problems in Imaging}, 
##          author={Andreas Habring and Martin Holler},
##          year={2021},
##          eprint={2104.12630},
##          archivePrefix={arXiv},
##          primaryClass={math.OC}
##          journal={SIAM Journal on Mathematics of Data Science}}
##


import cffi
import numpy as np
import ipdb as pdb

# JPEG color space values
JCS_UNKOWN = 0
JCS_GRAYSCALE = 1
JCS_RGB = 2
JCS_YCbCr = 3
JCS_CMYK = 4
JCS_YCCK = 5 


ffi = cffi.FFI()
ffi.cdef("""
typedef enum {
        JCS_UNKNOWN,            /* error/unspecified */
        JCS_GRAYSCALE,          /* monochrome */
        JCS_RGB,                /* red/green/blue as specified by the RGB_RED, RGB_GREEN,
                                   RGB_BLUE, and RGB_PIXELSIZE macros */
        JCS_YCbCr,              /* Y/Cb/Cr (also known as YUV) */
        JCS_CMYK,               /* C/M/Y/K */
        JCS_YCCK,               /* Y/Cb/Cr/K */
        JCS_EXT_RGB,            /* red/green/blue */
        JCS_EXT_RGBX,           /* red/green/blue/x */
        JCS_EXT_BGR,            /* blue/green/red */
        JCS_EXT_BGRX,           /* blue/green/red/x */
        JCS_EXT_XBGR,           /* x/blue/green/red */
        JCS_EXT_XRGB,           /* x/red/green/blue */
        /* When out_color_space it set to JCS_EXT_RGBX, JCS_EXT_BGRX,
           JCS_EXT_XBGR, or JCS_EXT_XRGB during decompression, the X byte is
           undefined, and in order to ensure the best performance,
           libjpeg-turbo can set that byte to whatever value it wishes.  Use
           the following colorspace constants to ensure that the X byte is set
           to 0xFF, so that it can be interpreted as an opaque alpha
           channel. */
        JCS_EXT_RGBA,           /* red/green/blue/alpha */
        JCS_EXT_BGRA,           /* blue/green/red/alpha */
        JCS_EXT_ABGR,           /* alpha/blue/green/red */
        JCS_EXT_ARGB            /* alpha/red/green/blue */
} J_COLOR_SPACE;

typedef unsigned short UINT16;
typedef unsigned char JSAMPLE;
typedef unsigned int JDIMENSION;
typedef short JCOEF;

typedef JCOEF JBLOCK[64];
typedef JBLOCK *JBLOCKROW;
typedef JBLOCKROW *JBLOCKARRAY;
typedef JBLOCKARRAY *JBLOCKIMAGE;

typedef struct jpeg_decompress_struct *j_decompress_ptr;
typedef struct jvirt_barray_control *jvirt_barray_ptr;

typedef struct {
  UINT16 quantval[64]; 
  ...;
} JQUANT_TBL;

typedef struct {
  int component_id;  
  int component_index;  
  int h_samp_factor;  
  int v_samp_factor;  
  int quant_tbl_no;
  JDIMENSION width_in_blocks;
  JDIMENSION height_in_blocks;
  ...;
} jpeg_component_info;

struct jpeg_error_mgr {
  ...;
};
struct jpeg_memory_mgr {
  JBLOCKARRAY (*access_virt_barray) (j_decompress_ptr cinfo,
                                     jvirt_barray_ptr ptr,
                                     JDIMENSION start_row,
                                     JDIMENSION num_rows,
                                     bool writable);

  ...;
};

struct jpeg_decompress_struct {
  JDIMENSION image_width;
  JDIMENSION image_height;
  int num_components;
  struct jpeg_error_mgr *err;
  struct jpeg_memory_mgr *mem;
  JQUANT_TBL *quant_tbl_ptrs[4];
  J_COLOR_SPACE jpeg_color_space;
  int data_precision; 
  jpeg_component_info *comp_info;

  ...;
};
struct jvirt_barray_control {
  long dummy;
};

struct jpeg_error_mgr *jpeg_std_error(struct jpeg_error_mgr *err);

extern int jpeg_lib_version;

void jpeg_CreateDecompress(j_decompress_ptr cinfo,
                           int version, size_t structsize);
void jpeg_stdio_src(j_decompress_ptr cinfo, FILE *infile);
void jpeg_mem_src(j_decompress_ptr cinfo,
                  const unsigned char *inbuffer,
                  unsigned long insize);
int jpeg_read_header(j_decompress_ptr cinfo, bool require_image);
jvirt_barray_ptr *jpeg_read_coefficients(j_decompress_ptr cinfo);
bool jpeg_finish_decompress(j_decompress_ptr cinfo);
""")

jpeglib = ffi.verify("""
#define INCOMPLETE_TYPES_BROKEN
#include <jpeglib.h>
int jpeg_lib_version = JPEG_LIB_VERSION;
""", libraries=["jpeg"])

def get_image_info(cinfo):
    info = {}
    info['num_components'] = cinfo.num_components
    info['width'] = cinfo.image_width
    info['height'] = cinfo.image_height
    info['color_space'] = cinfo.jpeg_color_space
    return(info)

def get_comp_info(cinfo, comp):
    info = {}
    num_components = cinfo.num_components
    samp_factor = np.array([[cinfo.comp_info[i].h_samp_factor,
                             cinfo.comp_info[i].v_samp_factor]
                             for i in range(num_components)])
    subsampling = samp_factor.max(axis=0)
    info['component_id'] = cinfo.comp_info[comp].component_id
    info['h_subsampling'] = subsampling[0]/samp_factor[comp,0]
    info['v_subsampling'] = subsampling[1]/samp_factor[comp,1]
    quant_tbl = cinfo.quant_tbl_ptrs[cinfo.comp_info[comp].quant_tbl_no].quantval
    quant_tbl = np.frombuffer(ffi.buffer(quant_tbl), dtype=np.uint16)
    info['quant_tbl'] = quant_tbl.reshape((8,8))
    return(info)

def get_coeff_array(cinfo, coeff, comp):
    W = cinfo.comp_info[comp].width_in_blocks
    H = cinfo.comp_info[comp].height_in_blocks
    coeff_array = np.zeros((H,W,64), dtype=np.int16)
    rows = cinfo.mem.access_virt_barray(cinfo, coeff[comp], 0,
                                        cinfo.comp_info[comp].v_samp_factor,
                                        False)
    for i in range(H):
        for j in range(W):
            coeff_array[i,j,:] = np.frombuffer(ffi.buffer(rows[i][j]),
                                               dtype=np.int16)

    coeff_array = coeff_array.reshape((H,W,8,8))
    return(coeff_array)
    
def jpeg_decompress(file):
    jerr = ffi.new("struct jpeg_error_mgr *")
    cinfo = ffi.new("struct jpeg_decompress_struct *")
    cinfo.err = jpeglib.jpeg_std_error(jerr)
    jpeglib.jpeg_CreateDecompress(cinfo, jpeglib.jpeg_lib_version,
                    ffi.sizeof("struct jpeg_decompress_struct"))
    try:
        jpeglib.jpeg_stdio_src(cinfo, file)
    except TypeError:
        buf = file.read() + '\x00'*128
        buf_len = len(buf)
        jpeglib.jpeg_mem_src(cinfo, buf, buf_len)
    jpeglib.jpeg_read_header(cinfo, True)
    return(cinfo)


def jpeg_read_coeffs(cinfo):
    coeff = jpeglib.jpeg_read_coefficients(cinfo)
    return(coeff)
    

def jpeg_finish(cinfo):
    jpeglib.jpeg_finish_decompress(cinfo)


def jpeg_read_file(file):
    cinfo = jpeg_decompress(file)
    coeffs = jpeg_read_coeffs(cinfo)
    
    image_info = get_image_info(cinfo)
    num_components = image_info['num_components']
    comp_info = [get_comp_info(cinfo, i) for i in range(num_components)]
    coeff = [get_coeff_array(cinfo, coeffs, i) for i in
             range(num_components)]
        
    jpeg_finish(cinfo)
    return(image_info, comp_info, coeff)
    

if __name__ == "__main__":
    file = open("cbug_head.jpg", "rb")
    (image_info, comp_info, coeff) = jpeg_read_file(file)
    file.close() 
    print(image_info, comp_info, coeff)
     

