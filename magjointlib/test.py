from __future__ import division
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import pycuda.autoinit
import numpy.testing
from pycuda import gpuarray, tools


def array_format_to_dtype(af):
    if af == drv.array_format.UNSIGNED_INT8:
        return np.uint8
    elif af == drv.array_format.UNSIGNED_INT16:
        return np.uint16
    elif af == drv.array_format.UNSIGNED_INT32:
        return np.uint32
    elif af == drv.array_format.SIGNED_INT8:
        return np.int8
    elif af == drv.array_format.SIGNED_INT16:
        return np.int16
    elif af == drv.array_format.SIGNED_INT32:
        return np.int32
    elif af == drv.array_format.FLOAT:
        return np.float32
    else:
        raise TypeError(
                "cannot convert array_format '%s' to a numpy dtype"
                % array_format)

#
# numpy3d_to_array
# this function was
# taken from pycuda mailing list (striped for C ordering only)
#
def numpy3d_to_array(np_array, allow_surface_bind=True):

    import pycuda.autoinit

    d, h, w = np_array.shape

    descr = drv.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = drv.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    if allow_surface_bind:
        descr.flags = drv.array3d_flags.SURFACE_LDST

    device_array = drv.Array(descr)

    copy = drv.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return device_array


def array_to_numpy3d(cuda_array):

    import pycuda.autoinit

    descriptor = cuda_array.get_descriptor_3d()

    w = descriptor.width
    h = descriptor.height
    d = descriptor.depth

    shape = d, h, w

    dtype = array_format_to_dtype(descriptor.format)

    numpy_array=np.zeros(shape, dtype)

    copy = drv.Memcpy3D()
    copy.set_src_array(cuda_array)
    copy.set_dst_host(numpy_array)

    itemsize = numpy_array.dtype.itemsize

    copy.width_in_bytes = copy.dst_pitch = w*itemsize
    copy.dst_height = copy.height = h
    copy.depth = d

    copy()

    return numpy_array


src_module=r'''
#include <stdint.h>
#include <cuda.h>
#include <surface_functions.h>

texture<float, cudaTextureType3D, cudaReadModeElementType> tex_in;

__global__ void interpol(int32_t Nz, int32_t Ny, int32_t Nx, float3 *p)
{

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  if (x < Nx && y < Ny && z < Nz) {
    int index = x+y+z;
    float value = tex3D(tex_in, (float) x, (float) y, float (z));
    p[index].x = value;
    //p[index].y = value;
    //p[index].z = value;
  }

}
'''

mod=SourceModule(src_module, cache_dir=False, keep=False)

kernel=mod.get_function("interpol")
arg_types = (np.int32, np.int32, np.int32)

tex_in=mod.get_texref('tex_in')

# random shape
shape_x = np.int32(3)
shape_y = np.int32(3)
shape_z = np.int32(3)

p1 = np.zeros((shape_x*shape_y*shape_z,1),dtype=np.float32,order='C')
p1_gpu = gpuarray.to_gpu(p1)

dtype=np.float32 # should match src_module's datatype

# numpy_array_in=np.random.randn(shape_z, shape_y, shape_x).astype(dtype).copy()
numpy_array_in = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,1,1],[2,2,2],[3,3,3]],[[1,2,3],[1,2,3],[1,2,3]]],dtype=np.float32)
cuda_array_in = numpy3d_to_array(numpy_array_in)
tex_in.set_array(cuda_array_in)

bdim = (16, 16, 1)
dx, mx = divmod(shape_x, bdim[0])
dy, my = divmod(shape_y, bdim[1])
dz, mz = divmod(shape_z, bdim[2])
gdim = ( int((dx + (mx>0))), int((dy + (my>0))), int((dz + (mz>0))))

interpol = mod.get_function("interpol")
interpol(shape_z, shape_y, shape_x, p1_gpu, block=bdim, grid=gdim, texrefs=[tex_in])

# kernel.prepare(arg_types,texrefs=[tex_in])
# kernel.prepared_call(grid, block, shape_z, shape_y, shape_x, p1_gpu)

# print(array_to_numpy3d(cuda_array_in))
print(np.reshape(p1_gpu.get(),(shape_x,shape_y,shape_z)))
