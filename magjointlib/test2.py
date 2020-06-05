import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

w = 2
h = 3
d = 4
shape = (d, h, w)

a = np.arange(24).reshape(*shape,order='C').astype('float32')
print(a.shape,a.strides)
print(a)


descr = drv.ArrayDescriptor3D()
descr.width = w
descr.height = h
descr.depth = d
descr.format = drv.dtype_to_array_format(a.dtype)
descr.num_channels = 1
descr.flags = 0

ary = drv.Array(descr)

copy = drv.Memcpy3D()
copy.set_src_host(a)
copy.set_dst_array(ary)
copy.width_in_bytes = copy.src_pitch = a.strides[1]
copy.src_height = copy.height = h
copy.depth = d

copy()

mod = SourceModule("""
    texture<float, 3, cudaReadModeElementType> mtx_tex;

    __global__ void copy_texture(float *dest)
    {
      int x = threadIdx.x;
      int y = threadIdx.y;
      int z = threadIdx.z;
      int dx = blockDim.x;
      int dy = blockDim.y;
      int i = (z*dy + y)*dx + x;
      dest[i] = tex3D(mtx_tex, x, y, z);
    }
""")

copy_texture = mod.get_function("copy_texture")
mtx_tex = mod.get_texref("mtx_tex")

mtx_tex.set_array(ary)

dest = np.zeros(shape, dtype=np.float32, order="C")
copy_texture(drv.Out(dest), block=(w,h,d), texrefs=[mtx_tex])

print(dest)
