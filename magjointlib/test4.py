import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

w = 2
h = 2
shape = (d, h)

a = np.arange(w*h).reshape(*shape,order='C').astype('float32')
print(a.shape,a.strides)
print(a)
print('------------------------')


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
    texture<float, 3, cudaReadModeElementType> mtx_tex_0;
    texture<float, 3, cudaReadModeElementType> mtx_tex_1;
    texture<float, 3, cudaReadModeElementType> mtx_tex_2;

    __global__ void copy_texture(float3 *dest)
    {
      int x = threadIdx.x;
      int y = threadIdx.y;
      int z = threadIdx.z;
      int dx = blockDim.x;
      int dy = blockDim.y;
      int i = (z*dy + y)*dx + x;
      dest[i].x = tex3D(mtx_tex_0, x, y, z);
      dest[i].y = tex3D(mtx_tex_1, x, y, z);
      dest[i].z = tex3D(mtx_tex_2, x, y, z);
    }
""")

copy_texture = mod.get_function("copy_texture")

mtx_tex_0 = mod.get_texref("mtx_tex_0")
mtx_tex_0.set_array(ary)
mtx_tex_1 = mod.get_texref("mtx_tex_1")
mtx_tex_1.set_array(ary)
mtx_tex_2 = mod.get_texref("mtx_tex_2")
mtx_tex_2.set_array(ary)

dest = np.zeros((w,h,d,3), dtype=np.float32, order="C")
copy_texture(drv.Out(dest), block=(w,h,d), texrefs=[mtx_tex_0,mtx_tex_1,mtx_tex_2])

print(dest[:,:,:,0])
print(dest[:,:,:,1])
print(dest[:,:,:,2])
