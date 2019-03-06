#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
#ifdef FLO_USE_DOUBLE_PRECISION
__device__ double atomicAdd(double* __restrict__ i_address, const double i_val)
{
  auto address_as_ull = reinterpret_cast<unsigned long long int*>(i_address);
  unsigned long long int old = *address_as_ull, ass;
  do
  {
    ass = old;
    old = atomicCAS(address_as_ull,
                    ass,
                    __double_as_longlong(i_val + __longlong_as_double(ass)));
  } while (ass != old);
  return __longlong_as_double(old);
}
#endif
// block dim should be 3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write two values per edge
__global__ void
d_intrinsic_dirac_atomic(const real3* __restrict__ di_vertices,
                         const int* __restrict__ di_faces,
                         const real* __restrict__ di_face_area,
                         const int* __restrict__ di_cumulative_valence,
                         const int* __restrict__ di_entry_offset,
                         const uint i_nfaces,
                         int* __restrict__ do_rows,
                         int* __restrict__ do_columns,
                         real4* __restrict__ do_values)
{
  // Declare one shared memory block
  extern __shared__ uint8_t shared_memory[];
  // Create pointers into the block dividing it for the different uses
  real* __restrict__ cached_value = (real*)shared_memory;
  // There is a cached value for each corner of the face so we offset
  real3* __restrict__ points = (real3*)(cached_value + blockDim.x * 3);
  // There are nfaces *3 vertex values (duplicated for each face vertex)
  real* __restrict__ edge_norm2 = (real*)(points + blockDim.x * 3);
  // There are nfaces *3 squared edge lengths (duplicated for each face vertex)
  uint32_t* __restrict__ eid = (uint32_t*)(edge_norm2 + blockDim.x * 3);

  // Calculate which face this thread is acting on
  const uint fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uchar3 loop = edge_loop(threadIdx.y >> 1);

  // Compute local edge indices rotated by the corner this thread corresponds to
  const uint16_t local_e0 = threadIdx.x * 3 + loop.x;
  const uint16_t local_e1 = threadIdx.x * 3 + loop.y;
  const uint16_t local_e2 = threadIdx.x * 3 + loop.z;

  // This thread will write to column or row major triangle based on even or odd
  const uint8_t major = !(threadIdx.y & 1);

  // Only write once per face
  if (!threadIdx.y)
  {
    // Duplicate for each corner of the face to reduce bank conflicts
    cached_value[local_e0] = cached_value[local_e1] = cached_value[local_e2] =
      di_face_area[fid] * 8.f;
  }
  // Write the vertex positions into shared memory
  if (major)
  {
    points[local_e0] = di_vertices[di_faces[fid * 3 + loop.x]];
  }
  __syncthreads();
  // Compute squared length of edges and write to shared memory
  if (major)
  {
    const real3 e = points[local_e2] - points[local_e1];
    edge_norm2[local_e0] = dot(e, e);
  }
  __syncthreads();
  if (major)
  {
    // Save the cotangent value into shared memory as multiple threads will,
    // write it into the final matrix
    cached_value[local_e0] =
      (edge_norm2[local_e1] + edge_norm2[local_e2] - edge_norm2[local_e0]) /
      cached_value[local_e0];
  }
  // Write the opposing edge ID's into shared memory to reduce global reads
  eid[local_e0 * 2 + !major] =
    di_faces[fid * 3 + nth_element(loop, 1 + !major)];
  __syncthreads();

  const uint32_t R = eid[local_e0 * 2 + !major];
  const uint32_t C = eid[local_e0 * 2 + major];
  const uint32_t address = di_entry_offset[fid * 6 + threadIdx.y] + R + (C > R);
  // Write the row and column indices
  do_rows[address] = R;
  do_columns[address] = C;
  atomicAdd(do_values + address, -cached_value[local_e0]);
}

}  // namespace

void intrinsic_dirac(
  const thrust::device_ptr<const real3> di_vertices,
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const real> di_face_area,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const thrust::device_ptr<const int2> di_entry_offset,
  const int i_nverts,
  const int i_nfaces,
  const int i_total_valence,
  thrust::device_ptr<int> do_diagonals,
  thrust::device_ptr<int> do_rows,
  thrust::device_ptr<int> do_columns,
  thrust::device_ptr<real4> do_values)
{
  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = i_nfaces * 6 / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size =
    sizeof(flo::real) * block_dim.x * 15 + sizeof(uint32_t) * 6 * block_dim.x;

  // When passing the face and offset data to cuda, we reinterpret them as int
  // arrays. The advantage of this is coalesced memory reads by neighboring
  // threads, and access at a more granular level.
  // The cast is inherently safe due to the alignment of cuda vector types,
  // and reinterpret casting guarantees no changes to the underlying values
  d_intrinsic_dirac_atomic<<<nblocks, block_dim, shared_memory_size>>>(
    di_vertices.get(),
    reinterpret_cast<const int*>(di_faces.get()),
    di_face_area.get(),
    di_cumulative_valence.get(),
    reinterpret_cast<const int*>(di_entry_offset.get()),
    i_nfaces,
    do_rows.get(),
    do_columns.get(),
    do_values.get());
  cudaDeviceSynchronize();

  thrust::counting_iterator<int> counter(0);
  thrust::copy_if(counter + di_cumulative_valence[1] + 1,
                  counter + i_total_valence + i_nverts,
                  do_diagonals + 1,
                  [do_rows = do_rows.get()] __device__ (int x) {
                    return !do_rows[x];
                  });

  // Iterator for diagonal matrix entries
  auto diag_begin = thrust::make_permutation_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(do_rows, do_columns)),
    do_diagonals);

  // Generate the diagonal entry, row and column indices
  thrust::transform(
    counter, counter + i_nverts, diag_begin, [] __device__(const int i) {
      return thrust::make_tuple(i, i);
    });

  thrust::reduce_by_key(
    do_rows,
    do_rows + i_total_valence + i_nverts,
    thrust::make_transform_iterator(do_values, thrust::negate<flo::real>()),
    thrust::make_discard_iterator(),
    thrust::make_permutation_iterator(do_values, do_diagonals));
}

FLO_DEVICE_NAMESPACE_END

