/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Defines basic properties needed by CTA-level GEMMs assuming
   expectations about data layout of the global memory fragments, data types,
   and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting simt
   instructions.
*/

/**
 * \file include/cutlass/convolution/threadblock/default_mma_core_simt.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/convolution/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/layout/tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   Src Tensor    : layout::TensorNCxHWx<32>
///   Filter Tensor : layout::TensorCxRSKx<32>
///   Operator      : TensorOp class, for mma i8816
///
/// This uses the default warp-level operator given tile sizes
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Data type of accumulator
        typename ElementDst_,
        /// Layout of accumulator
        typename LayoutDst_,
        /// Operation performed by Convolution
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNCxHWx<32>, kAlignmentSrc, int8_t,
                      layout::TensorCxRSKx<32>, kAlignmentFilter, ElementDst_,
                      LayoutDst_, arch::OpClassTensorOp, 2, Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<32>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<32>;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 32;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    /// Default Operator
    using Operator = Operator_;

    /// Number of warps present
    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    // Divisility requirements
    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    /// Number of threads per warp
    static int const kWarpSize = gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    /// Size of a threadblock-scoped access
    static int const kAccessSizeInBits = 128;

    // Warp thread arrangement
    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;
    
    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess,
            false>;

    /// Transpose the ThreadMap of iterator Src
    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 1,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess,
            false>;

    /// Transpose the ThreadMap of iterator Filter
    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter, 0,
        SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level Tensor Op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            InstructionShape,  /// Instruction-level Gemm shape - concept
                               /// gemm::GemmShape
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            layout::RowMajor,  /// Layout of Dst
                               /// Tensor's matrix
                               /// (concept:
                               /// MatrixLayout)
            Operator,          /// Operator describing the tensor operation
            PartitionsK,       /// Number of partitions along K dimension
            AccumulatorsInRowMajor>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
        gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                     MatrixShape<0, 0>, WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace convolution
}  // namespace cutlass



