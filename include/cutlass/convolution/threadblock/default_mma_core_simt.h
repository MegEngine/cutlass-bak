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
#include "cutlass/layout/tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace threadblock {

namespace detail {

// convert a WarpShape which is the whole tile of elements into warp num
// threads. The goal is for each thread's tile of elements to be as square as
// possible for performance (4x4 will be faster than 2x8).
template <typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}

}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   A: layout::TensorCxRSKx<4>
///   B: layout::TensorCxRSKx<4>
///   Operator: simt class, for dp4a
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<1, 1, 4>, int8_t,
                      layout::TensorCxRSKx<4>, kAlignmentSrc, int8_t,
                      layout::TensorCxRSKx<4>, kAlignmentFilter, ElementDst_,
                      LayoutDst_, arch::OpClassSimt, 2, Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

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
    static int const kWarpSize = gemm::warp::WarpSize<arch::OpClassSimt>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajorInterleaved<4>;
    using SmemLayoutFilter = layout::ColumnMajorInterleaved<4>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kN * 4, Shape::kK / 4>, kThreads,
            kAlignmentSrc>;

    using SmemThreadMapSrc = IteratorThreadMapSrc;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 0,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kM * 4, Shape::kK / 4>, kThreads,
            kAlignmentFilter>;

    using SmemThreadMapFilter = IteratorThreadMapFilter;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            1, SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level op
    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsSrc = 128 / sizeof_bits<ElementSrc>::value;
    static const int numElementsFilter =
            128 / sizeof_bits<ElementFilter>::value;
    static const int LaneM = cutlass::const_min(4, ThreadTileM);
    static const int LaneN = cutlass::const_min(4, ThreadTileN);
    // these should have max of thread tile also
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 4>;

    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,                // WarpShape
            cutlass::layout::ColumnMajorInterleaved<LaneLayout>,  // LaneLayout
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            layout::RowMajor,  /// Layout of Dst Tensor's matrix (concept:
                               /// MatrixLayout)
            Policy,      /// Policy describing warp-level MmaSimtOp (concept:
                         /// MmaSimtOp policy)
            PartitionsK  /// Number of partitions along K dimension
            >;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>,
                                         WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   A: layout::TensorNCxHWx<4>
///   B: layout::TensorCxRSKx<4>
///   Operator: simt class, for dp4a
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<1, 1, 4>, int8_t,
                      layout::TensorNCxHWx<4>, kAlignmentSrc, int8_t,
                      layout::TensorCxRSKx<4>, kAlignmentFilter, ElementDst_,
                      LayoutDst_, arch::OpClassSimt, 2, Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

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
    static int const kWarpSize = gemm::warp::WarpSize<arch::OpClassSimt>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajorInterleaved<4>;
    using SmemLayoutFilter = layout::ColumnMajorInterleaved<4>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kN * 4, Shape::kK / 4>, kThreads,
        kAlignmentSrc>;

    using SmemThreadMapSrc = IteratorThreadMapSrc;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 0,
        SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kM * 4, Shape::kK / 4>, kThreads,
        kAlignmentFilter>;

    using SmemThreadMapFilter = IteratorThreadMapFilter;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter, 1,
        SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level op
    static const int WarpNumThreadsM =
        detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                      !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsSrc = 128 / sizeof_bits<ElementSrc>::value;
    static const int numElementsFilter =
        128 / sizeof_bits<ElementFilter>::value;
    static const int LaneM = cutlass::const_min(4, ThreadTileM);
    static const int LaneN = cutlass::const_min(4, ThreadTileN);
    // these should have max of thread tile also
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 4>;

    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
        cutlass::MatrixShape<WarpNumThreadsM,
                             WarpNumThreadsN>,                // WarpShape
        cutlass::layout::ColumnMajorInterleaved<LaneLayout>,  // LaneLayout
        LaneMmaShape>;

    using LayoutFragmentC = typename cutlass::platform::conditional<
            cutlass::platform::is_same<LayoutDst,
                                       layout::TensorNCxHWx<32>>::value,
            layout::ColumnMajor, layout::RowMajor>::type;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            LayoutFragmentC,   /// Layout of Dst Tensor's matrix (concept:
                               /// MatrixLayout)
            Policy,      /// Policy describing warp-level MmaSimtOp (concept:
                         /// MmaSimtOp policy)
            PartitionsK  /// Number of partitions along K dimension
            >;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
        gemm::threadblock::MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                     MatrixShape<0, 0>, WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace convolution
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
