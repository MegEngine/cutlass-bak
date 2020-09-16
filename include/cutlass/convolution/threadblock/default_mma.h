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
/**
 * \file include/cutlass/convolution/threadblock/default_mma.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_ld_constant.h"

#include "cutlass/convolution/threadblock/default_mma_core.h"
#include "cutlass/convolution/threadblock/default_mma_core_simt.h"
#include "cutlass/convolution/threadblock/default_mma_core_sm75.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
        /// Element type for Src Tensor operand
        typename ElementSrc_,
        /// Layout type for Src Tensor operand
        typename LayoutSrc_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Element type for Filter Tensor operand
        typename ElementFilter_,
        /// Layout type for Filter Tensor operand
        typename LayoutFilter_,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst_,
        /// Operator class tag
        typename OperatorClass_,
        /// Tag indicating architecture to tune for
        typename ArchTag_,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape_,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape_,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape_,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation perfomed by GEMM
        typename Operator,
        /// Store the accumulators in row major or column major.  Row major is
        /// used when output layout is interleaved.
        bool AccumulatorsInRowMajor = false,
        /// Whether use special optimization for convolution 1x1
        bool NeedLoadFromConstMem = true>
struct DefaultMma;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT IDP4A Kernels with TensorCxRSKx<4> tensors
template <
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Operation performed by GEMM
        typename Operator>
struct DefaultMma<int8_t, layout::TensorCxRSKx<4>, kAlignmentSrc, int8_t,
                  layout::TensorCxRSKx<4>, kAlignmentFilter, ElementAccumulator,
                  layout::TensorCxRSKx<4>, arch::OpClassSimt, ArchTag,
                  ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, 2,
                  Operator, true> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    using LayoutDst = layout::TensorCxRSKx<4>;
    using OperatorClass = arch::OpClassSimt;

    // Define the MmaCore components
    using MmaCore = typename cutlass::convolution::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 1, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
            cutlass::transform::threadblock::TileMap<
                    LayoutSrc, cutlass::transform::threadblock::TileMapType::
                                       kRow2C_Col2N>>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementFilter, LayoutFilter, 1,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
                    cutlass::transform::threadblock::TileMap<
                            LayoutFilter, cutlass::transform::threadblock::
                                                  TileMapType::kRow2C_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::convolution::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT IDP4A Kernels with
/// Input Tensor: layout::NCxHWx<4>
/// Filter Tensor: layout::CxRSKx<4>
/// Output Tensor: layout::NCxHWx<4>
template <
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Operation performed by GEMM
        typename Operator, bool NeedLoadFromConstMem>
struct DefaultMma<int8_t, layout::TensorNCxHWx<4>, kAlignmentSrc, int8_t,
                  layout::TensorCxRSKx<4>, kAlignmentFilter, ElementAccumulator,
                  layout::TensorNCxHWx<4>, arch::OpClassSimt, ArchTag,
                  ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, 2,
                  Operator, true, NeedLoadFromConstMem> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    using LayoutDst = layout::TensorNCxHWx<4>;
    using OperatorClass = arch::OpClassSimt;

    // Define the MmaCore components
    using MmaCore = typename cutlass::convolution::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 1, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
            cutlass::transform::threadblock::TileMap<
                    LayoutSrc, cutlass::transform::threadblock::TileMapType::
                                       kRow2C_Col2NHW>,
            NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementFilter, LayoutFilter, 1,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
                    cutlass::transform::threadblock::TileMap<
                            LayoutFilter, cutlass::transform::threadblock::
                                                  TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma =
            cutlass::convolution::threadblock::MmaPrecomputeOffset<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT IDP4A Kernels with
/// Input Tensor: layout::NCxHWx<4>
/// Filter Tensor: layout::CxRSKx<4>
/// Output Tensor: layout::NCxHWx<32>
template <
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Operation performed by GEMM
        typename Operator, bool NeedLoadFromConstMem>
struct DefaultMma<int8_t, layout::TensorNCxHWx<4>, kAlignmentSrc, int8_t,
                  layout::TensorCxRSKx<4>, kAlignmentFilter, ElementAccumulator,
                  layout::TensorNCxHWx<32>, arch::OpClassSimt, ArchTag,
                  ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, 2,
                  Operator, true, NeedLoadFromConstMem> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    using LayoutDst = layout::TensorNCxHWx<32>;
    using OperatorClass = arch::OpClassSimt;

    // Define the MmaCore components
    using MmaCore = typename cutlass::convolution::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 1, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
            cutlass::transform::threadblock::TileMap<
                    LayoutSrc, cutlass::transform::threadblock::TileMapType::
                                       kRow2C_Col2NHW>,
            NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementFilter, LayoutFilter, 1,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
                    cutlass::transform::threadblock::TileMap<
                            LayoutFilter, cutlass::transform::threadblock::
                                                  TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma =
            cutlass::convolution::threadblock::MmaPrecomputeOffset<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for TensorOp(i8816) Kernels with
/// Input Tensor: layout::CxRSKx<4>
/// Filter Tensor: layout::CxRSKx<16>
/// Output Tensor: layout::CxRSKx<4>
template <
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Operation performed by GEMM
        typename Operator>
struct DefaultMma<int8_t, layout::TensorCxRSKx<4>, kAlignmentSrc, int8_t,
                  layout::TensorCxRSKx<16>, kAlignmentFilter,
                  ElementAccumulator, layout::TensorCxRSKx<4>,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, true> {
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<16>;
    using LayoutDst = layout::TensorCxRSKx<4>;
    using OperatorClass = arch::OpClassTensorOp;

    // Define the MmaCore components
    using MmaCore = typename cutlass::convolution::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 1, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
            cutlass::transform::threadblock::TileMap<
                    LayoutSrc, cutlass::transform::threadblock::TileMapType::
                                       kRow2C_Col2NHW>>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementFilter, LayoutFilter, 1,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
                    cutlass::transform::threadblock::TileMap<
                            LayoutFilter, cutlass::transform::threadblock::
                                                  TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma =
            cutlass::convolution::threadblock::MmaPrecomputeOffset<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for TensorOp(i8816) Kernels with
/// Input Tensor: layout::TensorNCxHWx<32>
/// Filter Tensor: layout::TensorCxRSKx<32>
/// Output Tensor: layout::TensorNCxHWx<32> or layout::TensorNCxHWx<4>
template <
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Layout type for Dst and Z Tensor operand
        typename LayoutDst,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Operation performed by GEMM
        typename Operator, bool NeedLoadFromConstMem>
struct DefaultMma<int8_t, layout::TensorNCxHWx<32>, kAlignmentSrc, int8_t,
                  layout::TensorCxRSKx<32>, kAlignmentFilter,
                  ElementAccumulator, LayoutDst, arch::OpClassTensorOp, ArchTag,
                  ThreadblockShape, WarpShape, InstructionShape, 2, Operator,
                  true, NeedLoadFromConstMem> {
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<32>;
    using ElementFilter = int8_t;
    using LayoutFilter = layout::TensorCxRSKx<32>;
    using OperatorClass = arch::OpClassTensorOp;

    // Define the MmaCore components
    using MmaCore = typename cutlass::convolution::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define iterators over tiles from the Src Tensor operand
    using TileMap = cutlass::transform::threadblock::TileMap<
            LayoutSrc,
            cutlass::transform::threadblock::TileMapType::kRow2C_Col2NHW>;
    using IteratorSrc = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 1, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess, TileMap,
            NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementFilter, LayoutFilter, 1,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
                    cutlass::transform::threadblock::TileMap<
                            LayoutFilter, cutlass::transform::threadblock::
                                                  TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma =
            cutlass::convolution::threadblock::MmaPrecomputeOffset<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace convolution
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
