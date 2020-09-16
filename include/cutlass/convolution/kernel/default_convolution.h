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
 * \file include/cutlass/convolution/kernel/default_convolution.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/convolution/threadblock/default_mma.h"
#include "cutlass/convolution/threadblock/default_mma_core_simt.h"
#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/convolution_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/convolution_epilogue_tensor_op.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif  // CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace kernel {

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
        /// Element type for Dst and Z Tensor operands
        typename ElementDst_,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst_,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Convolution Type
        ConvType ConvolutionType,
        /// Operator class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by GEMM
        typename Operator,
        /// Whether use special optimization for conv1x1
        bool NeedLoadFromConstMem = true>
struct DefaultConvolution;

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for SIMT DP4A

template <
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Layout type for Dst and Z Tensor operand
        typename LayoutDst,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Convolution Type
        ConvType ConvolutionType,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Operation performed by GEMM
        typename Operator>
struct DefaultConvolution<int8_t, layout::TensorCxRSKx<4>, kAlignmentSrc,
                          int8_t, layout::TensorCxRSKx<4>, kAlignmentFilter,
                          ElementDst, LayoutDst, ElementAccumulator,
                          ConvolutionType, arch::OpClassSimt, ArchTag,
                          ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>,
                          EpilogueOutputOp, ThreadblockSwizzle, 2, Operator> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using LayoutFilter = layout::TensorCxRSKx<4>;

    using OperatorClass = arch::OpClassSimt;
    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::convolution::threadblock::DefaultMma<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, arch::OpClassSimt,
            arch::Sm50, ThreadblockShape, WarpShape, InstructionShape, 2,
            Operator, true>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = 4;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level Convolution operator.
    using ConvolutionKernel =
            kernel::Convolution<Mma, Epilogue, ThreadblockSwizzle,
                                ConvolutionType>;
};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW4 layout

template <int kAlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int kAlignmentFilter,
          /// Layout type for Dst and Z Tensor operand
          typename LayoutDst,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Tag indicating architecture to tune for
          typename ArchTag,
          /// Element type for internal accumulation
          typename ElementAccumulator,
          /// Convolution Type
          ConvType ConvolutionType,
          /// Threadblock-level tile size (concept: gemm::GemmShape)
          typename ThreadblockShape,
          /// Warp-level tile size (concept: gemm::GemmShape)
          typename WarpShape,
          /// Epilogue output operator
          typename EpilogueOutputOp,
          /// Threadblock-level swizzling operator
          typename ThreadblockSwizzle,
          /// Operation performed by GEMM
          typename Operator, bool NeedLoadFromConstMem>
struct DefaultConvolution<
        int8_t, layout::TensorNCxHWx<4>, kAlignmentSrc, int8_t,
        layout::TensorCxRSKx<4>, kAlignmentFilter, ElementDst, LayoutDst,
        ElementAccumulator, ConvolutionType, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, EpilogueOutputOp,
        ThreadblockSwizzle, 2, Operator, NeedLoadFromConstMem> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using LayoutFilter = layout::TensorCxRSKx<4>;

    using OperatorClass = arch::OpClassSimt;
    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::convolution::threadblock::DefaultMma<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, arch::OpClassSimt,
            arch::Sm50, ThreadblockShape, WarpShape, InstructionShape, 2,
            Operator, true, NeedLoadFromConstMem>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = 4;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level Convolution operator.
    using ConvolutionKernel = kernel::ConvolutionPrecomputeOffset<
            Mma, Epilogue, ThreadblockSwizzle, ConvolutionType>;
};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for CHWN4 layout

template <int kAlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int kAlignmentFilter,
          /// Layout type for Dst and Z Tensor operand
          typename LayoutDst,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Element type for internal accumulation
          typename ElementAccumulator,
          /// Convolution Type
          ConvType ConvolutionType,
          /// Threadblock-level tile size (concept: gemm::GemmShape)
          typename ThreadblockShape,
          /// Warp-level tile size (concept: gemm::GemmShape)
          typename WarpShape,
          /// Instruction-level tile size (concept: gemm::GemmShape)
          typename InstructionShape,
          /// Epilogue output operator
          typename EpilogueOutputOp,
          /// Threadblock-level swizzling operator
          typename ThreadblockSwizzle,
          /// Operation performed by GEMM
          typename Operator>
struct DefaultConvolution<int8_t, layout::TensorCxRSKx<4>, kAlignmentSrc,
                          int8_t, layout::TensorCxRSKx<16>, kAlignmentFilter,
                          ElementDst, LayoutDst, ElementAccumulator,
                          ConvolutionType, arch::OpClassTensorOp, arch::Sm75,
                          ThreadblockShape, WarpShape, InstructionShape,
                          EpilogueOutputOp, ThreadblockSwizzle, 2, Operator> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using LayoutFilter = layout::TensorCxRSKx<16>;

    using OperatorClass = arch::OpClassTensorOp;
    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::convolution::threadblock::DefaultMma<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst,
            arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
            InstructionShape, 2, Operator, true>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = 4;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level Convolution operator.
    using ConvolutionKernel =
            kernel::Convolution<Mma, Epilogue, ThreadblockSwizzle,
                                ConvolutionType>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout

template <int kAlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int kAlignmentFilter,
          /// Layout type for Dst and Z Tensor operand
          typename LayoutDst,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Element type for internal accumulation
          typename ElementAccumulator,
          /// Convolution Type
          ConvType ConvolutionType,
          /// Threadblock-level tile size (concept: gemm::GemmShape)
          typename ThreadblockShape,
          /// Warp-level tile size (concept: gemm::GemmShape)
          typename WarpShape,
          /// Instruction-level tile size (concept: gemm::GemmShape)
          typename InstructionShape,
          /// Epilogue output operator
          typename EpilogueOutputOp,
          /// Threadblock-level swizzling operator
          typename ThreadblockSwizzle,
          /// Interleaving quantity
          int Interleaved,
          /// Operation performed by GEMM
          typename Operator, bool NeedLoadFromConstMem>
struct DefaultConvolution<
        int8_t, layout::TensorNCxHWx<Interleaved>, kAlignmentSrc, int8_t,
        layout::TensorCxRSKx<Interleaved>, kAlignmentFilter, ElementDst,
        LayoutDst, ElementAccumulator, ConvolutionType, arch::OpClassTensorOp,
        arch::Sm75, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, 2, Operator, NeedLoadFromConstMem> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;

    using OperatorClass = arch::OpClassTensorOp;
    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::convolution::threadblock::DefaultMma<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst,
            arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
            InstructionShape, 2, Operator, true, NeedLoadFromConstMem>::ThreadblockMma;

    /// 64 bit store
    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level Convolution operator.
    using ConvolutionKernel = kernel::ConvolutionPrecomputeOffset<
            Mma, Epilogue, ThreadblockSwizzle, ConvolutionType>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout

template <int kAlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int kAlignmentFilter,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Element type for internal accumulation
          typename ElementAccumulator,
          /// Convolution Type
          ConvType ConvolutionType,
          /// Threadblock-level tile size (concept: gemm::GemmShape)
          typename ThreadblockShape,
          /// Warp-level tile size (concept: gemm::GemmShape)
          typename WarpShape,
          /// Instruction-level tile size (concept: gemm::GemmShape)
          typename InstructionShape,
          /// Epilogue output operator
          typename EpilogueOutputOp,
          /// Threadblock-level swizzling operator
          typename ThreadblockSwizzle,
          /// Interleaving quantity
          int Interleaved,
          /// Operation performed by GEMM
          typename Operator, bool NeedLoadFromConstMem>
struct DefaultConvolution<
        int8_t, layout::TensorNCxHWx<Interleaved>, kAlignmentSrc, int8_t,
        layout::TensorCxRSKx<Interleaved>, kAlignmentFilter, ElementDst,
        layout::TensorNCxHWx<4>, ElementAccumulator, ConvolutionType,
        arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, 2, Operator,
        NeedLoadFromConstMem> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<4>;

    using OperatorClass = arch::OpClassTensorOp;
    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::convolution::threadblock::DefaultMma<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst,
            arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
            InstructionShape, 2, Operator, true,
            NeedLoadFromConstMem>::ThreadblockMma;

    /// 32 bit store
    static int const kEpilogueElementsPerAccess =
            32 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level Convolution operator.
    using ConvolutionKernel = kernel::ConvolutionPrecomputeOffset<
            Mma, Epilogue, ThreadblockSwizzle, ConvolutionType>;
};

////////////////////////////////////////////////////////////////////////////////


}  // namespace kernel
}  // namespace convolution
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
