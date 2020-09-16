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
    \brief Template for a multistage GEMM kernel. Does not compute batching or
   support split-K.
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
        /// Element type for A matrix operand
        typename ElementA_,
        /// Layout type for A matrix operand
        typename LayoutA_,
        /// Element type for B matrix operand
        typename ElementB_,
        /// Layout type for B matrix operand
        typename LayoutB_,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Layout type for C and D matrix operands
        typename LayoutC_,
        /// Operator class tag
        typename OperatorClass_,
        /// Tag indicating architecture to tune for
        typename ArchTag_,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape_,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape_,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape_,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Complex transformation on operand A
        ComplexTransform TransformA = ComplexTransform::kNone,
        /// Complex transformation on operand B
        ComplexTransform TransformB = ComplexTransform::kNone,
        /// Multiply-add operator (arch::OpMultiplyAddComplex,
        /// arch::OpMultiplyGaussianComplex)
        typename Operator = arch::OpMultiplyAddComplex,
        /// Store the accumulators in row major or column major.  Row major is
        /// used when output layout is interleaved.
        bool AccumulatorsInRowMajor = false>
struct DefaultMultistageMmaComplex;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output
template <
        /// Element type for A matrix operand
        typename ElementA,
        /// Layout type for A matrix operand
        typename LayoutA,
        /// Element type for B matrix operand
        typename ElementB,
        /// Layout type for B matrix operand
        typename LayoutB,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Tag indicating architecture to tune for
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Number of stages used in the multistage mainloop
        int Stages,
        /// Complex transformation on operand A
        ComplexTransform TransformA,
        /// Complex transformation on operand B
        ComplexTransform TransformB,
        /// Multiply-add operator (arch::OpMultiplyAddComplex,
        /// arch::OpMultiplyGaussianComplex)
        typename Operator>
struct DefaultMultistageMmaComplex<
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
        layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
        InstructionShape, Stages, TransformA, TransformB, Operator> {
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::
            DefaultMultistageMmaComplexCore<
                    ThreadblockShape, WarpShape, InstructionShape, ElementA,
                    LayoutA, ElementB, LayoutB, ElementAccumulator,
                    layout::RowMajor, OperatorClass, Stages, TransformA,
                    TransformB, Operator>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA =
            cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB =
            cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
    using IteratorB =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
            MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
            typename MmaCore::MmaPolicy, Stages>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
