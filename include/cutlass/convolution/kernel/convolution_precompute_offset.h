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
    \brief Template for a pipelined Convolution kernel.
 */
/**
 * \file include/cutlass/convolution/kernel/convolution_precompute_offset.h
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

#include "cutlass/convolution/convolution.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          ConvType ConvolutionType =
                  ConvType::kConvolution  ///! Convolution Type
          >
struct ConvolutionPrecomputeOffset {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static const ConvType kConvolutionType = ConvolutionType;
    using ConvolutionParameter = ConvParam<kConvolutionType>;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    /// Parameters structure
    struct Params {
        ConvolutionParameter conv_param;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        typename Mma::IteratorSrc::Params params_src;
        typename Mma::IteratorSrc::TensorRef ref_src;
        typename Mma::IteratorFilter::Params params_filter;
        typename Mma::IteratorFilter::TensorRef ref_filter;
        typename Epilogue::BiasTileIterator::Params params_bias;
        typename Epilogue::BiasTileIterator::TensorRef ref_bias;
        typename Epilogue::OutputTileIterator::Params params_dst;
        typename Epilogue::OutputTileIterator::TensorRef ref_dst;
        typename Epilogue::OutputTileIterator::Params params_z;
        typename Epilogue::OutputTileIterator::TensorRef ref_z;
        typename OutputOp::Params output_op;
        typename Mma::TransformSrc::Params transform_src;
        typename Mma::TransformFilter::Params transform_filter;

        cutlass::gemm::GemmCoord gemm_problem_size;
        int* workspace;
        int conv_k_iterations;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params() : conv_k_iterations(0) {}

        CUTLASS_HOST_DEVICE
        Params(ConvolutionParameter const& conv_param,
               cutlass::gemm::GemmCoord const& grid_tiled_shape,
               typename Mma::IteratorSrc::TensorRef ref_src,
               typename Mma::IteratorFilter::TensorRef ref_filter,
               typename Epilogue::BiasTileIterator::TensorRef ref_bias,
               typename Epilogue::OutputTileIterator::TensorRef ref_z,
               typename Epilogue::OutputTileIterator::TensorRef ref_dst,
               typename OutputOp::Params output_op =
                       typename OutputOp::Params(),
               typename Mma::TransformSrc::Params transform_src =
                       typename Mma::TransformSrc::Params(),
               typename Mma::TransformFilter::Params transform_filter =
                       typename Mma::TransformFilter::Params(),
               int* workspace_ = nullptr)
                : conv_param(conv_param),
                  grid_tiled_shape(grid_tiled_shape),
                  params_src(ref_src.layout(),
                             static_cast<ConvParamBase>(conv_param)),
                  ref_src(ref_src),
                  params_filter(ref_filter.layout()),
                  ref_filter(ref_filter),
                  params_bias(ref_bias.layout()),
                  ref_bias(ref_bias),
                  params_dst(ref_dst.layout(),
                             static_cast<ConvParamBase>(conv_param)),
                  ref_dst(ref_dst),
                  params_z(ref_z.layout(),
                           static_cast<ConvParamBase>(conv_param)),
                  ref_z(ref_z),
                  output_op(output_op),
                  transform_src(transform_src),
                  transform_filter(transform_filter),
                  gemm_problem_size{
                          conv_param.co(),
                          conv_param.n() * conv_param.ho() * conv_param.wo(),
                          conv_param.ci() * conv_param.fh() * conv_param.fw()},
                  workspace(workspace_) {
            conv_k_iterations = (gemm_problem_size.k() + Mma::Shape::kK - 1) /
                                Mma::Shape::kK;
        }
    };

    /// Shared memory storage structure
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    ConvolutionPrecomputeOffset() {}

    /// Determines whether kernel satisfies alignment
    static Status can_implement(
            ConvolutionParameter conv_param,
            typename Mma::IteratorSrc::TensorRef ref_src,
            typename Mma::IteratorFilter::TensorRef ref_filter,
            typename Epilogue::BiasTileIterator::TensorRef ref_bias,
            typename Epilogue::OutputTileIterator::TensorRef ref_z,
            typename Epilogue::OutputTileIterator::TensorRef ref_dst) {
        static int const kAlignmentSrc =
                Mma::IteratorSrc::AccessType::kElements;
        static int const kAlignmentFilter =
                Mma::IteratorFilter::AccessType::kElements;
        static int const kAlignmentDst =
                Epilogue::OutputTileIterator::kElementsPerAccess;

        if (!TensorRef_aligned(ref_src, kAlignmentSrc)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_filter, kAlignmentFilter)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_dst, kAlignmentDst)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_z, kAlignmentDst)) {
            return Status::kErrorMisalignedOperand;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(ConvolutionParameter conv_param) {
        return 0;
    }

    /// Executes one Convolution
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage) {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_offset =
                threadblock_swizzle.template get_tile_offset<Mma::Shape>();

        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_src{0, threadblock_tile_offset.n()};

        cutlass::MatrixCoord tb_offset_filter{threadblock_tile_offset.m(), 0};

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to Src and Filter Tensor operands
        typename Mma::IteratorSrc iterator_src(
                params.params_src, params.ref_src.data(),
                {params.gemm_problem_size.k(), params.gemm_problem_size.n()},
                thread_idx, tb_offset_src);

        typename Mma::IteratorFilter iterator_filter(
                params.params_filter, params.ref_filter.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.k()},
                thread_idx, tb_offset_filter);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentDst accumulators;

        accumulators.clear();

        mma(params.conv_k_iterations, accumulators, iterator_src,
            iterator_filter, accumulators, params.transform_src,
            params.transform_filter);

        //
        // Epilogue
        //

        OutputOp output_op(params.output_op);

        //
        // Masked tile iterators constructed from members
        //
        // assume identity swizzle
        cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m(),
                                                threadblock_tile_offset.n());

        // Tile iterator load bias tensor
        typename Epilogue::BiasTileIterator iterator_bias(
                params.params_bias, params.ref_bias.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, threadblock_offset);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_dst(
                params.params_dst, params.ref_dst.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, threadblock_offset);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_z(
                params.params_z, params.ref_z.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, threadblock_offset);

        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_dst, accumulators, iterator_bias,
                 iterator_z);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace convolution
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
