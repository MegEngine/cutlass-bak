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
 * \file include/cutlass/convolution/device/convolution.h
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
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"

#include "cutlass/convolution/convolution.h"
#include "cutlass/convolution/kernel/convolution.h"
#include "cutlass/convolution/kernel/convolution_precompute_offset.h"
#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/convolution/device/default_convolution_configuration.h"
#include "cutlass/convolution/kernel/default_convolution.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Convolution device-level operator.
 */
template <
        /// Element type for Src Tensor operand
        typename ElementSrc_,
        /// Layout type for Src Tensor operand
        typename LayoutSrc_,
        /// Element type for Filter Tensor operand
        typename ElementFilter_,
        /// Layout type for Filter Tensor operand
        typename LayoutFilter_,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst_,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst_,
        /// Element type for Bias Tensor operands
        typename ElementBias_,
        /// Layout type for Bias Tensor operands
        typename LayoutBias_,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Convolution Type
        ConvType ConvolutionType = ConvType::kConvolution,
        /// Operator class tag
        typename OperatorClass_ = arch::OpClassSimt,
        /// Tag indicating architecture to tune for
        typename ArchTag_ = arch::Sm61,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle_ = typename threadblock::
                ConvolutionCxRSKxThreadblockSwizzle<ConvolutionType>,
        /// Number of stages used in the pipelined mainloop
        int Stages = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kStages,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentSrc = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int AlignmentFilter = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentFilter,
        /// whether use special optimization for convolution 1x1
        bool NeedLoadFromConstMem = true,
        /// Operation performed by Convolution
        typename Operator_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::Operator>
class Convolution {
public:
    using ElementSrc = ElementSrc_;
    using LayoutSrc = LayoutSrc_;
    using TensorRefSrc = TensorRef<ElementSrc const, LayoutSrc>;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = LayoutFilter_;
    using TensorRefFilter = TensorRef<ElementFilter const, LayoutFilter>;
    using ElementBias = ElementBias_;
    using LayoutBias = LayoutBias_;
    using TensorRefBias = TensorRef<ElementBias const, LayoutBias>;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using TensorRefDst = TensorRef<ElementDst const, LayoutDst>;
    using TensorRefZ = TensorRef<ElementDst, LayoutDst>;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static const ConvType kConvolutionType = ConvolutionType;
    using ConvolutionParameter = ConvParam<kConvolutionType>;
    static int const kStages = Stages;
    static int const kAlignmentSrc = AlignmentSrc;
    static int const kAlignmentFilter = AlignmentFilter;
    static int const kAlignmentDst = EpilogueOutputOp::kCount;
    static bool const kNeedLoadFromConstMem = NeedLoadFromConstMem;

    /// Define the kernel
    using ConvolutionKernel = typename kernel::DefaultConvolution<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementDst, LayoutDst, ElementAccumulator,
            kConvolutionType, OperatorClass, ArchTag, ThreadblockShape,
            WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
            kStages, Operator, kNeedLoadFromConstMem>::ConvolutionKernel;
    using TransformSrc = typename ConvolutionKernel::Mma::TransformSrc;
    using TransformFilter = typename ConvolutionKernel::Mma::TransformFilter;

    /// Argument structure
    struct Arguments {
        ConvolutionParameter conv_param;
        TensorRef<ElementSrc const, LayoutSrc> ref_src;
        TensorRef<ElementFilter const, LayoutFilter> ref_filter;
        TensorRef<ElementBias const, LayoutBias> ref_bias;
        TensorRef<ElementDst const, LayoutDst> ref_z;
        TensorRef<ElementDst, LayoutDst> ref_dst;
        typename EpilogueOutputOp::Params epilogue;
        typename TransformSrc::Params transform_src;
        typename TransformFilter::Params transform_filter;

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments() : conv_param(ConvolutionParameter()) {}

        /// Constructs an Arguments structure
        CUTLASS_HOST_DEVICE
        Arguments(ConvolutionParameter conv_param_,
                  TensorRef<ElementSrc const, LayoutSrc> ref_src_,
                  TensorRef<ElementFilter const, LayoutFilter> ref_filter_,
                  TensorRef<ElementBias const, LayoutBias> ref_bias_,
                  TensorRef<ElementDst const, LayoutDst> ref_z_,
                  TensorRef<ElementDst, LayoutDst> ref_dst_,
                  typename EpilogueOutputOp::Params epilogue_ =
                          typename EpilogueOutputOp::Params(),
                  typename TransformSrc::Params transform_src_ =
                          typename TransformSrc::Params(),
                  typename TransformFilter::Params transform_filter_ =
                          typename TransformFilter::Params())
                : conv_param(conv_param_),
                  ref_src(ref_src_),
                  ref_filter(ref_filter_),
                  ref_bias(ref_bias_),
                  ref_z(ref_z_),
                  ref_dst(ref_dst_),
                  epilogue(epilogue_),
                  transform_src(transform_src_),
                  transform_filter(transform_filter_) {}
    };

private:
    /// Kernel parameters object
    typename ConvolutionKernel::Params params_;

public:
    /// Constructs the GEMM.
    Convolution() {}

    /// Determines whether the GEMM can execute the given problem.
    static Status can_implement(Arguments const& args) {
        Status status = ConvolutionKernel::can_implement(
                args.conv_param, args.ref_src.non_const_ref(),
                args.ref_filter.non_const_ref(), args.ref_bias.non_const_ref(),
                args.ref_z.non_const_ref(), args.ref_dst);

        if (status != Status::kSuccess) {
            return status;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args) {
        return ConvolutionKernel::get_workspace_size(args.conv_param);
    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        // Determine grid shape
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord grid_shape =
                threadblock_swizzle.get_tiled_shape(
                        args.conv_param,
                        {ThreadblockShape::kM, ThreadblockShape::kN,
                         ThreadblockShape::kK});

        // Initialize the Params structure
        params_ = typename ConvolutionKernel::Params{
                args.conv_param,
                grid_shape,
                args.ref_src.non_const_ref(),
                args.ref_filter.non_const_ref(),
                args.ref_bias.non_const_ref(),
                args.ref_z.non_const_ref(),
                args.ref_dst,
                args.epilogue,
                args.transform_src,
                args.transform_filter,
                static_cast<int*>(workspace)};

        return Status::kSuccess;
    }

    /// Lightweight update given a subset of arguments
    Status update(Arguments const& args, void* workspace = nullptr) {
        params_.ref_src.reset(args.ref_src.non_const_ref().data());
        params_.ref_filter.reset(args.ref_filter.non_const_ref().data());
        params_.ref_bias.reset(args.ref_bias.non_const_ref().data());
        params_.ref_z.reset(args.ref_z.non_const_ref().data());
        params_.ref_dst.reset(args.ref_dst.data());
        params_.output_op = args.epilogue;
        params_.transform_src = args.transform_src;
        params_.transform_filter = args.transform_filter;
        params_.workspace = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(ConvolutionKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvolutionKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<ConvolutionKernel>
                <<<grid, block, smem_size, stream>>>(params_);

        result = cudaGetLastError();

        return result == cudaSuccess ? Status::kSuccess
                                     : Status::kErrorInternal;
    }

    /// Runs the kernel using initialized state.
    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    /// Runs the kernel using initialized state.
    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace convolution
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
