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
    \brief Implements several possible threadblock-swizzling functions mapping
   blockIdx to Convolution problems.
*/

/**
 * \file include/cutlass/convolution/threadblock/threadblock_swizzle.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/convolution/convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace convolution {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for Convolution
template <ConvType ConvolutionType>
struct ConvolutionCxRSKxThreadblockSwizzle {
    typedef ConvParam<ConvolutionType> ConvolutionParameter;

    CUTLASS_HOST_DEVICE
    ConvolutionCxRSKxThreadblockSwizzle() {}

    /// Returns the shape of the problem in units of logical tiles
    CUTLASS_HOST_DEVICE
    gemm::GemmCoord get_tiled_shape(ConvolutionParameter const& conv_param,
                                    gemm::GemmCoord const& tile_size) const {
        return gemm::GemmCoord(
                conv_param.ho() * conv_param.wo(),
                (conv_param.n() + tile_size.n() - 1) / tile_size.n(),
                (conv_param.co() + tile_size.m() - 1) / tile_size.m());
    }

    /// Computes CUDA grid dimensions given a size in units of logical tiles
    CUTLASS_HOST_DEVICE
    dim3 get_grid_shape(gemm::GemmCoord const& tiled_shape) const {
        return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    }

    /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
    template <typename Shape>
    CUTLASS_DEVICE Tensor4DCoord
    get_tile_offset(ConvolutionParameter const& conv_param) const {
        int block_idx_x = cutlass::gemm::threadblock::RematerializeBlockIdxX();
        int block_idx_y = cutlass::gemm::threadblock::RematerializeBlockIdxY();
        int block_idx_z = cutlass::gemm::threadblock::RematerializeBlockIdxZ();
        int h = block_idx_x / conv_param.wo();
        int w = block_idx_x - conv_param.wo() * h;
        int c = block_idx_z * Shape::kM;
        int n = block_idx_y * Shape::kN;

        return Tensor4DCoord{n, h, w, c};
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for Convolution
template <ConvType ConvolutionType>
struct ConvolutionNCxHWxThreadblockSwizzle {
    typedef ConvParam<ConvolutionType> ConvolutionParameter;

    CUTLASS_HOST_DEVICE
    ConvolutionNCxHWxThreadblockSwizzle() {}

    /// Returns the shape of the problem in units of logical tiles
    CUTLASS_HOST_DEVICE
    gemm::GemmCoord get_tiled_shape(ConvolutionParameter const& conv_param,
                                    gemm::GemmCoord const& tile_size) const {
        return gemm::GemmCoord(
                (conv_param.n() * conv_param.ho() * conv_param.wo() +
                 tile_size.n() - 1) /
                        tile_size.n(),
                (conv_param.co() + tile_size.m() - 1) / tile_size.m(), 1);
    }

    /// Computes CUDA grid dimensions given a size in units of logical tiles
    CUTLASS_HOST_DEVICE
    dim3 get_grid_shape(gemm::GemmCoord const& tiled_shape) const {
        return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    }

    /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
    template <typename Shape>
    CUTLASS_DEVICE gemm::GemmCoord get_tile_offset() const {
        int block_idx_x = cutlass::gemm::threadblock::RematerializeBlockIdxX();
        int block_idx_y = cutlass::gemm::threadblock::RematerializeBlockIdxY();
        return gemm::GemmCoord(block_idx_y * Shape::kM, block_idx_x * Shape::kN,
                               1);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace convolution
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
