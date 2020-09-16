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
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Reference implementation for GEMM in host-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "./gemm.h"
#include "cutlass/arch/mma.h"
#include "cutlass/convolution/convolution.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/host_tensor.h"

namespace cutlass {
namespace reference {
namespace host {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general convolution among tensors of rank=4 pointed
/// to by TensorRef objects.
template <convolution::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverter<ElementDst, ScalarType>>
void compute_convolution(convolution::ConvParam<ConvolutionType> conv_param,
                         ScalarType alpha,
                         TensorRef<ElementSrc, LayoutSrc> tensor_src,
                         TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                         ScalarType beta,
                         TensorRef<ElementBias, LayoutBias> tensor_bias,
                         ScalarType gamma,
                         TensorRef<ElementDst, LayoutDst> tensor_z,
                         TensorRef<ElementDst, LayoutDst> tensor_dst,
                         ComputeType initial_accum) {
    static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                          LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                  "Tensors must be of rank 4");
    using TensorCoordSrc = typename LayoutSrc::TensorCoord;
    using TensorCoordFilter = typename LayoutFilter::TensorCoord;
    using TensorCoordBias = typename LayoutBias::TensorCoord;
    using TensorCoordDst = typename LayoutDst::TensorCoord;

    int const N = conv_param.n();
    int const IC = conv_param.ci();
    int const OC = conv_param.co();
    int const IH = conv_param.hi();
    int const IW = conv_param.wi();
    int const OH = conv_param.ho();
    int const OW = conv_param.wo();
    int const FH = conv_param.fh();
    int const FW = conv_param.fw();
    int const PH = conv_param.pad_h();
    int const PW = conv_param.pad_w();
    int const SH = conv_param.stride_h();
    int const SW = conv_param.stride_w();

    // Blocking necessary to speedup reference implementation
    int const Mblock = 16;
    int const Nblock = 16;

    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    for (int n_block = 0; n_block < N; n_block += Nblock) {
        for (int oc_block = 0; oc_block < OC; oc_block += Mblock) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    ComputeType accum[Mblock][Nblock];

                    for (int j = 0; j < Nblock; j++) {
                        for (int i = 0; i < Mblock; i++) {
                            accum[i][j] = initial_accum;
                        }
                    }

                    int ih_base = oh * SH - PH;
                    int iw_base = ow * SW - PW;

                    for (int fh = 0; fh < FH; fh++) {
                        for (int fw = 0; fw < FW; fw++) {
                            for (int ic = 0; ic < IC; ic++) {
                                for (int j = 0; j < Nblock; j++) {
                                    for (int i = 0; i < Mblock; i++) {
                                        int n = n_block + i;
                                        int oc = oc_block + j;

                                        int ih = ih_base + fh;
                                        int iw = iw_base + fw;
                                        if (n < N && oc < OC) {
                                            ElementSrc src;
                                            if (ih >= 0 && ih < IH && iw >= 0 &&
                                                iw < IW) {
                                                src = tensor_src.at(
                                                        TensorCoordSrc(n, ih,
                                                                       iw, ic));
                                            } else {
                                                src = 0;
                                            }
                                            ElementFilter filter =
                                                    tensor_filter.at(
                                                            TensorCoordFilter(
                                                                    oc, fh, fw,
                                                                    ic));

                                            ComputeType compute_src(
                                                    cast_if_scalar<ComputeType>(
                                                            src));
                                            ComputeType compute_filter(
                                                    cast_if_scalar<ComputeType>(
                                                            filter));

                                            accum[i][j] = inner_product_op(
                                                    compute_filter, compute_src,
                                                    accum[i][j]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int j = 0; j < Nblock; j++) {
                        for (int i = 0; i < Mblock; i++) {
                            int n = n_block + i;
                            int oc = oc_block + j;

                            TensorCoordDst coord(n, oh, ow, oc);
                            TensorCoordBias coord_bias(0, 0, 0, oc);

                            if (n < N && oc < OC) {
                                ScalarType intermediate = std::round(
                                        alpha * ScalarType(accum[i][j]) +
                                        beta * ScalarType(tensor_bias.at(
                                                       coord_bias)) +
                                        gamma * ScalarType(tensor_z.at(coord)));
                                tensor_dst.at(coord) = convert_op(intermediate);
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed
/// to by TensorRef objects.
template <convolution::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverter<ElementDst, ScalarType>>
void compute_convolution(convolution::ConvParam<ConvolutionType> conv_param,
                         ScalarType alpha,
                         TensorRef<ElementSrc, LayoutSrc> tensor_src,
                         TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                         ScalarType beta,
                         TensorRef<ElementBias, LayoutBias> tensor_bias,
                         TensorRef<ElementDst, LayoutDst> tensor_dst,
                         ComputeType initial_accum) {
    compute_convolution<ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                        ElementDst, LayoutDst, ScalarType, ComputeType,
                        InnerProductOp, ConvertOp>(
            conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias, 0,
            tensor_dst, tensor_dst, initial_accum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <convolution::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = cutlass::arch::OpMultiplyAdd>
struct Convolution;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <convolution::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType>
struct Convolution<ConvolutionType, ElementSrc, LayoutSrc, ElementFilter,
                   LayoutFilter, ElementDst, LayoutDst, ElementBias, LayoutBias,
                   ScalarType, ComputeType, arch::OpMultiplyAdd> {
    void operator()(convolution::ConvParam<ConvolutionType> conv_param,
                    ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            multiply_add<ComputeType>>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(convolution::ConvParam<ConvolutionType> conv_param,
                    ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            multiply_add<ComputeType>>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add-saturate
template <convolution::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType>
struct Convolution<ConvolutionType, ElementSrc, LayoutSrc, ElementFilter,
                   LayoutFilter, ElementDst, LayoutDst, ElementBias, LayoutBias,
                   ScalarType, ComputeType, arch::OpMultiplyAddSaturate> {
    void operator()(convolution::ConvParam<ConvolutionType> conv_param,
                    ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            multiply_add<ComputeType>,
                            NumericConverterClamp<ElementDst, ScalarType>>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(convolution::ConvParam<ConvolutionType> conv_param,
                    ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            multiply_add<ComputeType>,
                            NumericConverterClamp<ElementDst, ScalarType>>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for XOR-popc
template <convolution::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType>
struct Convolution<ConvolutionType, ElementSrc, LayoutSrc, ElementFilter,
                   LayoutFilter, ElementDst, LayoutDst, ElementBias, LayoutBias,
                   ScalarType, ComputeType, arch::OpXorPopc> {
    void operator()(convolution::ConvParam<ConvolutionType> conv_param,
                    ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            xor_add<ComputeType>>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(convolution::ConvParam<ConvolutionType> conv_param,
                    ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            xor_add<ComputeType>>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace host
}  // namespace reference
}  // namespace cutlass
