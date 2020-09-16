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
 * \file dnn/src/cuda/cutlass/convolution/convolution.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/coord.h"
#include "cutlass/cutlass.h"

namespace cutlass {
namespace convolution {

enum class ConvType : uint32_t {
    kConvolution = 0,
    kBatchConvolution = 1,
    kLocal = 2,
    kLocalShare = 3,
};

struct ConvParamBase {
    typedef int Index;
    typedef Coord<2, Index> Vector;

    Index N, IC, OC, HW, NHW;
    Vector src_shape, filter_shape, dst_shape;
    Vector stride, padding, dilate;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    ConvParamBase()
            : N(0),
              IC(0),
              OC(0),
              HW(0),
              NHW(0),
              src_shape(Vector()),
              filter_shape(Vector()),
              dst_shape(Vector()),
              stride(Vector()),
              padding(Vector()),
              dilate(Vector()) {}

    /// Constructs from batch, input channels, output channels and Vector
    /// parameters
    CUTLASS_HOST_DEVICE
    ConvParamBase(Index n_, Index ic, Index oc, Vector const& src,
                  Vector const& filter, Vector const& dst,
                  Vector const& stride_, Vector const& padding_,
                  Vector const& dilate_)
            : N(n_),
              IC(ic),
              OC(oc),
              HW(dst[0] * dst[1]),
              NHW(N * HW),
              src_shape(make_Coord(src[0], src[1])),
              filter_shape(make_Coord(filter[0], filter[1])),
              dst_shape(make_Coord(dst[0], dst[1])),
              stride(make_Coord(stride_[0], stride_[1])),
              padding(make_Coord(padding_[0], padding_[1])),
              dilate(make_Coord(dilate_[0], dilate_[1])) {}

    /// Constructs from Index variables
    CUTLASS_HOST_DEVICE
    ConvParamBase(Index n_, Index ic, Index oc, Index ih, Index iw, Index fh_,
                  Index fw_, Index oh, Index ow, Index sh, Index sw, Index ph,
                  Index pw, Index dh, Index dw)
            : N(n_),
              IC(ic),
              OC(oc),
              HW(oh * ow),
              NHW(N * HW),
              src_shape(make_Coord(ih, iw)),
              filter_shape(make_Coord(fh_, fw_)),
              dst_shape(make_Coord(oh, ow)),
              stride(make_Coord(sh, sw)),
              padding(make_Coord(ph, pw)),
              dilate(make_Coord(dh, dw)) {}

    CUTLASS_HOST_DEVICE
    Index const& n() const { return N; }

    CUTLASS_HOST_DEVICE
    Index& n() { return N; }

    CUTLASS_HOST_DEVICE
    Index const& ci() const { return IC; }

    CUTLASS_HOST_DEVICE
    Index& ci() { return IC; }

    CUTLASS_HOST_DEVICE
    Index const& co() const { return OC; }

    CUTLASS_HOST_DEVICE
    Index& co() { return OC; }

    CUTLASS_HOST_DEVICE
    Index const& hi() const { return src_shape.at(0); }

    CUTLASS_HOST_DEVICE
    Index& hi() { return src_shape.at(0); }

    CUTLASS_HOST_DEVICE
    Index const& wi() const { return src_shape.at(1); }

    CUTLASS_HOST_DEVICE
    Index& wi() { return src_shape.at(1); }

    CUTLASS_HOST_DEVICE
    Index const& fh() const { return filter_shape.at(0); }

    CUTLASS_HOST_DEVICE
    Index& fh() { return filter_shape.at(0); }

    CUTLASS_HOST_DEVICE
    Index const& fw() const { return filter_shape.at(1); }

    CUTLASS_HOST_DEVICE
    Index& fw() { return filter_shape.at(1); }

    CUTLASS_HOST_DEVICE
    Index const& ho() const { return dst_shape.at(0); }

    CUTLASS_HOST_DEVICE
    Index& ho() { return dst_shape.at(0); }

    CUTLASS_HOST_DEVICE
    Index const& wo() const { return dst_shape.at(1); }

    CUTLASS_HOST_DEVICE
    Index& wo() { return dst_shape.at(1); }

    CUTLASS_HOST_DEVICE
    Index const& hw() const { return HW; }

    CUTLASS_HOST_DEVICE
    Index& hw() { return HW; }

    CUTLASS_HOST_DEVICE
    Index const& nhw() const { return NHW; }

    CUTLASS_HOST_DEVICE
    Index& nhw() { return NHW; }

    CUTLASS_HOST_DEVICE
    Index const& stride_h() const { return stride.at(0); }

    CUTLASS_HOST_DEVICE
    Index& stride_h() { return stride.at(0); }

    CUTLASS_HOST_DEVICE
    Index const& stride_w() const { return stride.at(1); }

    CUTLASS_HOST_DEVICE
    Index& stride_w() { return stride.at(1); }

    CUTLASS_HOST_DEVICE
    Index const& pad_h() const { return padding.at(0); }

    CUTLASS_HOST_DEVICE
    Index& pad_h() { return padding.at(0); }

    CUTLASS_HOST_DEVICE
    Index const& pad_w() const { return padding.at(1); }

    CUTLASS_HOST_DEVICE
    Index& pad_w() { return padding.at(1); }

    CUTLASS_HOST_DEVICE
    Index const& dilate_h() const { return dilate.at(0); }

    CUTLASS_HOST_DEVICE
    Index& dilate_h() { return dilate.at(0); }

    CUTLASS_HOST_DEVICE
    Index const& dilate_w() const { return dilate.at(1); }

    CUTLASS_HOST_DEVICE
    Index& dilate_w() { return dilate.at(1); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <ConvType conv_type_>
struct ConvParam : public ConvParamBase {
    static ConvType const conv_type = conv_type_;

    using Base = ConvParamBase;

    CUTLASS_HOST_DEVICE
    ConvParam() : Base() {}

    /// Constructs from batch, input channels, output channels and Vector
    /// parameters
    CUTLASS_HOST_DEVICE
    ConvParam(Index n_, Index ic, Index oc, Vector const& src,
              Vector const& filter, Vector const& dst, Vector const& stride_,
              Vector const& padding_, Vector const& dilate_)
            : Base(n_, ic, oc, src, filter, dst, stride_, padding_, dilate_) {}

    /// Constructs from Index variables
    CUTLASS_HOST_DEVICE
    ConvParam(Index n_, Index ic, Index oc, Index ih, Index iw, Index fh_,
              Index fw_, Index oh, Index ow, Index sh, Index sw, Index ph,
              Index pw, Index dh, Index dw)
            : Base(n_, ic, oc, ih, iw, fh_, fw_, oh, ow, sh, sw, ph, pw, dh,
                   dw) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace convolution
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
