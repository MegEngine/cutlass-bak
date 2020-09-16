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
    \brief Tests for device-wide GEMM interface
*/

/**
 * \file test/unit/convolution/device/testbed.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

namespace test {
namespace convolution {
namespace device {
namespace {
/////////////////////////////////////////////////////////////////////////////////////////////////

inline char const* to_string(cutlass::Status status) {
    switch (status) {
        case cutlass::Status::kSuccess:
            return "kSuccess";
        case cutlass::Status::kErrorMisalignedOperand:
            return "kErrorMisalignedOperand";
        case cutlass::Status::kErrorInvalidLayout:
            return "kErrorInvalidLayout";
        case cutlass::Status::kErrorInvalidProblem:
            return "kErrorInvalidProblem";
        case cutlass::Status::kErrorNotSupported:
            return "kErrorNotSupported";
        case cutlass::Status::kErrorWorkspaceNull:
            return "kErrorWorkspaceNull";
        case cutlass::Status::kErrorInternal:
            return "kErrorInternal";
        case cutlass::Status::kInvalid:
            return "kInvalid";
        default:
            break;
    }
    return "invalid";
}

/////////////////////////////////////////////////////////////////////////////////////////////////
inline std::ostream& operator<<(
        std::ostream& out, cutlass::convolution::ConvParamBase const& params) {
    out << "{\n"
        << "          batch size: " << std::to_string(params.n()) << "\n"
        << "      output channel: " << std::to_string(params.co()) << "\n"
        << "       input channel: " << std::to_string(params.ci()) << "\n"
        << "  input tensor shape: " << std::to_string(params.hi()) << "x"
        << std::to_string(params.wi()) << "\n"
        << " output tensor shape: " << std::to_string(params.ho()) << "x"
        << std::to_string(params.wo()) << "\n"
        << "        filter shape: " << std::to_string(params.fh()) << "x"
        << std::to_string(params.fw()) << "\n"
        << "              stride: " << std::to_string(params.stride_h()) << "x"
        << std::to_string(params.stride_w()) << "\n"
        << "             padding: " << std::to_string(params.pad_h()) << "x"
        << std::to_string(params.pad_w()) << "\n"
        << "              dilate: " << std::to_string(params.dilate_h()) << "x"
        << std::to_string(params.dilate_w()) << "\n"
        << "}\n";

    return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

class TimerGPU {
public:
    cudaEvent_t start, stop;
    cudaStream_t stream;
    TimerGPU(cudaStream_t stream_ = 0) : stream{stream_} {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }
    ~TimerGPU() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float read() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};
}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
struct Testbed {
    using ElementAccumulator = typename Convolution::ElementAccumulator;
    using ElementCompute = typename Convolution::ConvolutionKernel::Epilogue::
            OutputOp::ElementCompute;

    /// Initialization
    cutlass::Distribution::Kind init_src;
    cutlass::Distribution::Kind init_filter;
    cutlass::Distribution::Kind init_bias;
    cutlass::Distribution::Kind init_z;
    uint64_t seed;

    cutlass::HostTensor<typename Convolution::ElementSrc,
                        typename Convolution::LayoutSrc>
            tensor_src;
    cutlass::HostTensor<typename Convolution::ElementFilter,
                        typename Convolution::LayoutFilter>
            tensor_filter;
    cutlass::HostTensor<typename Convolution::ElementBias,
                        typename Convolution::LayoutBias>
            tensor_bias;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_z;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_dst;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            reference_dst;

    //
    // Methods
    //

    Testbed(cutlass::Distribution::Kind init_src_ =
                    cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_filter_ =
                    cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_bias_ =
                    cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_z_ =
                    cutlass::Distribution::Uniform,
            uint64_t seed_ = 2080)
            : init_src(init_src_),
              init_filter(init_filter_),
              init_bias(init_bias_),
              init_z(init_z_),
              seed(seed_) {}

    /// Helper to initialize a tensor view
    template <typename Element, typename Layout>
    bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                           cutlass::Distribution::Kind dist_kind,
                           uint64_t seed) {
        if (dist_kind == cutlass::Distribution::Uniform) {
            double scope_max, scope_min;
            int bits_input = cutlass::sizeof_bits<Element>::value;
            int bits_output = cutlass::sizeof_bits<
                    typename Convolution::ElementDst>::value;

            if (bits_input == 1) {
                scope_max = 2;
                scope_min = 0;
            } else if (bits_input <= 8) {
                scope_max = 8;
                scope_min = -8;
            } else if (bits_output == 16) {
                scope_max = 5;
                scope_min = -5;
            } else {
                scope_max = 8;
                scope_min = -8;
            }

            cutlass::reference::host::TensorFillRandomUniform(
                    view, seed, scope_max, scope_min, 0);
        } else if (dist_kind == cutlass::Distribution::Identity) {
            cutlass::reference::host::TensorFillIdentity(view);
        } else if (dist_kind == cutlass::Distribution::Gaussian) {
            cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0,
                                                               0.5);
        } else if (dist_kind == cutlass::Distribution::Sequential) {
            cutlass::reference::host::BlockFillSequential(view.data(),
                                                          view.capacity());
        } else if (dist_kind == cutlass::Distribution::Constant) {
            cutlass::reference::host::TensorFill(view, Element(1));
        } else {
            // TODO: Implement the rest
            EXPECT_TRUE(false) << "Not implemented";
            return false;
        }

        return true;
    }

    /// Initializes data structures
    void initialize(
            cutlass::convolution::ConvParam<Convolution::kConvolutionType>
                    conv_param) {
        //
        // Allocate the CONVOLUTION workspace
        //

        tensor_src.resize(typename Convolution::LayoutSrc::TensorCoord{
                conv_param.n(), conv_param.hi(), conv_param.wi(),
                conv_param.ci()});
        tensor_filter.resize(typename Convolution::LayoutFilter::TensorCoord{
                conv_param.co(), conv_param.fh(), conv_param.fw(),
                conv_param.ci()});
        tensor_bias.resize(typename Convolution::LayoutBias::TensorCoord{
                1, 1, 1, conv_param.co()});
        tensor_z.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.n(), conv_param.ho(), conv_param.wo(),
                conv_param.co()});
        tensor_dst.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.n(), conv_param.ho(), conv_param.wo(),
                conv_param.co()});
        reference_dst.resize(
                typename Convolution::LayoutDst::TensorCoord{
                        conv_param.n(), conv_param.ho(), conv_param.wo(),
                        conv_param.co()},
                false);

        EXPECT_TRUE(initialize_tensor(tensor_src.host_view(), init_src,
                                      seed + 2019));
        EXPECT_TRUE(initialize_tensor(tensor_filter.host_view(), init_filter,
                                      seed + 2018));
        EXPECT_TRUE(initialize_tensor(tensor_bias.host_view(), init_bias,
                                      seed + 2017));
        EXPECT_TRUE(
                initialize_tensor(tensor_z.host_view(), init_z, seed + 2016));

        cutlass::reference::host::TensorCopy(reference_dst.host_view(),
                                             tensor_z.host_view());

        tensor_src.sync_device();
        tensor_filter.sync_device();
        tensor_bias.sync_device();
        tensor_z.sync_device();
        tensor_dst.sync_device();
    }

    /// Compares computed reference with device reference and outputs to a file
    /// if incorrect
    bool compare_reference() {
        tensor_dst.sync_host();

        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_src.host_view()),
                  0);
        EXPECT_GT(
                cutlass::reference::host::TensorNorm(tensor_filter.host_view()),
                0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_bias.host_view()),
                  0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_z.host_view()),
                  0);

        if (tensor_dst.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              tensor_dst.host_view()),
                      0);

        if (reference_dst.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              reference_dst.host_view()),
                      0);

        bool passed = cutlass::reference::host::TensorEquals(
                reference_dst.host_view(), tensor_dst.host_view());

        EXPECT_TRUE(passed);

        if (!passed) {
            std::stringstream fname_ref;

            fname_ref << "error_Convolution_device_reference_"
                      << Convolution::ThreadblockShape::kM << "x"
                      << Convolution::ThreadblockShape::kN << "x"
                      << Convolution::ThreadblockShape::kK << "_"
                      << Convolution::WarpShape::kM << "x"
                      << Convolution::WarpShape::kN << "x"
                      << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_ref(fname_ref.str());

            file_ref << "Reference =\n" << reference_dst.host_view();

            std::stringstream fname_comp;

            fname_comp << "error_Convolution_device_computed_"
                       << Convolution::ThreadblockShape::kM << "x"
                       << Convolution::ThreadblockShape::kN << "x"
                       << Convolution::ThreadblockShape::kK << "_"
                       << Convolution::WarpShape::kM << "x"
                       << Convolution::WarpShape::kN << "x"
                       << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_comp(fname_comp.str());

            file_comp << "Computed =\n" << tensor_dst.host_view();
        }

        return passed;
    }

    /// Verifies the result is a GEMM
    bool verify(cutlass::convolution::ConvParam<Convolution::kConvolutionType>
                        conv_param,
                ElementCompute alpha, ElementCompute beta,
                ElementCompute gamma) {
        //
        // Verify
        //

        cutlass::reference::host::Convolution<
                Convolution::kConvolutionType, typename Convolution::ElementSrc,
                typename Convolution::LayoutSrc,
                typename Convolution::ElementFilter,
                typename Convolution::LayoutFilter,
                typename Convolution::ElementDst,
                typename Convolution::LayoutDst,
                typename Convolution::ElementBias,
                typename Convolution::LayoutBias, ElementCompute,
                ElementAccumulator, typename Convolution::Operator>
                reference_convolution;

        reference_convolution(conv_param, alpha, tensor_src.host_ref(),
                              tensor_filter.host_ref(), beta,
                              tensor_bias.host_ref(), gamma,
                              tensor_z.host_ref(), reference_dst.host_ref(),
                              ElementAccumulator(0));

        return compare_reference();
    }

    /// Executes one test
    bool run(cutlass::convolution::ConvParam<Convolution::kConvolutionType>
                     conv_param,
             ElementCompute alpha = ElementCompute(1),
             ElementCompute beta = ElementCompute(1),
             ElementCompute gamma = ElementCompute(0)) {
        this->initialize(conv_param);

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{conv_param,
                                                  tensor_src.device_ref(),
                                                  tensor_filter.device_ref(),
                                                  tensor_bias.device_ref(),
                                                  tensor_z.device_ref(),
                                                  tensor_dst.device_ref(),
                                                  {alpha, beta, gamma}};

        Convolution conv_op;

        size_t workspace_size = Convolution::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = conv_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Run the CONVOLUTION
        //

        status = conv_op();

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Verify
        //

        bool passed = this->verify(conv_param, alpha, beta, gamma);

        if (!passed) {
            std::cout << "Error with alpha = " << alpha << ", beta = " << beta
                      << ", gamma = " << gamma << "\n"
                      << conv_param << std::endl;
        }

        return passed;
    }

    bool perf(cutlass::convolution::ConvParam<Convolution::kConvolutionType>
                      conv_param,
              ElementCompute alpha = ElementCompute(1),
              ElementCompute beta = ElementCompute(1),
              ElementCompute gamma = ElementCompute(0), int iterations = 1,
              bool verify = false) {
        this->initialize(conv_param);

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{conv_param,
                                                  tensor_src.device_ref(),
                                                  tensor_filter.device_ref(),
                                                  tensor_bias.device_ref(),
                                                  tensor_z.device_ref(),
                                                  tensor_dst.device_ref(),
                                                  {alpha, beta, gamma}};

        Convolution conv_op;

        size_t workspace_size = Convolution::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = conv_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Run the CONVOLUTION
        //

        status = conv_op();
        status = conv_op();

        TimerGPU timer;
        for (int i = 0; i < iterations; ++i) {
            status = conv_op();
        }
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        float time_ms = timer.read() / static_cast<float>(iterations);
        float ops =
                2.f * static_cast<float>(static_cast<int64_t>(conv_param.n()) *
                                         conv_param.co() * conv_param.ho() *
                                         conv_param.wo() * conv_param.fh() *
                                         conv_param.fw() * conv_param.ci());
        std::cout << conv_param << "Time = " << time_ms << "ms"
                  << "\n"
                  << "Performance = " << ops / (time_ms * 1e9) << "Tops"
                  << std::endl;

        bool passed = true;
        if (verify) {
            //
            // Verify
            //

            passed = this->verify(conv_param, alpha, beta, gamma);

            if (!passed) {
                std::cout << "Error with alpha = " << alpha
                          << ", beta = " << beta << ", gamma = " << gamma
                          << "\n"
                          << conv_param << std::endl;
            }
        }
        return passed;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
bool TestAllConvolution() {
    bool passed = true;

    double problem_alpha[] = {1.0};

    double problem_beta[] = {-1.0};

    double problem_gamma[] = {1.114124184164124};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    static const cutlass::convolution::ConvType kConvolutionType =
            Convolution::kConvolutionType;
    using ConvolutionParameter =
            cutlass::convolution::ConvParam<kConvolutionType>;
    std::vector<ConvolutionParameter> args;

    for (int n : {128, 48, 33}) {
        for (int ic : {32, 24, 64}) {
            for (int oc : {128, 32, 24}) {
                for (int ih : {8}) {
                    for (int iw : {8}) {
                        for (int fh : {3, 5, 7}) {
                            for (int ph : {static_cast<int>(fh / 2), 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ic, oc, ih, iw, fh, fh, oh, ow,
                                            sh, sh, ph, ph, 1, 1});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (cutlass::platform::is_same<
                                cutlass::layout::TensorNCxHWx<32>,
                                typename Convolution::LayoutDst>::value &&
                        arg.co() % 32 != 0)
                        continue;
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
bool TestConvolutionMma() {
    bool passed = true;

    double problem_alpha[] = {1.0};

    double problem_beta[] = {-1.3141413421};

    double problem_gamma[] = {1.0};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    static const cutlass::convolution::ConvType kConvolutionType =
            Convolution::kConvolutionType;
    using ConvolutionParameter =
            cutlass::convolution::ConvParam<kConvolutionType>;
    std::vector<ConvolutionParameter> args;

    for (int n : {128, 48, 33}) {
        for (int ic : {64, 96}) {
            for (int oc : {256, 96}) {
                for (int ih : {8}) {
                    for (int iw : {8}) {
                        for (int fh : {3, 5, 7}) {
                            for (int ph : {fh / 2, 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ic, oc, ih, iw, fh, fh, oh, ow,
                                            sh, sh, ph, ph, 1, 1});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
bool TestConvolutionPerf(int iterations = 1, int batch = 64,
                         bool tensor_op = false) {
    bool passed = true;

    double problem_alpha[] = {1.0};
    double problem_beta[] = {-1.0};
    double problem_gamma[] = {0.0};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    static const cutlass::convolution::ConvType kConvolutionType =
            Convolution::kConvolutionType;
    using ConvolutionParameter =
            cutlass::convolution::ConvParam<kConvolutionType>;
    std::vector<ConvolutionParameter> args;

    /// resnet-50
    args.emplace_back(ConvolutionParameter{batch, 4, 64, 224, 224, 7, 7, 112,
                                           112, 2, 2, 3, 3, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 64, 256, 56, 56, 1, 1, 56, 56,
                                           1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 512, 56, 56, 1, 1, 28,
                                           28, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 128, 56, 56, 1, 1, 28,
                                           28, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 128, 28, 28, 1, 1, 28,
                                           28, 1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 128, 512, 28, 28, 1, 1, 28,
                                           28, 1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 128, 128, 28, 28, 3, 3, 28,
                                           28, 1, 1, 1, 1, 1, 1});

    args.emplace_back(ConvolutionParameter{batch, 512, 1024, 28, 28, 1, 1, 14,
                                           14, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 256, 28, 28, 1, 1, 14,
                                           14, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 1024, 256, 14, 14, 1, 1, 14,
                                           14, 1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 256, 14, 14, 3, 3, 14,
                                           14, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 1024, 14, 14, 1, 1, 14,
                                           14, 1, 1, 0, 0, 1, 1});

    args.emplace_back(ConvolutionParameter{batch, 1024, 2048, 14, 14, 1, 1, 7,
                                           7, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 1024, 512, 14, 14, 1, 1, 7, 7,
                                           2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 2048, 512, 7, 7, 1, 1, 7, 7,
                                           1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 512, 7, 7, 3, 3, 7, 7, 1,
                                           1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 2048, 7, 7, 1, 1, 7, 7,
                                           1, 1, 0, 0, 1, 1});

    /// VGG-16
    args.emplace_back(ConvolutionParameter{batch, 64, 64, 224, 224, 3, 3, 224,
                                           224, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 64, 128, 112, 112, 3, 3, 112,
                                           112, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 128, 128, 112, 112, 3, 3, 112,
                                           112, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 128, 256, 56, 56, 3, 3, 56,
                                           56, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 256, 56, 56, 3, 3, 56,
                                           56, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 512, 28, 28, 3, 3, 28,
                                           28, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 512, 28, 28, 3, 3, 28,
                                           28, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 512, 14, 14, 3, 3, 14,
                                           14, 1, 1, 1, 1, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 512, 7, 7, 3, 3, 7, 7, 1,
                                           1, 1, 1, 1, 1});

    bool verify = true;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (tensor_op && (arg.ci() % 32 != 0 || arg.co() % 32 != 0))
                        continue;
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

template <typename Convolution>
bool TestConvolution1x1Perf(int iterations = 1, int batch = 64,
                         bool tensor_op = false) {
    bool passed = true;

    double problem_alpha[] = {1.0};
    double problem_beta[] = {-1.0};
    double problem_gamma[] = {0.0};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    static const cutlass::convolution::ConvType kConvolutionType =
            Convolution::kConvolutionType;
    using ConvolutionParameter =
            cutlass::convolution::ConvParam<kConvolutionType>;
    std::vector<ConvolutionParameter> args;

    /// resnet-50
    args.emplace_back(ConvolutionParameter{batch, 64, 256, 56, 56, 1, 1, 56, 56,
                                           1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 512, 56, 56, 1, 1, 28,
                                           28, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 128, 56, 56, 1, 1, 28,
                                           28, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 128, 28, 28, 1, 1, 28,
                                           28, 1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 128, 512, 28, 28, 1, 1, 28,
                                           28, 1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 1024, 28, 28, 1, 1, 14,
                                           14, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 256, 28, 28, 1, 1, 14,
                                           14, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 1024, 256, 14, 14, 1, 1, 14,
                                           14, 1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 256, 1024, 14, 14, 1, 1, 14,
                                           14, 1, 1, 0, 0, 1, 1});

    args.emplace_back(ConvolutionParameter{batch, 1024, 2048, 14, 14, 1, 1, 7,
                                           7, 2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 1024, 512, 14, 14, 1, 1, 7, 7,
                                           2, 2, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 2048, 512, 7, 7, 1, 1, 7, 7,
                                           1, 1, 0, 0, 1, 1});
    args.emplace_back(ConvolutionParameter{batch, 512, 2048, 7, 7, 1, 1, 7, 7,
                                           1, 1, 0, 0, 1, 1});

    bool verify = true;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (tensor_op && (arg.ci() % 32 != 0 || arg.co() % 32 != 0))
                        continue;
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

}  // namespace device
}  // namespace convolution
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
