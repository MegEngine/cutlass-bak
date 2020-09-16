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
    \brief Templates implementing loading of tiles from pitch-linear rank=2
   tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last
   "residue" tile first, with the objective of minimizing predicate mask updates
   during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

/**
 * \file include/cutlass/transform/threadblock/predicated_tile_iterator.h
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/convolution/convolution.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

namespace detail {
template <typename Shape_, int Interleaved>
CUTLASS_HOST_DEVICE void compute_offset(int* constant_offset_, int fh_, int fw_,
                                        int hi_, int wi_, int residue_offset_) {
    // hardcoded typedef
    using Shape = Shape_;
    using ShortIndex = int8_t;
    using Index = int;
    static int const kInterleaved = Interleaved;

    Index* offset_ptr = constant_offset_;
    ShortIndex* fhfw_ptr = reinterpret_cast<ShortIndex*>(constant_offset_ + 1);
    Index s = 0;
    Index filter_pixels = fh_ * fw_;
    Index image_pixels = hi_ * wi_;
    // first absolute offset
    CUTLASS_PRAGMA_UNROLL
    for (; s < Shape::kStrided; ++s) {
        Index c = s / (filter_pixels);
        Index fhfw = s - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        offset_ptr[0] = c * image_pixels * kInterleaved +
                        fh * wi_ * kInterleaved + fw * kInterleaved;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
    // step residue_offset_
    CUTLASS_PRAGMA_UNROLL
    for (; s < 2 * Shape::kStrided; ++s) {
        Index s_ = s - Shape::kStrided + residue_offset_;
        Index c = s_ / (filter_pixels);
        Index fhfw = s_ - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        offset_ptr[0] = c * image_pixels * kInterleaved +
                        fh * wi_ * kInterleaved + fw * kInterleaved;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        s_ = s_ - residue_offset_;
        c = s_ / (filter_pixels);
        fhfw = s_ - (filter_pixels)*c;
        fh = fhfw / fw_;
        fw = fhfw - fw_ * fh;
        offset_ptr[0] -= (c * image_pixels * kInterleaved +
                          fh * wi_ * kInterleaved + fw * kInterleaved);
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
    CUTLASS_PRAGMA_UNROLL
    for (; s < (2 + filter_pixels) * Shape::kStrided; ++s) {
        Index s_ = s - Shape::kStrided + residue_offset_;
        Index c = s_ / (filter_pixels);
        Index fhfw = s_ - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        offset_ptr[0] = c * image_pixels * kInterleaved +
                        fh * wi_ * kInterleaved + fw * kInterleaved;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        s_ = s_ - Shape::kStrided;
        c = s_ / (filter_pixels);
        fhfw = s_ - (filter_pixels)*c;
        fh = fhfw / fw_;
        fw = fhfw - fw_ * fh;
        offset_ptr[0] -= (c * image_pixels * kInterleaved +
                          fh * wi_ * kInterleaved + fw * kInterleaved);
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
}
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for TensorNCxHWx<Interleaved>
/// Layout. Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int Interleaved,
          typename TileMap_>
class PredicatedTileIterator<Shape_, Element_,
                             layout::TensorNCxHWx<Interleaved>, AdvanceRank,
                             ThreadMap_, AccessSize, TileMap_, true> {
public:
    static_assert(
            AdvanceRank == 1,
            "Specialization for tensor NCxHWx iterator must along advance "
            "along the "
            "strided(rank=1) dimension.");

    using Shape = layout::PitchLinearShape<Shape_::kColumn * Interleaved,
                                           Shape_::kRow / Interleaved>;
    using Element = Element_;
    static int const kInterleaved = Interleaved;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

    using ShortIndex = int8_t;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
 
    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    /// Type used for internal memory accesses
    using AccessType =
            AlignedArray<Element, AccessSize,
                         (AccessSize * sizeof_bits<Element>::value / 8)>;

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                  "Vectors implied by the thread map must be divisible by the "
                  "access type.");
    static_assert(AccessType::kElements == kInterleaved,
                  "Access size must equal to interleaving quantity");

    static int const kContiguousCount =
            ThreadMap::Iterations::kContiguous * kAccessesPerVector;

    struct Mask {
        static int const kCount = kContiguousCount < 8 ? 8 : kContiguousCount;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// Parameters object is precomputed state and is host-constructible
    class Params {
    public:
        /// Hardcoded maximum filter sizes
        static int const kMaxFilterPixels = 7 * 7;
        /// Element size in Index
        static int const kElementSize =
                (cutlass::sizeof_bits<Index>::value +
                 4 * cutlass::sizeof_bits<ShortIndex>::value) /
                cutlass::sizeof_bits<Index>::value;
        static int const kPrecomputedOffsetBufferSize =
                (2 + kMaxFilterPixels) * kElementSize * Shape::kStrided;

        static_assert(Shape::kStrided <= 8,
                      "Shape::kStrided is larger than 8, param may exceed "
                      "maximum kernel parameter buffer size");
        friend PredicatedTileIterator;

    private:
        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;

        /// Parameters used for mapping logical coordinates to physical
        /// coordinates
        TileMap tile_map_;
        Index stride_h_, stride_w_, pad_h_, pad_w_;
        Index hi_, wi_, n_;
        Index residue_offset_;
        Index constant_offset_max_;
        Index constant_offset_rewind_;
        Index constant_offset_[kPrecomputedOffsetBufferSize];

    public:
        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()), tile_map_(TileMap()) {}

        /// Construct the Params object given a pitch-linear tensor's layout
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, convolution::ConvParamBase const& params =
                                             convolution::ConvParamBase{})
                : layout_(layout),
                  tile_map_(TileMap(params.hw(), params.wo())),
                  stride_h_(params.stride_h()),
                  stride_w_(params.stride_w()),
                  pad_h_(params.pad_h()),
                  pad_w_(params.pad_w()),
                  hi_(params.hi()),
                  wi_(params.wi()),
                  n_(params.n()) {
            Index conv_iterations = params.ci() * params.fh() * params.fw();
            residue_offset_ =
                    (conv_iterations / kInterleaved) % Shape::kStrided;
            if (!residue_offset_) {
                residue_offset_ = Shape::kStrided;
            }
            detail::compute_offset<Shape, kInterleaved>(
                    constant_offset_, params.fh(), params.fw(), hi_, wi_,
                    residue_offset_);
            constant_offset_max_ =
                    (1 + params.fh() * params.fw()) * Shape::kStrided;
            constant_offset_rewind_ =
                    Shape::kStrided * (1 - params.fh() * params.fw());
        }

        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            TensorCoord tensor_coord = tile_map_(coord);
            tensor_coord.h() = tensor_coord.h() * stride_h_ - pad_h_;
            tensor_coord.w() = tensor_coord.w() * stride_w_ - pad_w_;
            return tensor_coord;
        }
    };

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_[kContiguousCount];

    /// predicates
    Mask mask_;

    /// Extent for the first steady-state tile
    Index residue_extent_;

    Index h_start_[kContiguousCount];
    Index h_end_[kContiguousCount];
    Index w_start_[kContiguousCount];
    Index w_end_[kContiguousCount];

    Index constant_offset_;
    Index strided_[ThreadMap::Iterations::kStrided];

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_DEVICE
    void initialize_predicate_and_pointers_(Pointer pointer,
                                            Index thread_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            int c = access_idx / kAccessesPerVector;
            int v = access_idx % kAccessesPerVector;

            Index col_offset = c * ThreadMap::Delta::kContiguous +
                               v * AccessType::kElements + thread_offset;

            TensorCoord coord =
                    params_(LogicalCoord{0, col_offset / kInterleaved});

            pointer_[access_idx] = pointer + params_.layout_(coord);
            h_start_[access_idx] = -coord.h();
            w_start_[access_idx] = -coord.w();
            h_end_[access_idx] = params_.hi_ - coord.h();
            w_end_[access_idx] = params_.wi_ - coord.w();
            mask_.predicates[access_idx] = coord.n() < params_.n_;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] =
                    params_.constant_offset_[2 *
                                             (constant_offset_ +
                                              s * ThreadMap::Delta::kStrided)];
        }
    }

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            /// Precomputed parameters object
            Params const& params,
            /// Pointer to start of tensor
            Pointer pointer,
            /// Extent of tensor
            LogicalCoord extent,
            /// ID of each participating thread
            int thread_id,
            /// Initial offset of threadblock
            LogicalCoord const& threadblock_offset)
            : params_(params), is_residue_tile_(true) {
        residue_extent_ = min(threadblock_offset.row() / kInterleaved +
                                      params_.residue_offset_,
                              extent.row() / kInterleaved);

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                LogicalCoord(threadblock_offset.row() / kInterleaved,
                             threadblock_offset.column() * kInterleaved) +
                LogicalCoord(thread_offset_.strided(),
                             thread_offset_.contiguous());

        // Intialize constant offset
        constant_offset_ = thread_offset.row();

        // Intialize internal pointers
        initialize_predicate_and_pointers_(pointer, thread_offset.column());

        residue_extent_ = residue_extent_ - thread_offset.row();
    }

    /// Construct a PredicatedTileIterator with zero threadblock offset
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            pointer_[access_idx] +=
                    sizeof_bits<Element>::value * pointer_offset / 8;
        }
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        if (constant_offset_ < params_.constant_offset_max_) {
            constant_offset_ += Shape::kStrided;
        } else {
            constant_offset_ += params_.constant_offset_rewind_;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] +=
                    params_.constant_offset_[2 *
                                             (constant_offset_ +
                                              s * ThreadMap::Delta::kStrided)];
        }
        is_residue_tile_ = false;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() { mask_.clear(); }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() { mask_.enable(); }

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) { mask_ = mask; }

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& /* mask */) { /* return mask_; */ }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            auto ptr_ = reinterpret_cast<ShortIndex const*>(
                    params_.constant_offset_ +
                    2 * (constant_offset_ + s * ThreadMap::Delta::kStrided) +
                    1);
            ShortIndex h = ptr_[0];
            ShortIndex w = ptr_[1];

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    bool guard = mask_.predicates[access_idx] &&
                                 ((h >= h_start_[access_idx]) &&
                                  (h < h_end_[access_idx]) &&
                                  (w >= w_start_[access_idx]) &&
                                  (w < w_end_[access_idx]));
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(pointer_[access_idx] +
                                                          strided_[s]) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        store_with_byte_offset(
                frag, pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            auto ptr_ = reinterpret_cast<ShortIndex const*>(
                    params_.constant_offset_ +
                    2 * (constant_offset_ + s * ThreadMap::Delta::kStrided) +
                    1);
            ShortIndex h = ptr_[0];
            ShortIndex w = ptr_[1];

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    bool guard = mask_.predicates[access_idx] &&
                                 ((h >= h_start_[access_idx]) &&
                                  (h < h_end_[access_idx]) &&
                                  (w >= w_start_[access_idx]) &&
                                  (w < w_end_[access_idx]));
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(pointer_[access_idx] +
                                                          strided_[s]) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
        }
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for TensorNCxHWx<32> Layout.
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, typename TileMap_>
class PredicatedTileIterator<Shape_, Element_, layout::TensorNCxHWx<32>,
                             AdvanceRank, ThreadMap_, AccessSize, TileMap_,
                             true> {
public:
    static_assert(
            AdvanceRank == 1,
            "Specialization for tensor NCxHWx iterator must along advance "
            "along the "
            "strided(rank=1) dimension.");

    static int const kInterleaved = 32;
    using Shape = layout::PitchLinearShape<Shape_::kColumn * kInterleaved,
                                           Shape_::kRow / kInterleaved>;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

    using ShortIndex = int8_t;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    /// Type used for internal memory accesses
    using AccessType =
            AlignedArray<Element, AccessSize,
                         (AccessSize * sizeof_bits<Element>::value / 8)>;

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                  "Vectors implied by the thread map must be divisible by the "
                  "access type.");
    static_assert(AccessType::kElements <= kInterleaved,
                  "Access size cannot be greater than interleaving quantity");

    static int const kPredicatesPerByte = 4;
    static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

    static int const kContiguousCount =
            ThreadMap::Iterations::kContiguous * kAccessesPerVector;

    /// Number of 32b words containing predicates
    static int const kPredicateByteCount =
            (kContiguousCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
    static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

    static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

    static_assert(kPredicateWordCount <= 4, "Too many predicates.");

    /// Predicate vector stores mask to guard accesses
    using Mask = Array<uint32_t, kPredicateWordCount>;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// Parameters object is precomputed state and is host-constructible
    class Params {
    public:
        /// Hardcoded maximum filter sizes
        static int const kMaxFilterPixels = 7 * 7;
        /// Element size in Index
        static int const kElementSize =
                (cutlass::sizeof_bits<Index>::value +
                 4 * cutlass::sizeof_bits<ShortIndex>::value) /
                cutlass::sizeof_bits<Index>::value;
        static int const kPrecomputedOffsetBufferSize =
                (2 + kMaxFilterPixels) * kElementSize * Shape::kStrided;

        static_assert(Shape::kStrided <= 8,
                      "Shape::kStrided is larger than 8, param may exceed "
                      "maximum kernel parameter buffer size");
 
        friend PredicatedTileIterator;

    private:
        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;

        /// Parameters used for mapping logical coordinates to physical
        /// coordinates
        TileMap tile_map_;
        Index stride_h_, stride_w_, pad_h_, pad_w_;
        Index hi_, wi_, n_;
        Index fh_, fw_;
        Index residue_offset_;
        Index constant_offset_max_;
        Index constant_offset_rewind_;
        Index constant_offset_[kPrecomputedOffsetBufferSize];

    public:
        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()), tile_map_(TileMap()) {}

        /// Construct the Params object given a pitch-linear tensor's layout
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, convolution::ConvParamBase const& params =
                                             convolution::ConvParamBase{})
                : layout_(layout),
                  tile_map_(TileMap(params.hw(), params.wo())),
                  stride_h_(params.stride_h()),
                  stride_w_(params.stride_w()),
                  pad_h_(params.pad_h()),
                  pad_w_(params.pad_w()),
                  hi_(params.hi()),
                  wi_(params.wi()),
                  n_(params.n()),
                  fh_(params.fh()),
                  fw_(params.fw()) {
            Index conv_iterations = params.ci() * params.fh() * params.fw();
            residue_offset_ =
                    (conv_iterations / kInterleaved) % Shape::kStrided;
            if (!residue_offset_) {
                residue_offset_ = Shape::kStrided;
            }
            detail::compute_offset<Shape, kInterleaved>(
                    constant_offset_, fh_, fw_, hi_, wi_, residue_offset_);
            constant_offset_max_ =
                    (1 + params.fh() * params.fw()) * Shape::kStrided;
            constant_offset_rewind_ =
                    Shape::kStrided * (1 - params.fh() * params.fw());
        }

        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            TensorCoord tensor_coord = tile_map_(coord);
            tensor_coord.h() = tensor_coord.h() * stride_h_ - pad_h_;
            tensor_coord.w() = tensor_coord.w() * stride_w_ - pad_w_;
            return tensor_coord;
        }
    };

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_[kContiguousCount];

    /// Array of boolean values to contain steady-state predicates
    /// Guard predicates
    uint32_t predicates_[kPredicateWordCount];

    /// Extent for the first steady-state tile
    Index residue_extent_;

    uint32_t extent_[kContiguousCount];

    Index constant_offset_;
    Index strided_[ThreadMap::Iterations::kStrided];

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_DEVICE
    void initialize_predicate_and_pointers_(Pointer pointer,
                                            Index thread_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            int c = access_idx / kAccessesPerVector;
            int v = access_idx % kAccessesPerVector;

            Index col_offset = c * ThreadMap::Delta::kContiguous +
                               v * AccessType::kElements + thread_offset;

            TensorCoord coord =
                    params_(LogicalCoord{0, col_offset / kInterleaved});

            pointer_[access_idx] = pointer + params_.layout_(coord) +
                                   col_offset % kInterleaved;
            ShortIndex x = params_.wi_ - coord.w() < params_.fw_
                                   ? coord.w() - params_.wi_ + 1
                                   : 1 - params_.fw_;
            ShortIndex y = params_.hi_ - coord.h() < params_.fh_
                                   ? coord.h() - params_.hi_ + 1
                                   : 1 - params_.fh_;
            ShortIndex z = -coord.w() >= 0 ? -coord.w() : 0;
            ShortIndex w = -coord.h() >= 0 ? -coord.h() : 0;
            extent_[access_idx] = (((uint32_t)(uint8_t)x) << 24) |
                                  (((uint32_t)(uint8_t)y) << 16) |
                                  (((uint32_t)(uint8_t)z) << 8) |
                                  (((uint32_t)(uint8_t)w));
            bool guard = coord.n() < params_.n_;
            int word_idx = access_idx / kPredicatesPerWord;
            int residual = access_idx % kPredicatesPerWord;
            int byte_idx = residual / kPredicatesPerByte;
            int bit_idx = residual % kPredicatesPerByte;
            predicates_[word_idx] |=
                    (unsigned(guard) << (byte_idx * 8 + bit_idx));
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] =
                    params_.constant_offset_[2 *
                                             (constant_offset_ +
                                              s * ThreadMap::Delta::kStrided)];
        }
    }

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            /// Precomputed parameters object
            Params const& params,
            /// Pointer to start of tensor
            Pointer pointer,
            /// Extent of tensor
            LogicalCoord extent,
            /// ID of each participating thread
            int thread_id,
            /// Initial offset of threadblock
            LogicalCoord const& threadblock_offset)
            : params_(params), is_residue_tile_(true) {
        residue_extent_ = min(threadblock_offset.row() / kInterleaved +
                                      params_.residue_offset_,
                              extent.row() / kInterleaved);

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                LogicalCoord(threadblock_offset.row() / kInterleaved,
                             threadblock_offset.column() * kInterleaved) +
                LogicalCoord(thread_offset_.strided(),
                             thread_offset_.contiguous());

        // Intialize constant offset
        constant_offset_ = thread_offset.row();

        // Intialize internal pointers
        initialize_predicate_and_pointers_(pointer, thread_offset.column());

        residue_extent_ = residue_extent_ - thread_offset.row();
    }

    /// Construct a PredicatedTileIterator with zero threadblock offset
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            pointer_[access_idx] +=
                    sizeof_bits<Element>::value * pointer_offset / 8;
        }
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        if (constant_offset_ < params_.constant_offset_max_) {
            constant_offset_ += Shape::kStrided;
        } else {
            constant_offset_ += params_.constant_offset_rewind_;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] +=
                    params_.constant_offset_[2 *
                                             (constant_offset_ +
                                              s * ThreadMap::Delta::kStrided)];
        }
        is_residue_tile_ = false;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0xffffffff;
        }
    }

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = mask[i];
        }
    }

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            mask[i] = predicates_[i];
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            auto ptr_ = reinterpret_cast<ShortIndex const*>(
                    params_.constant_offset_ +
                    2 * (constant_offset_ + s * ThreadMap::Delta::kStrided) +
                    1);
            uint32_t spatial = *(reinterpret_cast<uint32_t const*>(ptr_));

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    int word_idx = access_idx / kPredicatesPerWord;
                    int residual = access_idx % kPredicatesPerWord;
                    int byte_idx = residual / kPredicatesPerByte;
                    int bit_idx = residual % kPredicatesPerByte;
                    bool guard = ((predicates_[word_idx] &
                                   (1u << (byte_idx * 8 + bit_idx))) != 0);
                    if ((Shape::kContiguous / kInterleaved) == 256) {
                        uint32_t pred = 0;
                        asm volatile("vset4.s32.s32.ge.add %0, %1, %2, %3;"
                                     : "=r"(pred)
                                     : "r"(spatial), "r"(extent_[access_idx]),
                                       "r"(pred));
                        guard &= (pred == 4);
                    } else {
                        ShortIndex x, y;
                        uint32_t val = extent_[access_idx];
                        x = spatial & 0xff;
                        y = val & 0xff;
                        guard &= (x >= y);
                        x = (spatial >> 8) & 0xff;
                        y = (val >> 8) & 0xff;
                        guard &= (x >= y);
                        x = (spatial >> 16) & 0xff;
                        y = (val >> 16) & 0xff;
                        guard &= (x >= y);
                        x = (spatial >> 24) & 0xff;
                        y = (val >> 24) & 0xff;
                        guard &= (x >= y);
                    }
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(pointer_[access_idx] +
                                                          strided_[s]) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        store_with_byte_offset(
                frag, pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            auto ptr_ = reinterpret_cast<ShortIndex const*>(
                    params_.constant_offset_ +
                    2 * (constant_offset_ + s * ThreadMap::Delta::kStrided) +
                    1);
            uint32_t spatial = *(reinterpret_cast<uint32_t const*>(ptr_));

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    int word_idx = access_idx / kPredicatesPerWord;
                    int residual = access_idx % kPredicatesPerWord;
                    int byte_idx = residual / kPredicatesPerByte;
                    int bit_idx = residual % kPredicatesPerByte;
                    bool guard = ((predicates_[word_idx] &
                                   (1u << (byte_idx * 8 + bit_idx))) != 0);
                    if ((Shape::kContiguous / kInterleaved) == 256) {
                        uint32_t pred = 0;
                        asm volatile("vset4.s32.s32.ge.add %0, %1, %2, %3;"
                                     : "=r"(pred)
                                     : "r"(spatial), "r"(extent_[access_idx]),
                                       "r"(pred));
                        guard &= (pred == 4);
                    } else {
                        ShortIndex x, y;
                        uint32_t val = extent_[access_idx];
                        x = spatial & 0xff;
                        y = val & 0xff;
                        guard &= (x >= y);
                        x = (spatial >> 8) & 0xff;
                        y = (val >> 8) & 0xff;
                        guard &= (x >= y);
                        x = (spatial >> 16) & 0xff;
                        y = (val >> 16) & 0xff;
                        guard &= (x >= y);
                        x = (spatial >> 24) & 0xff;
                        y = (val >> 24) & 0xff;
                        guard &= (x >= y);
                    }
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(pointer_[access_idx] +
                                                          strided_[s]) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
        }
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for TensorNCxHWx<Interleaved>
/// Layout. Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int Interleaved,
          typename TileMap_>
class PredicatedTileIterator<Shape_, Element_,
                             layout::TensorNCxHWx<Interleaved>, AdvanceRank,
                             ThreadMap_, AccessSize, TileMap_, false> {
public:
    static_assert(
            AdvanceRank == 1,
            "Specialization for tensor NCxHWx iterator must along advance "
            "along the "
            "strided(rank=1) dimension.");

    static int const kInterleaved = Interleaved;
    using Shape = layout::PitchLinearShape<Shape_::kColumn * kInterleaved,
                                           Shape_::kRow / kInterleaved>;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    /// Type used for internal memory accesses
    using AccessType =
            AlignedArray<Element, AccessSize,
                         (AccessSize * sizeof_bits<Element>::value / 8)>;

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                  "Vectors implied by the thread map must be divisible by the "
                  "access type.");
    static_assert(AccessType::kElements <= kInterleaved,
                  "Access size cannot be greater than interleaving quantity");

    static int const kPredicatesPerByte = 4;
    static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

    static int const kContiguousCount =
            ThreadMap::Iterations::kContiguous * kAccessesPerVector;

    /// Number of 32b words containing predicates
    static int const kPredicateByteCount =
            (kContiguousCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
    static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

    static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

    static_assert(kPredicateWordCount <= 4, "Too many predicates.");

    /// Predicate vector stores mask to guard accesses
    using Mask = Array<uint32_t, kPredicateWordCount>;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// Parameters object is precomputed state and is host-constructible
    class Params {
    public:
        friend PredicatedTileIterator;

    private:
        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;
        /// amount (in byte) to increment pointer to move to next access along
        /// strided dimension
        LongIndex inc_strided_;
        /// amount (in byte) to increment pointer from last access to first
        /// access of next tile
        LongIndex inc_next_;
        LongIndex inc_iterations_;

        /// Parameters used for mapping logical coordinates to physical
        /// coordinates
        TileMap tile_map_;
        Index stride_h_, stride_w_, pad_h_, pad_w_;
        Index hi_, wi_, n_;

    public:
        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()), tile_map_(TileMap()) {}

        /// Construct the Params object given a pitch-linear tensor's layout
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, convolution::ConvParamBase const& params =
                                             convolution::ConvParamBase{})
                : layout_(layout),
                  tile_map_(TileMap(params.hw(), params.wo())),
                  stride_h_(params.stride_h()),
                  stride_w_(params.stride_w()),
                  pad_h_(params.pad_h()),
                  pad_w_(params.pad_w()),
                  hi_(params.hi()),
                  wi_(params.wi()),
                  n_(params.n()) {
            int stride = layout_.stride()[TileMap::kStrideAxis];
            inc_strided_ = (LongIndex(stride) * ThreadMap::Delta::kStrided) *
                           sizeof_bits<Element>::value / 8;

            inc_iterations_ = LongIndex(ThreadMap::Iterations::kStrided - 1) *
                              ThreadMap::Delta::kStrided * LongIndex(stride) *
                              sizeof_bits<Element>::value / 8;

            inc_next_ = Shape::kStrided * LongIndex(stride) *
                                sizeof_bits<Element>::value / 8 -
                        inc_iterations_;
        }

        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            TensorCoord tensor_coord = tile_map_(coord);
            tensor_coord.h() = tensor_coord.h() * stride_h_ - pad_h_;
            tensor_coord.w() = tensor_coord.w() * stride_w_ - pad_w_;
            return tensor_coord;
        }
    };

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_[kContiguousCount];

    /// Array of boolean values to contain steady-state predicates
    /// Guard predicates
    uint32_t predicates_[kPredicateWordCount];

    /// Offset to the first steady-state tile
    Index residue_offset_;

    Index residue_extent_;

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_DEVICE
    void initialize_predicate_and_pointers_(Pointer pointer,
                                            LogicalCoord const& thread_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            int c = access_idx / kAccessesPerVector;
            int v = access_idx % kAccessesPerVector;

            Index col_offset = c * ThreadMap::Delta::kContiguous +
                               v * AccessType::kElements +
                               thread_offset.column();
            TensorCoord coord =
                    params_(LogicalCoord{thread_offset.row() * kInterleaved,
                                         col_offset / kInterleaved});

            pointer_[access_idx] = pointer + params_.layout_(coord) +
                                    col_offset % kInterleaved;
            bool guard = coord.n() < params_.n_ && coord.h() >= 0 &&
                         coord.h() < params_.hi_ && coord.w() >= 0 &&
                         coord.w() < params_.wi_;
            int word_idx = access_idx / kPredicatesPerWord;
            int residual = access_idx % kPredicatesPerWord;
            int byte_idx = residual / kPredicatesPerByte;
            int bit_idx = residual % kPredicatesPerByte;
            predicates_[word_idx] |=
                    (unsigned(guard) << (byte_idx * 8 + bit_idx));
        }
    }

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            /// Precomputed parameters object
            Params const& params,
            /// Pointer to start of tensor
            Pointer pointer,
            /// Extent of tensor
            LogicalCoord extent,
            /// ID of each participating thread
            int thread_id,
            /// Initial offset of threadblock
            LogicalCoord const& threadblock_offset)
            : params_(params), is_residue_tile_(true) {
        residue_offset_ = (extent.row() / kInterleaved -
                           threadblock_offset.row() / kInterleaved) %
                          Shape::kStrided;
        if (!residue_offset_) {
            residue_offset_ = Shape::kStrided;
        }

        residue_extent_ =
                min(threadblock_offset.row() / kInterleaved + residue_offset_,
                    extent.row() / kInterleaved);

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                LogicalCoord(threadblock_offset.row() / kInterleaved,
                             threadblock_offset.column() * kInterleaved) +
                LogicalCoord(thread_offset_.strided(),
                             thread_offset_.contiguous());

        // Intialize internal pointers
        initialize_predicate_and_pointers_(pointer, thread_offset);

        residue_extent_ = residue_extent_ - thread_offset.row();
    }

    /// Construct a PredicatedTileIterator with zero threadblock offset
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            pointer_[access_idx] +=
                    sizeof_bits<Element>::value * pointer_offset / 8;
        }
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        if (is_residue_tile_) {
            add_pointer_offset(residue_offset_ *
                               params_.layout_.stride()[TileMap::kStrideAxis]);
            CUTLASS_PRAGMA_UNROLL
            for (int access_idx = 0; access_idx < kContiguousCount;
                 ++access_idx) {
                pointer_[access_idx] -= params_.inc_iterations_;
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int access_idx = 0; access_idx < kContiguousCount;
                 ++access_idx) {
                pointer_[access_idx] += params_.inc_next_;
            }
        }
        is_residue_tile_ = false;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0xffffffff;
        }
    }

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = mask[i];
        }
    }

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            mask[i] = predicates_[i];
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    int word_idx = access_idx / kPredicatesPerWord;
                    int residual = access_idx % kPredicatesPerWord;
                    int byte_idx = residual / kPredicatesPerByte;
                    int bit_idx = residual % kPredicatesPerByte;
                    bool guard = ((predicates_[word_idx] &
                                   (1u << (byte_idx * 8 + bit_idx))) != 0);
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr = reinterpret_cast<char const*>(
                                                   pointer_[access_idx]) +
                                           byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
            if (s < ThreadMap::Iterations::kStrided - 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int access_idx = 0; access_idx < kContiguousCount;
                     ++access_idx) {
                    pointer_[access_idx] += params_.inc_strided_;
                }
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        store_with_byte_offset(
                frag, pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    int word_idx = access_idx / kPredicatesPerWord;
                    int residual = access_idx % kPredicatesPerWord;
                    int byte_idx = residual / kPredicatesPerByte;
                    int bit_idx = residual % kPredicatesPerByte;
                    bool guard = ((predicates_[word_idx] &
                                   (1u << (byte_idx * 8 + bit_idx))) != 0);
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr = reinterpret_cast<char const*>(
                                                   pointer_[access_idx]) +
                                           byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
            if (s < ThreadMap::Iterations::kStrided - 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int access_idx = 0; access_idx < kContiguousCount;
                     ++access_idx) {
                    pointer_[access_idx] += params_.inc_strided_;
                }
            }
        }
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////


}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
