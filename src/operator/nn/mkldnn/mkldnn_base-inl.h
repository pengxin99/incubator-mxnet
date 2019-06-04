/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_base-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*         zhengda1936@gmail.com
*
*******************************************************************************/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>
#include "mkldnn.hpp"
#include "mxnet/ndarray.h"
#include "mxnet/resource.h"
#include "mxnet/op_attr_types.h"
using namespace mkldnn;
namespace mxnet {

// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine {
 public:
  static CpuEngine *Get() {
    // I's thread-safe in C++11.
    // ensure same mkldnn engine is used across threads
    static CpuEngine myInstance;
    return &myInstance;
  }
  CpuEngine(CpuEngine const &) = delete;             // Copy construct
  CpuEngine(CpuEngine &&) = delete;                  // Move construct
  CpuEngine &operator=(CpuEngine const &) = delete;  // Copy assign
  CpuEngine &operator=(CpuEngine &&) = delete;       // Move assign

  mkldnn::engine &get_engine() { return _cpu_engine; }

 protected:
  CpuEngine() : _cpu_engine(mkldnn::engine::kind::cpu, 0) {}
  ~CpuEngine() {}

 private:
  mkldnn::engine _cpu_engine;
};

// type enumerator
template <typename T>
struct data_type_enum {};

template <>
struct data_type_enum<float> {
  enum class{ type = mkldnn::memory::data_type::f32 };
};

template <>
struct data_type_enum<int32_t> {
  enum class{ type = mkldnn::memory::data_type::s32 };
};

template <>
struct data_type_enum<int8_t> {
  enum class{ type = mkldnn::memory::data_type::s8 };
};

template <>
struct data_type_enum<uint8_t> {
  enum class{ type = mkldnn::memory::data_type::u8 };
};

static inline bool SupportMKLDNNArray(int dtype, const mxnet::TShape &shape) {
  int ndim = shape.ndim();
  bool support = ndim == 1 || ndim == 2 || ndim == 4;
  support = support && (dtype == mshadow::kFloat32 || dtype == mshadow::kInt32
                        || dtype == mshadow::kInt8 || dtype == mshadow::kUint8);
  return support;
}

static inline bool SupportStorageMKLDNN(int stype) {
  return stype == kDefaultStorage;
}

static inline bool SupportMKLDNN(int dtype, const mxnet::TShape &shape) {
  int ndim = shape.ndim();
  return dtype == mshadow::kFloat32 && (ndim == 1 || ndim == 2 || ndim == 4);
}

static inline bool SupportMKLDNNQuantize(int dtype) {
  return dtype == mshadow::kFloat32 || dtype == mshadow::kInt8 ||
         dtype == mshadow::kUint8;
}

static inline bool SupportMKLDNN(const NDArray &input) {
  return SupportMKLDNN(input.dtype(), input.shape())
      && SupportStorageMKLDNN(input.storage_type());
}

static inline bool MKLDNNEnvSet() {
    // comment following lines static to simplify mkldnn unittest
    // static bool is_mkldnn_enabled = dmlc::GetEnv("MXNET_MKLDNN_ENABLED", true);
    bool is_mkldnn_enabled = dmlc::GetEnv("MXNET_MKLDNN_ENABLED", true);
    LOG(INFO) << "change for mkldnnv1.0 quick test, need recover when 1.0 upgrade done, return " << is_mkldnn_enabled;   
     return is_mkldnn_enabled;
}

static inline int GetMKLDNNCacheSize() {
  static int mkldnn_cache_size = dmlc::GetEnv("MXNET_MKLDNN_CACHE_NUM", -1);
  return mkldnn_cache_size;
}

// TODO(alex): (MXNET-1075) Will remove env variable and calculate cache size during runtime
template<typename S, typename I, typename H>
static typename std::unordered_map<S, I, H>::iterator AddToCache(
    std::unordered_map<S, I, H>* cache, const S &key, const I &item) {
  int mkldnn_cache_size = GetMKLDNNCacheSize();
  if (mkldnn_cache_size != -1 && static_cast<int>(cache->size()) > mkldnn_cache_size)
    cache->erase(cache->begin());
  auto ins_return = cache->insert(std::pair<S, I>(key, item));
  CHECK(ins_return.second);
  return ins_return.first;
}

/*
 * This is to align address to a certain alignment.
 */
void *AlignMem(void *mem, size_t size, size_t alignment, size_t *space);

namespace op {
struct ActivationParam;
struct ConvolutionParam;
struct DeconvolutionParam;
struct SoftmaxParam;
struct SoftmaxOutputParam;
struct TransposeParam;
bool SupportMKLDNNAct(const ActivationParam& param);
bool SupportMKLDNNAct(const ActivationParam& param, const NDArray &input);
bool SupportMKLDNNConv(const ConvolutionParam& params, const NDArray &input);
bool SupportMKLDNNDeconv(const DeconvolutionParam& params, const NDArray &input);
bool SupportMKLDNNSoftmax(const SoftmaxParam& param);
bool SupportMKLDNNSoftmaxOutput(const SoftmaxOutputParam &param);
bool SupportMKLDNNTranspose(const TransposeParam& param, const NDArray &data);
}  // namespace op

static int GetTypeSize(int dtype) {
  int size = -1;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    size = sizeof(DType);
  });
  return size;
}

static inline size_t GetArraySize(const NDArray &arr) {
  if (arr.IsMKLDNNData()) {
    return arr.GetMKLDNNData()->get_desc().get_size();
  }
  return arr.shape().Size() * GetTypeSize(arr.dtype());
}

static inline mkldnn::memory::data_type get_mkldnn_type(int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return mkldnn::memory::data_type::f32;
    case mshadow::kInt32:
      return mkldnn::memory::data_type::s32;
    case mshadow::kInt8:
      return mkldnn::memory::data_type::s8;
    case mshadow::kUint8:
      return mkldnn::memory::data_type::u8;
    default:
      LOG(FATAL) << "unknown type for MKLDNN";
      return mkldnn::memory::data_type::undef;
  }
}

static inline int get_mxnet_type(mkldnn_data_type_t dtype) {
  auto mkldnn_dtype = static_cast<mkldnn::memory::data_type>(dtype);
  switch (mkldnn_dtype) {
    case mkldnn::memory::data_type::f32:
      return mshadow::kFloat32;
    case mkldnn::memory::data_type::s32:
      return mshadow::kInt32;
    case mkldnn::memory::data_type::s8:
      return mshadow::kInt8;
    case mkldnn::memory::data_type::u8:
      return mshadow::kUint8;
    default:
      LOG(FATAL) << "unknown MKLDNN type";
      return mshadow::kFloat32;
  }
}

static inline size_t GetMemDescSize(const mkldnn::memory::desc &md) {
  if (md.data.ndims == 0) return 0;

  size_t ret = 1;
  for (int i = 0; i < md.data.ndims; i++) {
    ret *= md.data.dims[i];
  }

  ret *= mshadow::mshadow_sizeof(get_mxnet_type(md.data.data_type));
  return ret;
}

inline static mkldnn::memory::desc GetMemDesc(const NDArray &arr, int dtype = -1) {
  int ndim = arr.shape().ndim();
  mkldnn::memory::dims dims(ndim);
  dtype = (dtype == -1) ? arr.dtype() : dtype;
  for (size_t i = 0; i < dims.size(); i++) dims[i] = arr.shape()[i];
  return mkldnn::memory::desc{dims, get_mkldnn_type(dtype), mkldnn::memory::format_tag::any};
}

inline static mkldnn::memory::desc GetWeightDesc(const NDArray &arr,
                                                 int num_groups,
                                                 bool quantized = false) {
  int dtype = quantized ? mshadow::kInt8 : arr.dtype();
  if (num_groups == 1) {
    return GetMemDesc(arr, dtype);
  } else {
    auto ndim = arr.shape().ndim();
    CHECK((ndim == 3) || (ndim == 4))
        << "MKL-DNN weight currectly supports 3d and 4d layout";
    auto tz = mkldnn::memory::dims{0};
    const int N = 0, H = 2, W = 3, C = 1;
    if (ndim == 3) {
      tz = mkldnn::memory::dims{
          num_groups, static_cast<int>(arr.shape()[N] / num_groups),
          static_cast<int>(arr.shape()[C]), static_cast<int>(arr.shape()[H])};
    } else {
      tz = mkldnn::memory::dims{
          num_groups, static_cast<int>(arr.shape()[N] / num_groups),
          static_cast<int>(arr.shape()[C]), static_cast<int>(arr.shape()[H]),
          static_cast<int>(arr.shape()[W])};
    }
    return mkldnn::memory::desc{tz, get_mkldnn_type(dtype), mkldnn::memory::format_tag::any};
  }  
}

typedef std::shared_ptr<mkldnn::memory> mkldnn_mem_ptr;
typedef std::shared_ptr<const mkldnn::memory> mkldnn_mem_const_ptr;

/*
 * This is to manage the temporary memory provided by MXNet for operators.
 * The temp memory is mainly used to keep the reordered data. In an operator, we
 * may need multiple pieces of memory for them. But MXNet can only provide
 * a single piece of memory. This class is to help break the temporary memory
 * from MXNet to store the reordered data.
 * The amount of temporary memory used in an operator depends on the layout of
 * input arrays and the operator. It's difficult to calculate it manually, so
 * the class also estimate the amount of memory automatically.
 */
class TmpMemMgr {
  // This points to the memory buffer where we can allocate temp memory.
  char *curr_mem;
  // The total size of the temp memory.
  size_t mem_size;
  // This contains the current available memory size.
  size_t curr_size;
  // This estimate the required temp memory size in an operator.
  size_t est_size;
  const size_t alignment = kMKLDNNAlign;

 public:
  static TmpMemMgr *Get() {
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local TmpMemMgr mgr;
#else
    static MX_THREAD_LOCAL TmpMemMgr mgr;
#endif
    return &mgr;
  }

  TmpMemMgr() {
    Reset();
    est_size = 0;
    mem_size = 0;
  }

  void Reset() {
    curr_mem = nullptr;
    curr_size = 0;
    // We don't reset est_size and mem_size because est_size contains the
    // estimated temp memory size from the last run and mem_size contains the
    // memroy size allocated in the last run.
  }

  void Init(const Resource &r) {
    // If the last time, if we estimate that we need more memory, we should the
    // larger memory size.
    mem_size = std::max(mem_size, est_size);
    if (mem_size > 0) {
      // Let's allocate some extra memory. If we don't use some of them all the time,
      // the OS won't physically allocate pages for them any way.
      this->curr_size = mem_size * 2;
      this->curr_mem = static_cast<char *>(r.get_host_space_internal(this->curr_size));
    }
    // reset est_size, so we can start to estimate the temp memory size.
    this->est_size = 0;
  }

  mkldnn::memory *Alloc(const mkldnn::memory::desc &desc);
};

class MKLDNNStream {
  std::vector<std::unordered_map<int, mkldnn::memory>> net_args;
  std::vector<mkldnn::primitive> net;
  // Here we hold all memory related to the operators in the stream.
  std::vector<std::shared_ptr<const mkldnn::memory> > mem_holder;
  stream s;
 public:
  static MKLDNNStream *Get();
  MKLDNNStream():s(CpuEngine::Get()->get_engine()) {}
  
  void RegisterArgs(std::unordered_map<int, mkldnn::memory> args) {
    net_args.push_back(args);
  LOG(INFO)<<"net_args size "<<net_args.size();
  }
  
  void RegisterPrim(const mkldnn::primitive &prim) {
    net.push_back(prim);
  }

  void RegisterMem(std::shared_ptr<const mkldnn::memory> mem) {
    mem_holder.push_back(mem);
  }

  bool HasOps() const {
    return !net.empty();
  }

  /*
   * After submitting mkldnn operations for execution, we need to
   * clean up memory held by the stream. However, sometimes users
   * might want to separate mkldnn execution and memory cleanup.
   */
  void Submit(bool cleanup = true) {
    if (!net.empty()) {
      CHECK(net.size() == net_args.size());
      for(size_t i = 0; i < net.size(); i++) {
        net.at(i).execute(s, net_args.at(i));
      }
      net.clear();
      net_args.clear();
    }
    if (cleanup)
      Cleanup();
  }
  void Cleanup() {
    mem_holder.clear();
    TmpMemMgr::Get()->Reset();
  }
};

enum OutDataOp {
  Noop,
  CopyBack,
  AddBack,
};

typedef std::pair<OutDataOp, mkldnn::memory *> mkldnn_output_t;
void MKLDNNCopy(const mkldnn::memory &mem, const mkldnn::memory* this_mem);

static inline bool SameFormat(
    mkldnn::memory::desc desc_a, mkldnn::memory::desc desc_b) {
  bool rslt = false;
  int i = 0;
  int j = 0;
  if(desc_a.data.format_kind == desc_b.data.format_kind
     && desc_a.data.ndims == desc_b.data.ndims) {
    switch(desc_a.data.format_kind) {
      case mkldnn_blocked:
        for (i = 0; i < desc_a.data.ndims; i++) {
          if (desc_a.data.format_desc.blocking.strides[i]
             != desc_b.data.format_desc.blocking.strides[i]) {
            break;
          }
        }
        if (i != desc_a.data.ndims) {
          break;
        }
        for (i = 0; i < desc_a.data.format_desc.blocking.inner_nblks; i++) {
          if ((desc_a.data.format_desc.blocking.inner_blks[i]
               != desc_b.data.format_desc.blocking.inner_blks[i])
             &&(desc_a.data.format_desc.blocking.inner_idxs[i]
               != desc_b.data.format_desc.blocking.inner_idxs[i])) {
            break; 
          }  
        }
        if(i != desc_a.data.format_desc.blocking.inner_nblks) {
              break;
        }
        rslt = true;
      break;
      case mkldnn_format_kind_wino:
        if (desc_a.data.format_desc.wino_desc.wino_format
           == desc_b.data.format_desc.wino_desc.wino_format) {
          rslt = true;
        }
      break;
      case mkldnn_format_kind_rnn_packed:
        if (desc_a.data.format_desc.rnn_packed_desc.format
           == desc_b.data.format_desc.rnn_packed_desc.format) {
          rslt = true;
        }
    break;
      default:
       // nothing need to add
       rslt = false;
      break;
    }
  }
  LOG(INFO)<<"mkldnnv1.0 SameFormat return "<< rslt;
  return rslt;
}

/*
 * Here we want to get MKLDNN memory whose desc is exactly the same as
 * the given one. operator== can't guarantee that. == can return true even if
 * the formats are different. I need to double check its format.
 */
static inline mkldnn::memory *GetMKLDNNExact(
    const mkldnn::memory *mem, mkldnn::memory::desc desc) {
  mkldnn::memory::desc src_desc = mem->get_desc();
  if (desc == src_desc && SameFormat(desc, src_desc)) {
    LOG(INFO)<<"mkldnnv1.0 GetMKLDNNExact return same format";
    return const_cast<mkldnn::memory *>(mem);
  } else {
    LOG(INFO)<<"mkldnnv1.0 GetMKLDNNExact return different format";
    std::shared_ptr<mkldnn::memory> ret(new mkldnn::memory(
            desc, CpuEngine::Get()->get_engine(), mem->get_data_handle()));
    MKLDNNStream::Get()->RegisterMem(ret);
    return ret.get();
  }
}

/*
 * These two functions try to create MKLDNN memory in an NDArray based on `req'.
 * The difference is that the first function can create MKLDNN memory with
 * special layouts in an NDArray, while the second one can only create MKLDNN
 * memory with default layouts.
 * Also an optional in_arr parameter can be passed in the first function with
 * the kWriteInPlace req to validate if mkldnn can support write in place;
 * otherwise new memory will be written to an copied back onto out_arr.
 * If these two functions are used, we have to call CommitOutput to write
 * the output back to the output NDArray.
 */
mkldnn_output_t CreateMKLDNNMem(const NDArray &out_arr,
                                const mkldnn::memory::desc &desc,
                                OpReqType req, const NDArray* in_arr = nullptr);
mkldnn_output_t CreateMKLDNNWeightGrad(const NDArray &out_arr,
                                       const mkldnn::memory::desc &desc,
                                       OpReqType req);
/* This function has to be used with one of the functions above. */
void CommitOutput(const NDArray &arr, const mkldnn_output_t &res);

static inline void InvalidateOutputs(const std::vector<NDArray> &arrs,
                                     const std::vector<OpReqType> &reqs) {
  for (size_t i = 0; i < arrs.size(); i++) {
    if (reqs[i] == kWriteTo || reqs[i] == kNullOp) {
      const_cast<NDArray &>(arrs[i]).InvalidateMKLDNNData();
    }
  }
}

// TODO(alexzai): (MXNET-856) Remove helper function after subgraph feature added
static inline void CreateDefaultInputs(const std::vector<NDArray> &arrs,
                                       std::vector<NDArray> *out_arrs) {
  out_arrs->clear();
  for (size_t i = 0; i < arrs.size(); ++i) {
    if (arrs[i].IsMKLDNNData())
      out_arrs->push_back(arrs[i].Reorder2Default());
    else
      out_arrs->push_back(arrs[i]);
  }
}

const mkldnn::memory *GetWeights(const NDArray &arr, int num_groups);

const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::memory::desc &target_desc,
                                 int num_groups);

mkldnn_format_tag_t GetDefaultFormat(const mkldnn::memory::desc &desc);
mkldnn_format_tag_t GetDefaultFormat(int num_dims);

bool IsDefaultFormat(const mkldnn::memory::desc &desc);
bool IsMKLDNN(const mkldnn::memory::desc &desc);

mkldnn::memory::desc GetDesc(mkldnn::memory::desc desc,
                                                mkldnn_format_tag_t format);

inline bool same_shape(const mxnet::TShape &shape, const mkldnn_dims_t dims, int ndims) {
  if (shape.ndim() != ndims)
    return false;
  for (int i = 0; i < ndims; i++)
    if (shape[i] != dims[i])
      return false;
  return true;
}

inline bool same_shape(const mkldnn::memory::desc &desc1,
                       const mkldnn::memory::desc &desc2) {
  if (desc1.data.ndims != desc2.data.ndims)
    return false;
  for (int i = 0; i < desc1.data.ndims; i++)
    if (desc1.data.dims[i] != desc2.data.dims[i])
      return false;
  return true;
}

inline bool same_shape(const mxnet::TShape &shape, int dtype,
                       const mkldnn::memory::desc &desc) {
  return same_shape(shape, desc.data.dims, desc.data.ndims)
      && get_mkldnn_type(dtype) == desc.data.data_type;
}

/*
 * There is a large overhead of getting mkldnn::memory::desc from
 * mkldnn::memory. This class is created to cache the metadata of mkldnn memory
 * to provide a much more lightweight method to access them.
 */
class MKLDNNMemory {
  std::shared_ptr<mkldnn::memory> mem;
  mkldnn::memory::desc desc;
  size_t size;      // The number of bytes.

 public:
  MKLDNNMemory(mkldnn::memory::desc mem_desc, void *addr): desc(mem_desc) {
    mem.reset(new mkldnn::memory(desc, CpuEngine::Get()->get_engine(), addr));
    size = desc.get_size();
  }

  explicit MKLDNNMemory(std::shared_ptr<mkldnn::memory> mem): desc(
      mem->get_desc()) {
    this->mem = mem;
    size = desc.get_size();
  }

  void SetDataHandle(void *handle) {
    mem->set_data_handle(handle);
  }

  void *GetDataHandle() const {
    return mem->get_data_handle();
  }

  std::shared_ptr<mkldnn::memory> GetMem() const {
    return mem;
  }

  mkldnn::memory *GetRaw() const {
    return mem.get();
  }

  size_t GetSize() const {
    return size;
  }

  mkldnn::memory::desc GetDesc() const {
    return mem->get_desc();
  }

  mkldnn::memory::desc GetDesc(mkldnn_format_tag_t format_tag) const {
   auto desc = mem->get_desc();
   mkldnn::memory::dims dims(desc.data.ndims);
   for (size_t i = 0; i < dims.size(); i++)
	 dims[i] = desc.data.dims[i];
   mkldnn::memory::data_type cpp_type = static_cast<mkldnn::memory::data_type>(desc.data.data_type);
   mkldnn::memory::desc data_md(dims, cpp_type, static_cast<mkldnn::memory::format_tag>(format_tag));
   return data_md;
  }

  mkldnn_format_tag_t GetDefaultFormat() const {
    return mxnet::GetDefaultFormat(desc);
  }

  mkldnn_format_tag_t GetFormat() const {
#if 0  
    return desc.data.format;
#endif 
    LOG(FATAL)<<"mkldnnv1.0 GetFormat";
  }

  bool IsMKLDNN() const {
    bool rslt = true;
    if(desc.data.format_kind == mkldnn_blocked) {
      if(desc.data.format_desc.blocking.inner_nblks == 0) {
        int i = 0;
        for(i = 0; i < desc.data.ndims-1; i++) {
      	if(desc.data.format_desc.blocking.strides[i] < desc.data.format_desc.blocking.strides[i + 1]) {
      	  break;
      	}
        }
        if(i == desc.data.ndims-1) {
      	rslt = false;
        }
      }
    }
    LOG(INFO)<<"mkldnnv1.0 IsMKLDNN return "<< rslt;
    return rslt;
  }

  bool SameFormat(const mkldnn::memory::desc &desc) const {
    return mem->get_desc() == desc;
  }

  bool SameFormat(const mxnet::TShape &shape, int dtype) const {
    return same_shape(shape, dtype, desc);
  }

  void ReorderTo(mkldnn::memory *other) const {
    stream s(CpuEngine::Get()->get_engine());
    mkldnn::reorder(*mem, *other).execute(s, *mem, *other);
  }
};

void FallBackCompute(FCompute fn, const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<NDArray> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &outputs);

/*
 * This class is used to check the correctness of MKLDNN operators.
 */
class OpCheck {
  std::vector<mxnet::NDArray> inputs;
  std::vector<mxnet::NDArray> outputs;
  bool backward;
  size_t num_checks;

 public:
  OpCheck(bool backward, size_t num_checks) {
    this->backward = backward;
    this->num_checks = num_checks;
  }

  void Init(const std::vector<mxnet::NDArray> &inputs_,
          const std::vector<mxnet::NDArray> &outputs_);

  void Run(mxnet::FCompute fn, const nnvm::NodeAttrs &attrs,
           const mxnet::OpContext &ctx,
           const std::vector<mxnet::NDArray> &inputs_,
           const std::vector<mxnet::OpReqType> &req,
           const std::vector<mxnet::NDArray> &outputs_);

  void CopyResult(const std::vector<mxnet::NDArray> &outputs_,
                  const std::vector<size_t>& indice);
};

bool MKLDNNStorageType(const nnvm::NodeAttrs &attrs,
                       const int dev_mask,
                       bool support_mkldnn,
                       DispatchMode *dispatch_mode,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs);

#define MKLDNN_OPCHECK_INIT(backward, num_checks, inputs, outputs)  \
    static bool debug = dmlc::GetEnv("MXNET_MKLDNN_DEBUG", false);  \
    OpCheck check(backward, num_checks);                            \
    if (debug) check.Init(inputs, outputs);

#define MKLDNN_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs)    \
    if (debug) check.Run(fn, attrs, ctx, inputs, req, outputs);
#define MKLDNN_OPCHECK_COPY_RESULT(outputs, indice) \
    if (debug) check.CopyResult(outputs, indice);

}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_

