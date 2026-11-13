// Shared mapped single-token Qwen proving-ground support.
#pragma once
#include "sub0/dense_quant.hpp"
#include "sub0/dense_quant_q8k.hpp"
#include "sub0/dense_quant_q16.hpp"
#include "sub0/file_map.hpp"
#include "sub0/backbone_quant.hpp"
#include "sub0/layout.hpp"
#include "qwen_projection_profile.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sub0::dense_probe {
using sub0::gdn::Projection;
enum class PackedActivation { Float, Q8K, Q16 };
struct Layer {
    const sub0::gguf::Reader& reader;
    int index=0;
    int workers=1;
    sub0::phase::ProjectionProfile* profile=nullptr; // borrowed; caller owns it for the full decode
    PackedActivation activation_mode=PackedActivation::Float;
    sub0::denseq::Iq1Kernel iq1_kernel=sub0::denseq::Iq1Kernel::Baseline;
    std::array<std::vector<float>,8> row_scratch;
    std::vector<float> scratch, raw_output, gathered;
    sub0::quant::Q8KActivations quantized_input;
    sub0::quant::Q16Activations quantized_input16;
    std::array<const sub0::gguf::TensorInfo*, 5> projections{};
    std::array<std::vector<float>,4> captured_projections;
    std::vector<std::uint32_t> qkv_rows, z_rows, head_rows, out_indices;
    explicit Layer(const sub0::gguf::Reader& r, int layer_index=0, int thread_count=1, sub0::phase::ProjectionProfile* projection_profile=nullptr, sub0::denseq::Iq1Kernel kernel=sub0::denseq::Iq1Kernel::Baseline, PackedActivation mode=PackedActivation::Float) : reader(r), index(layer_index), workers(thread_count), profile(projection_profile), activation_mode(mode), iq1_kernel(kernel), raw_output(sub0::GDN_CONV_DIM), gathered(sub0::GDN_VALUE_DIM),
        qkv_rows(sub0::GDN_CONV_DIM), z_rows(sub0::GDN_VALUE_DIM), head_rows(sub0::GDN_V_HEADS), out_indices(sub0::GDN_VALUE_DIM) {
        if(mode==PackedActivation::Q8K)
            quantized_input=sub0::quant::Q8KActivations(std::max({D_MODEL,D_FF,sub0::GDN_VALUE_DIM,sub0::D_Q}));
        if(mode==PackedActivation::Q16)
            quantized_input16=sub0::quant::Q16Activations(std::max({D_MODEL,D_FF,sub0::GDN_VALUE_DIM,sub0::D_Q}));
        scratch.reserve(256);
        if(workers<1 || workers>8)throw std::runtime_error("projection threads must be 1..8");
#ifndef _OPENMP
        if(workers!=1)throw std::runtime_error("parallel projections require OpenMP");
#endif
        for(auto& buffer:row_scratch)buffer.reserve(256);
        captured_projections[0].resize(sub0::GDN_CONV_DIM);captured_projections[1].resize(sub0::GDN_VALUE_DIM);
        captured_projections[2].resize(sub0::GDN_V_HEADS);captured_projections[3].resize(sub0::GDN_V_HEADS);
        const char* names[]{"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_out.weight"};
        if(!sub0::GDN_SCHEDULE.full_attn[static_cast<std::size_t>(index)])
            for (int i=0;i<5;++i) { projections[i]=find(names[i]); if(!projections[i]) throw std::runtime_error(names[i]); }
        sub0::bbq::gdn_row_permutation(sub0::GDN_CONV_DIM, 2*sub0::GDN_KEY_DIM, sub0::GDN_K_HEADS, sub0::GDN_V_HEADS, sub0::GDN_V_HEAD_DIM,qkv_rows);
        sub0::bbq::gdn_row_permutation(sub0::GDN_VALUE_DIM,0,sub0::GDN_K_HEADS,sub0::GDN_V_HEADS,sub0::GDN_V_HEAD_DIM,z_rows);
        sub0::bbq::gdn_row_permutation(sub0::GDN_V_HEADS,0,sub0::GDN_K_HEADS,sub0::GDN_V_HEADS,1,head_rows);
        sub0::bbq::gdn_out_gather_index(sub0::GDN_K_HEADS,sub0::GDN_V_HEADS,sub0::GDN_V_HEAD_DIM,out_indices);
        t_ffn_gate=find("blk.0.ffn_gate.weight");
        t_ffn_up=find("blk.0.ffn_up.weight");
        t_ffn_down=find("blk.0.ffn_down.weight");
        attn_norm_weights=load("blk.0.attn_norm.weight",D_MODEL);
        post_attention_norm_weights=load("blk.0.post_attention_norm.weight",D_MODEL);
        if(sub0::GDN_SCHEDULE.full_attn[static_cast<std::size_t>(index)]){
            t_attn_q=find("blk.0.attn_q.weight");
            t_attn_k=find("blk.0.attn_k.weight");
            t_attn_v=find("blk.0.attn_v.weight");
            t_attn_output=find("blk.0.attn_output.weight");
            attn_q_norm_weights=load("blk.0.attn_q_norm.weight",D_HEAD);
            attn_k_norm_weights=load("blk.0.attn_k_norm.weight",D_HEAD);
            qg.resize(2*sub0::D_Q);q.resize(sub0::D_Q);
            k.resize(sub0::D_KV);v.resize(sub0::D_KV);att.resize(sub0::D_Q);
            scores.resize(SEQ_LEN);
        } else {
            auto cr=load("blk.0.ssm_conv1d.weight",sub0::GDN_CONV_DIM*sub0::GDN_CONV_KERNEL);
            conv_weights.resize(cr.size());
            for(int c=0;c<sub0::GDN_CONV_DIM;++c)
                for(int k_idx=0;k_idx<sub0::GDN_CONV_KERNEL;++k_idx)
                    conv_weights[c*sub0::GDN_CONV_KERNEL+k_idx]=cr[qkv_rows[c]*sub0::GDN_CONV_KERNEL+k_idx];
            auto dr=load("blk.0.ssm_dt.bias",sub0::GDN_V_HEADS);
            auto ar=load("blk.0.ssm_a",sub0::GDN_V_HEADS);
            dt_weights.resize(sub0::GDN_V_HEADS);alog_weights.resize(sub0::GDN_V_HEADS);
            for(int i=0;i<sub0::GDN_V_HEADS;++i){
                dt_weights[i]=dr[head_rows[i]];
                float raw=ar[head_rows[i]];
                if(raw>=0)throw std::runtime_error("ssm_a sign");
                alog_weights[i]=std::log(-raw);
            }
            ssm_gain_weights=load("blk.0.ssm_norm.weight",sub0::GDN_V_HEAD_DIM);
            gdn_scratch.resize(sub0::gdn::scratch_floats(sub0::GDN_DIMS,1));
        }
    }
    const sub0::gguf::TensorInfo* t_ffn_gate=nullptr;
    const sub0::gguf::TensorInfo* t_ffn_up=nullptr;
    const sub0::gguf::TensorInfo* t_ffn_down=nullptr;
    const sub0::gguf::TensorInfo* t_attn_q=nullptr;
    const sub0::gguf::TensorInfo* t_attn_k=nullptr;
    const sub0::gguf::TensorInfo* t_attn_v=nullptr;
    const sub0::gguf::TensorInfo* t_attn_output=nullptr;

    std::vector<float> attn_norm_weights;
    std::vector<float> post_attention_norm_weights;
    std::vector<float> attn_q_norm_weights;
    std::vector<float> attn_k_norm_weights;
    std::vector<float> conv_weights;
    std::vector<float> dt_weights;
    std::vector<float> alog_weights;
    std::vector<float> ssm_gain_weights;
    std::vector<float> gdn_scratch;

    std::vector<float> qg, q, k, v, att;
    std::vector<float> scores;
    const sub0::gguf::TensorInfo* find(const char* name) const {
        std::string resolved=name;
        if(resolved.starts_with("blk.0."))resolved="blk."+std::to_string(index)+resolved.substr(5);
        return reader.find_tensor(resolved);
    }
    std::vector<float> load(const char* name, std::size_t count) const {
        const auto* t=find(name); std::vector<float> values;
        if(!t || t->element_count()!=count || !sub0::gguf::to_f32(*t,reader.tensor_bytes(*t),values)) throw std::runtime_error(name);
        return values;
    }
    bool project(const sub0::gguf::TensorInfo* t, const float* x,int in,float* y,int out) {
        if(!profile)return project_impl(t,x,in,y,out);
        const auto start=sub0::phase::Clock::now();
        const bool ok=project_impl(t,x,in,y,out);
        if(t)profile->record(t->type_raw,reader.tensor_bytes(*t).size(),
                            in>0 && out>0?2ull*static_cast<std::uint64_t>(in)*static_cast<std::uint64_t>(out):0,
                            sub0::phase::milliseconds(start),ok);
        return ok;
    }
    bool project_impl(const sub0::gguf::TensorInfo* t, const float* x,int in,float* y,int out) {
        if(!t || t->dims.size()!=2 || t->dims[0]!=static_cast<std::uint64_t>(in) || t->dims[1]!=static_cast<std::uint64_t>(out))return false;
        const auto raw=reader.tensor_bytes(*t);
        if((activation_mode==PackedActivation::Q8K || activation_mode==PackedActivation::Q16) && sub0::denseq::q8k_supported(t->type_raw)){
            const bool use16=activation_mode==PackedActivation::Q16;
            if(!(use16?quantized_input16.quantize({x,static_cast<std::size_t>(in)}):quantized_input.quantize({x,static_cast<std::size_t>(in)})))return false;
            const auto bytes=static_cast<std::size_t>(in)/256*sub0::gguf::block_spec(t->type_raw).bytes;
            std::array<bool,8> ok{};
#ifdef _OPENMP
            #pragma omp parallel for num_threads(workers) schedule(static)
#endif
            for(int part=0;part<workers;++part){
                const auto begin=static_cast<std::size_t>(out)*part/workers;
                const auto end=static_cast<std::size_t>(out)*(part+1)/workers;
                if(end*bytes>raw.size()){ok[part]=false;continue;}
                ok[part]=use16?sub0::denseq::project_q16(t->type_raw,raw.subspan(begin*bytes,(end-begin)*bytes),quantized_input16,{y+begin,end-begin}):
                    sub0::denseq::project_q8k(t->type_raw,raw.subspan(begin*bytes,(end-begin)*bytes),quantized_input,{y+begin,end-begin});
            }
            return std::all_of(ok.begin(),ok.begin()+workers,[](bool value){return value;});
        }
        if(workers==1)return sub0::denseq::project(t->type_raw,raw,{x,static_cast<std::size_t>(in)},{y,static_cast<std::size_t>(out)},scratch,iq1_kernel);
        const auto spec=sub0::gguf::block_spec(t->type_raw);
        if(in<=0 || out<=0 || !spec.elems || static_cast<std::uint64_t>(in)%spec.elems)return false;
        const std::size_t row_bytes=static_cast<std::size_t>(in)/spec.elems*spec.bytes;
        if(!row_bytes || static_cast<std::size_t>(out)>raw.size()/row_bytes)return false;
        std::array<bool,8> success{};
#ifdef _OPENMP
        #pragma omp parallel for num_threads(workers) schedule(static)
#endif
        for(int part=0;part<workers;++part){
            const int begin=static_cast<int>(static_cast<std::int64_t>(out)*part/workers);
            const int end=static_cast<int>(static_cast<std::int64_t>(out)*(part+1)/workers);
            success[part]=sub0::denseq::project(t->type_raw,raw.subspan(static_cast<std::size_t>(begin)*row_bytes,static_cast<std::size_t>(end-begin)*row_bytes),{x,static_cast<std::size_t>(in)},{y+begin,static_cast<std::size_t>(end-begin)},row_scratch[part],iq1_kernel);
        }
        return std::all_of(success.begin(),success.begin()+workers,[](bool ok){return ok;});
    }
    static bool callback(void* context,Projection role,const float* x,int in,float* y,int out) {
        auto& self=*static_cast<Layer*>(context); const auto* t=self.projections[static_cast<int>(role)];
        if(role==Projection::Out) {
            if(in!=sub0::GDN_VALUE_DIM || out!=D_MODEL) return false;
            for(int i=0;i<in;++i) self.gathered[i]=x[self.out_indices[i]];
            return self.project(t,self.gathered.data(),in,y,out);
        }
        if(out>static_cast<int>(self.raw_output.size())) return false;
        if(!self.project(t,x,in,self.raw_output.data(),out))return false;
        std::copy_n(self.raw_output.data(),out,self.captured_projections[static_cast<int>(role)].data());
        const auto& rows=role==Projection::Qkv?self.qkv_rows:role==Projection::Z?self.z_rows:self.head_rows;
        for(int i=0;i<out;++i)y[i]=self.raw_output[rows[i]];
        return true;
    }
};
inline void norm(const std::vector<float>& x,const std::vector<float>& gain,std::vector<float>& out) {
    float ms=0;for(float v:x)ms+=v*v;
    const float scale=1.f/std::sqrt(ms/x.size()+1e-6f);
    for(std::size_t i=0;i<x.size();++i)out[i]=x[i]*scale*gain[i];
}
}
