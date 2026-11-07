// Real GGUF layer-0 proving ground: embedding -> GDN mixer -> dense SwiGLU FFN.
// Partial inference only: no later layers, output head, tokenizer or generated text.
#include "sub0/dense_quant.hpp"
#include "sub0/file_map.hpp"
#include "sub0/backbone_quant.hpp"
#include "sub0/layout.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
using sub0::gdn::Projection;
struct Layer {
    const sub0::gguf::Reader& reader;
    std::vector<float> scratch, raw_output, gathered;
    std::array<const sub0::gguf::TensorInfo*, 5> projections{};
    std::vector<std::uint32_t> qkv_rows, z_rows, head_rows, out_indices;
    explicit Layer(const sub0::gguf::Reader& r) : reader(r), raw_output(sub0::GDN_CONV_DIM), gathered(sub0::GDN_VALUE_DIM),
        qkv_rows(sub0::GDN_CONV_DIM), z_rows(sub0::GDN_VALUE_DIM), head_rows(sub0::GDN_V_HEADS), out_indices(sub0::GDN_VALUE_DIM) {
        scratch.reserve(256);
        const char* names[]{"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_out.weight"};
        for (int i=0;i<5;++i) { projections[i]=reader.find_tensor(names[i]); if(!projections[i]) throw std::runtime_error(names[i]); }
        sub0::bbq::gdn_row_permutation(sub0::GDN_CONV_DIM, 2*sub0::GDN_KEY_DIM, sub0::GDN_K_HEADS, sub0::GDN_V_HEADS, sub0::GDN_V_HEAD_DIM,qkv_rows);
        sub0::bbq::gdn_row_permutation(sub0::GDN_VALUE_DIM,0,sub0::GDN_K_HEADS,sub0::GDN_V_HEADS,sub0::GDN_V_HEAD_DIM,z_rows);
        sub0::bbq::gdn_row_permutation(sub0::GDN_V_HEADS,0,sub0::GDN_K_HEADS,sub0::GDN_V_HEADS,1,head_rows);
        sub0::bbq::gdn_out_gather_index(sub0::GDN_K_HEADS,sub0::GDN_V_HEADS,sub0::GDN_V_HEAD_DIM,out_indices);
    }
    std::vector<float> load(const char* name, std::size_t count) const {
        const auto* t=reader.find_tensor(name); std::vector<float> values;
        if(!t || t->element_count()!=count || !sub0::gguf::to_f32(*t,reader.tensor_bytes(*t),values)) throw std::runtime_error(name);
        return values;
    }
    bool project(const sub0::gguf::TensorInfo* t, const float* x,int in,float* y,int out) {
        return t && t->dims.size()==2 && t->dims[0]==static_cast<std::uint64_t>(in) && t->dims[1]==static_cast<std::uint64_t>(out) &&
            sub0::denseq::project(t->type_raw,reader.tensor_bytes(*t),{x,static_cast<std::size_t>(in)},{y,static_cast<std::size_t>(out)},scratch);
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
        const auto& rows=role==Projection::Qkv?self.qkv_rows:role==Projection::Z?self.z_rows:self.head_rows;
        for(int i=0;i<out;++i)y[i]=self.raw_output[rows[i]];
        return true;
    }
};
void norm(const std::vector<float>& x,const std::vector<float>& gain,std::vector<float>& out) {
    float ms=0;for(float v:x)ms+=v*v;
    const float scale=1.f/std::sqrt(ms/x.size()+1e-6f);
    for(std::size_t i=0;i<x.size();++i)out[i]=x[i]*scale*gain[i];
}
}
int main(int argc,char**argv) try {
    if(argc!=2 && argc!=3){std::fprintf(stderr,"usage: %s MODEL.gguf [HIDDEN.f32]\n",argv[0]);return 2;}
    static_assert(sub0::GDN_DIMS.output_gate_silu,"generate the Qwen35 SiLU profile");
    sub0::FileMap file;std::string error;if(!file.open(argv[1],error))throw std::runtime_error(error);
    const sub0::gguf::Reader reader({file.data(),file.size()});
    if(!reader.ok())throw std::runtime_error("invalid GGUF");
    const auto* arch=reader.find("general.architecture");if(!arch||arch->s!="qwen35")throw std::runtime_error("requires qwen35");
    const auto dump_stage=[&](const char* stage,const std::vector<float>& values) {
        if(argc!=3)return;
        const std::string path=std::string(argv[2])+"."+stage;
        FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("stage dump open");
        const bool written=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();
        const int closed=std::fclose(f);if(!written||closed!=0)throw std::runtime_error("stage dump write");
    };
    Layer layer(reader); const auto& d=sub0::GDN_DIMS;
    const auto* embedding=reader.find_tensor("token_embd.weight");
    constexpr std::size_t token=1543;
    if(!embedding||embedding->dims.size()!=2||embedding->dims[0]!=D_MODEL||embedding->dims[1]<=token)throw std::runtime_error("embedding geometry");
    const auto bs=sub0::gguf::block_spec(embedding->type_raw);const auto decode=sub0::gguf::dequant_fn(embedding->type_raw);
    if(!decode||D_MODEL%bs.elems)throw std::runtime_error("embedding format");
    const std::size_t rb=D_MODEL/bs.elems*bs.bytes;const auto raw=reader.tensor_bytes(*embedding);
    if(raw.size()<(token+1)*rb)throw std::runtime_error("embedding truncated");
    std::vector<float> h,a(D_MODEL),mixer(D_MODEL),state(sub0::gdn::state_floats(d)),history(sub0::gdn::conv_hist_floats(d)),scratch(sub0::gdn::scratch_floats(d,1));
    if(!decode(raw.subspan(token*rb,rb),D_MODEL,h))throw std::runtime_error("embedding decode");
    norm(h,layer.load("blk.0.attn_norm.weight",D_MODEL),a);
    dump_stage("attn_norm-0",a);
    auto conv_raw=layer.load("blk.0.ssm_conv1d.weight",sub0::GDN_CONV_DIM*sub0::GDN_CONV_KERNEL),conv=conv_raw;
    for(int c=0;c<sub0::GDN_CONV_DIM;++c)for(int k=0;k<sub0::GDN_CONV_KERNEL;++k)conv[c*sub0::GDN_CONV_KERNEL+k]=conv_raw[layer.qkv_rows[c]*sub0::GDN_CONV_KERNEL+k];
    auto dt_raw=layer.load("blk.0.ssm_dt.bias",sub0::GDN_V_HEADS),alog_raw=layer.load("blk.0.ssm_a",sub0::GDN_V_HEADS),dt=dt_raw,alog=alog_raw;
    for(int i=0;i<sub0::GDN_V_HEADS;++i){dt[i]=dt_raw[layer.head_rows[i]];if(alog_raw[layer.head_rows[i]]>=0)throw std::runtime_error("ssm_a sign");alog[i]=std::log(-alog_raw[layer.head_rows[i]]);}
    const auto gain=layer.load("blk.0.ssm_norm.weight",sub0::GDN_V_HEAD_DIM);
    sub0::gdn::Native native{};native.project=&Layer::callback;native.project_context=&layer;const float* none=nullptr;
    sub0::gdn::forward<false,1,const float*>(d,1,a.data(),none,none,none,none,conv.data(),dt.data(),alog.data(),gain.data(),none,state.data(),history.data(),mixer.data(),scratch.data(),&native);
    dump_stage("linear_attn_out-0",mixer);
    for(int i=0;i<D_MODEL;++i)h[i]+=mixer[i];
    dump_stage("attn_residual-0",h);
    norm(h,layer.load("blk.0.post_attention_norm.weight",D_MODEL),a);
    dump_stage("attn_post_norm-0",a);
    std::vector<float> gate(D_FF),up(D_FF),down(D_MODEL);
    if(!layer.project(reader.find_tensor("blk.0.ffn_gate.weight"),a.data(),D_MODEL,gate.data(),D_FF)||!layer.project(reader.find_tensor("blk.0.ffn_up.weight"),a.data(),D_MODEL,up.data(),D_FF))throw std::runtime_error("FFN input projection");
    for(int i=0;i<D_FF;++i){const float s=gate[i]>=0?1.f/(1.f+std::exp(-gate[i])):std::exp(gate[i])/(1.f+std::exp(gate[i]));up[i]*=gate[i]*s;}
    if(!layer.project(reader.find_tensor("blk.0.ffn_down.weight"),up.data(),D_FF,down.data(),D_MODEL))throw std::runtime_error("FFN output projection");
    dump_stage("ffn_out-0",down);
    double sum=0,sq=0;for(int i=0;i<D_MODEL;++i){h[i]+=down[i];if(!std::isfinite(h[i]))throw std::runtime_error("nonfinite layer output");sum+=h[i];sq+=h[i]*h[i];}
    if(argc==3) {
        FILE* dump=std::fopen(argv[2],"wb");if(!dump)throw std::runtime_error("cannot open hidden dump");
        const bool written=std::fwrite(h.data(),sizeof(float),h.size(),dump)==h.size();
        const int closed=std::fclose(dump);if(!written||closed!=0)throw std::runtime_error("hidden dump write failed");
    }
    std::printf("partial layer-0 inference: token=%zu width=%d mean=%.9g rms=%.9g; mapped GDN+FFN, no later layers or logits\n",token,D_MODEL,sum/D_MODEL,std::sqrt(sq/D_MODEL));
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
