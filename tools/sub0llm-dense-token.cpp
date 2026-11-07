// Full-model, single raw token at position zero. No sequence cache or tokenizer.
#include "dense_qwen_support.hpp"
#include <charconv>
using namespace sub0::dense_probe;
int main(int argc,char** argv) try {
    if(argc<2||argc>4){std::fprintf(stderr,"usage: %s MODEL.gguf [TOKEN_ID [DUMP_PREFIX]]\n",argv[0]);return 2;}
    static_assert(sub0::GDN_DIMS.output_gate_silu && USE_ATTENTION_GATE && !sub0::USE_MOE);
    int token=1543;
    if(argc>=3){const std::string value=argv[2];const auto result=std::from_chars(value.data(),value.data()+value.size(),token);if(result.ec!=std::errc{}||result.ptr!=value.data()+value.size()||token<0||token>=VOCAB)throw std::runtime_error("invalid token");}
    if(token<0 || token>=VOCAB)throw std::runtime_error("default token outside configured vocabulary");
    sub0::FileMap file;std::string error;if(!file.open(argv[1],error))throw std::runtime_error(error);
    const sub0::gguf::Reader reader({file.data(),file.size()});
    const auto* arch=reader.find("general.architecture");if(!reader.ok()||!arch||arch->s!="qwen35")throw std::runtime_error("requires qwen35 GGUF");
    const auto* epsilon=reader.find("qwen35.attention.layer_norm_rms_epsilon");
    if(!epsilon || epsilon->kind!=sub0::gguf::Value::Kind::Float || std::abs(epsilon->f-1e-6)>1e-12)
        throw std::runtime_error("requires qwen35 RMS epsilon 1e-6");
    auto dump=[&](const std::string& suffix,const std::vector<float>& values){if(argc!=4)return;const std::string path=std::string(argv[3])+suffix;FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("dump open");bool ok=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();int closed=std::fclose(f);if(!ok||closed)throw std::runtime_error("dump write");};
    Layer initial(reader);const auto* emb=reader.find_tensor("token_embd.weight");
    if(!emb||emb->dims.size()!=2||emb->dims[0]!=D_MODEL||emb->dims[1]!=VOCAB)throw std::runtime_error("embedding dimensions");
    const auto spec=sub0::gguf::block_spec(emb->type_raw);auto decode=sub0::gguf::dequant_fn(emb->type_raw);
    if(!decode||D_MODEL%spec.elems)throw std::runtime_error("embedding format");
    const std::size_t row_bytes=D_MODEL/spec.elems*spec.bytes;auto bytes=reader.tensor_bytes(*emb);if(bytes.size()<row_bytes*VOCAB)throw std::runtime_error("embedding truncated");
    std::vector<float> h,a(D_MODEL),mixer(D_MODEL),gate(D_FF),up(D_FF),down(D_MODEL);
    if(!decode(bytes.subspan(static_cast<std::size_t>(token)*row_bytes,row_bytes),D_MODEL,h))throw std::runtime_error("embedding decode");
    const auto& d=sub0::GDN_DIMS;
    for(int l=0;l<N_LAYERS;++l){
        Layer layer(reader,l);norm(h,layer.load("blk.0.attn_norm.weight",D_MODEL),a);
        if(sub0::GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
            // At a single position, softmax has one element equal to one. Q/K and RoPE do not
            // affect the attention result. The fused Q|gate projection still supplies its gate.
            std::vector<float> qg(2*sub0::D_Q),v(sub0::D_KV),att(sub0::D_Q);
            if(!layer.project(layer.find("blk.0.attn_q.weight"),a.data(),D_MODEL,qg.data(),2*sub0::D_Q)||!layer.project(layer.find("blk.0.attn_v.weight"),a.data(),D_MODEL,v.data(),sub0::D_KV))throw std::runtime_error("full attention projection");
            for(int hd=0;hd<N_HEADS;++hd)for(int j=0;j<D_HEAD;++j){float x=qg[hd*2*D_HEAD+D_HEAD+j];float s=x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));att[hd*D_HEAD+j]=v[(hd/sub0::GQA_GROUP)*D_HEAD+j]*s;}
            if(!layer.project(layer.find("blk.0.attn_output.weight"),att.data(),sub0::D_Q,mixer.data(),D_MODEL))throw std::runtime_error("attention output");
        } else {
            auto cr=layer.load("blk.0.ssm_conv1d.weight",sub0::GDN_CONV_DIM*sub0::GDN_CONV_KERNEL),conv=cr;
            for(int c=0;c<sub0::GDN_CONV_DIM;++c)for(int k=0;k<sub0::GDN_CONV_KERNEL;++k)conv[c*sub0::GDN_CONV_KERNEL+k]=cr[layer.qkv_rows[c]*sub0::GDN_CONV_KERNEL+k];
            auto dr=layer.load("blk.0.ssm_dt.bias",sub0::GDN_V_HEADS),ar=layer.load("blk.0.ssm_a",sub0::GDN_V_HEADS),dt=dr,alog=ar;
            for(int i=0;i<sub0::GDN_V_HEADS;++i){dt[i]=dr[layer.head_rows[i]];float raw=ar[layer.head_rows[i]];if(raw>=0)throw std::runtime_error("ssm_a sign");alog[i]=std::log(-raw);}
            const auto gain=layer.load("blk.0.ssm_norm.weight",sub0::GDN_V_HEAD_DIM);
            std::vector<float> state(sub0::gdn::state_floats(d)),history(sub0::gdn::conv_hist_floats(d)),scratch(sub0::gdn::scratch_floats(d,1));
            sub0::gdn::Native native{};native.project=&Layer::callback;native.project_context=&layer;const float* np=nullptr;
            sub0::gdn::forward<false,1,const float*>(d,1,a.data(),np,np,np,np,conv.data(),dt.data(),alog.data(),gain.data(),np,state.data(),history.data(),mixer.data(),scratch.data(),&native);
        }
        for(int i=0;i<D_MODEL;++i)h[i]+=mixer[i];
        norm(h,layer.load("blk.0.post_attention_norm.weight",D_MODEL),a);
        if(!layer.project(layer.find("blk.0.ffn_gate.weight"),a.data(),D_MODEL,gate.data(),D_FF)||!layer.project(layer.find("blk.0.ffn_up.weight"),a.data(),D_MODEL,up.data(),D_FF))throw std::runtime_error("FFN projections");
        for(int i=0;i<D_FF;++i){float x=gate[i];float s=x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));up[i]*=x*s;}
        if(!layer.project(layer.find("blk.0.ffn_down.weight"),up.data(),D_FF,down.data(),D_MODEL))throw std::runtime_error("FFN down");
        for(int i=0;i<D_MODEL;++i){h[i]+=down[i];if(!std::isfinite(h[i]))throw std::runtime_error("nonfinite hidden");}
        dump(".layer"+std::to_string(l),h);std::fprintf(stderr,"layer %d/%d complete\n",l+1,N_LAYERS);
    }
    norm(h,initial.load("output_norm.weight",D_MODEL),a);dump(".norm",a);
    std::vector<float> logits(VOCAB);
    if(!initial.project(reader.find_tensor("output.weight"),a.data(),D_MODEL,logits.data(),VOCAB))throw std::runtime_error("LM head");
    for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");
    dump(".logits",logits);const auto best=std::max_element(logits.begin(),logits.end());
    std::printf("single-token full-model inference: input=%d layers=%d logits=%zu top_token=%zu top_logit=%.9g; no sequence cache or tokenizer\n",token,N_LAYERS,logits.size(),static_cast<std::size_t>(best-logits.begin()),*best);
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
