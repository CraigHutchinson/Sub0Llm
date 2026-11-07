// Mapped full-model raw-token sequence proving ground, with persistent GDN/KV state.
#include "dense_qwen_support.hpp"
#include <charconv>
using namespace sub0::dense_probe;
int main(int argc,char** argv) try {
    if(argc<2||argc>5){std::fprintf(stderr,"usage: %s MODEL.gguf [TOKEN_IDS [DUMP_PREFIX [CONTINUATION_STEPS]]]\n",argv[0]);return 2;}
    static_assert(sub0::GDN_DIMS.output_gate_silu && USE_ATTENTION_GATE && !sub0::USE_MOE);
    std::vector<int> tokens;
    const std::string prompt=argc>=3?argv[2]:"1543";
    std::size_t begin=0;
    while(begin<=prompt.size()) {
        const auto end=prompt.find(',',begin);const auto stop=end==std::string::npos?prompt.size():end;
        int id=-1;const auto result=std::from_chars(prompt.data()+begin,prompt.data()+stop,id);
        if(result.ec!=std::errc{}||result.ptr!=prompt.data()+stop||id<0||id>=VOCAB)throw std::runtime_error("invalid token");
        tokens.push_back(id);if(end==std::string::npos)break;begin=end+1;
    }
    int generate=0;
    if(argc==5){const std::string n=argv[4];auto r=std::from_chars(n.data(),n.data()+n.size(),generate);if(r.ec!=std::errc{}||r.ptr!=n.data()+n.size()||generate<0)throw std::runtime_error("invalid generation count");}
    if(tokens.size()>SEQ_LEN || static_cast<std::size_t>(generate)>SEQ_LEN-tokens.size())throw std::runtime_error("context window exceeded");
    const std::size_t prompt_size=tokens.size();tokens.reserve(prompt_size+generate);
    sub0::FileMap file;std::string error;if(!file.open(argv[1],error))throw std::runtime_error(error);
    const sub0::gguf::Reader reader({file.data(),file.size()});
    const auto* arch=reader.find("general.architecture");if(!reader.ok()||!arch||arch->s!="qwen35")throw std::runtime_error("requires qwen35 GGUF");
    const auto* epsilon=reader.find("qwen35.attention.layer_norm_rms_epsilon");
    if(!epsilon || epsilon->kind!=sub0::gguf::Value::Kind::Float || std::abs(epsilon->f-1e-6)>1e-12)
        throw std::runtime_error("requires qwen35 RMS epsilon 1e-6");
    std::size_t position=0;
    auto dump=[&](const std::string& suffix,const std::vector<float>& values){if(argc<4 || std::string(argv[3])=="-")return;const std::string path=std::string(argv[3])+(position?".pos"+std::to_string(position):"")+suffix;FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("dump open");bool ok=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();int closed=std::fclose(f);if(!ok||closed)throw std::runtime_error("dump write");};
    Layer initial(reader);const auto* emb=reader.find_tensor("token_embd.weight");
    if(!emb||emb->dims.size()!=2||emb->dims[0]!=D_MODEL||emb->dims[1]!=VOCAB)throw std::runtime_error("embedding dimensions");
    const auto spec=sub0::gguf::block_spec(emb->type_raw);auto decode=sub0::gguf::dequant_fn(emb->type_raw);
    if(!decode||D_MODEL%spec.elems)throw std::runtime_error("embedding format");
    const std::size_t row_bytes=D_MODEL/spec.elems*spec.bytes;auto bytes=reader.tensor_bytes(*emb);if(bytes.size()<row_bytes*VOCAB)throw std::runtime_error("embedding truncated");
    std::vector<float> h,a(D_MODEL),mixer(D_MODEL),gate(D_FF),up(D_FF),down(D_MODEL);
    struct Cache {std::vector<float> state,history,k,v;};
    std::vector<Cache> caches(N_LAYERS);
    for(int l=0;l<N_LAYERS;++l) {
        auto& c=caches[l];
        if(sub0::GDN_SCHEDULE.full_attn[l]){c.k.resize(static_cast<std::size_t>(SEQ_LEN)*sub0::D_KV);c.v.resize(c.k.size());}
        else {c.state.resize(sub0::gdn::state_floats(sub0::GDN_DIMS));c.history.resize(sub0::gdn::conv_hist_floats(sub0::GDN_DIMS));}
    }
    for(position=0;position<tokens.size();++position) {
    const int token=tokens[position];
    if(!decode(bytes.subspan(static_cast<std::size_t>(token)*row_bytes,row_bytes),D_MODEL,h))throw std::runtime_error("embedding decode");
    const auto& d=sub0::GDN_DIMS;
    for(int l=0;l<N_LAYERS;++l){
        Layer layer(reader,l);norm(h,layer.load("blk.0.attn_norm.weight",D_MODEL),a);
        if(sub0::GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
            std::vector<float> qg(2*sub0::D_Q),q(sub0::D_Q),k(sub0::D_KV),v(sub0::D_KV),att(sub0::D_Q);
            if(!layer.project(layer.find("blk.0.attn_q.weight"),a.data(),D_MODEL,qg.data(),2*sub0::D_Q)||!layer.project(layer.find("blk.0.attn_k.weight"),a.data(),D_MODEL,k.data(),sub0::D_KV)||!layer.project(layer.find("blk.0.attn_v.weight"),a.data(),D_MODEL,v.data(),sub0::D_KV))throw std::runtime_error("full attention projection");
            for(int hd=0;hd<N_HEADS;++hd)for(int j=0;j<D_HEAD;++j)q[hd*D_HEAD+j]=qg[hd*2*D_HEAD+j];
            auto normalize_rope=[&](std::vector<float>& values,int heads,const std::vector<float>& gain){
                for(int hd=0;hd<heads;++hd){float ms=0;for(int j=0;j<D_HEAD;++j)ms+=values[hd*D_HEAD+j]*values[hd*D_HEAD+j];
                    float inv=1.f/std::sqrt(ms/D_HEAD+1e-6f);for(int j=0;j<D_HEAD;++j)values[hd*D_HEAD+j]*=gain[j]*inv;
                    for(int j=0;j<sub0::ROTARY_HALF;++j){float angle=static_cast<float>(position)*std::pow(ROPE_THETA,-2.f*j/ROTARY_DIM);float cs=std::cos(angle),sn=std::sin(angle);int i=hd*D_HEAD+j;float x=values[i],y=values[i+sub0::ROTARY_HALF];values[i]=x*cs-y*sn;values[i+sub0::ROTARY_HALF]=x*sn+y*cs;}
                }
            };
            normalize_rope(q,N_HEADS,layer.load("blk.0.attn_q_norm.weight",D_HEAD));normalize_rope(k,N_KV_HEADS,layer.load("blk.0.attn_k_norm.weight",D_HEAD));
            auto& c=caches[l];std::copy(k.begin(),k.end(),c.k.begin()+position*sub0::D_KV);std::copy(v.begin(),v.end(),c.v.begin()+position*sub0::D_KV);
            std::vector<float> scores(position+1);
            for(int hd=0;hd<N_HEADS;++hd){const int kv=hd/sub0::GQA_GROUP;float mx=-INFINITY;
                for(std::size_t t=0;t<=position;++t){float dot=0;for(int j=0;j<D_HEAD;++j)dot+=q[hd*D_HEAD+j]*c.k[t*sub0::D_KV+kv*D_HEAD+j];scores[t]=dot/std::sqrt(static_cast<float>(D_HEAD));mx=std::max(mx,scores[t]);}
                float sum=0;for(float& score:scores){score=std::exp(score-mx);sum+=score;}
                for(int j=0;j<D_HEAD;++j){float value=0;for(std::size_t t=0;t<=position;++t)value+=scores[t]/sum*c.v[t*sub0::D_KV+kv*D_HEAD+j];float x=qg[hd*2*D_HEAD+D_HEAD+j];float sig=x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));att[hd*D_HEAD+j]=value*sig;}
            }
            if(!layer.project(layer.find("blk.0.attn_output.weight"),att.data(),sub0::D_Q,mixer.data(),D_MODEL))throw std::runtime_error("attention output");
        } else {
            auto cr=layer.load("blk.0.ssm_conv1d.weight",sub0::GDN_CONV_DIM*sub0::GDN_CONV_KERNEL),conv=cr;
            for(int c=0;c<sub0::GDN_CONV_DIM;++c)for(int k=0;k<sub0::GDN_CONV_KERNEL;++k)conv[c*sub0::GDN_CONV_KERNEL+k]=cr[layer.qkv_rows[c]*sub0::GDN_CONV_KERNEL+k];
            auto dr=layer.load("blk.0.ssm_dt.bias",sub0::GDN_V_HEADS),ar=layer.load("blk.0.ssm_a",sub0::GDN_V_HEADS),dt=dr,alog=ar;
            for(int i=0;i<sub0::GDN_V_HEADS;++i){dt[i]=dr[layer.head_rows[i]];float raw=ar[layer.head_rows[i]];if(raw>=0)throw std::runtime_error("ssm_a sign");alog[i]=std::log(-raw);}
            const auto gain=layer.load("blk.0.ssm_norm.weight",sub0::GDN_V_HEAD_DIM);
            std::vector<float> scratch(sub0::gdn::scratch_floats(d,1));
            auto& state=caches[l].state;auto& history=caches[l].history;
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
    std::printf("mapped sequence inference: position=%zu input=%d layers=%d logits=%zu top_token=%zu top_logit=%.9g; raw token interface\n",position,token,N_LAYERS,logits.size(),static_cast<std::size_t>(best-logits.begin()),*best);
    if(position+1==tokens.size() && tokens.size()<prompt_size+static_cast<std::size_t>(generate))tokens.push_back(static_cast<int>(best-logits.begin()));
    }
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
