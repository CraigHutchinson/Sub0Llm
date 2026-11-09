// Mapped full-model raw-token sequence proving ground, with persistent GDN/KV state.
#include "dense_qwen_support.hpp"
#include <charconv>
#include "sub0/qwen_tokenizer.hpp"
#include <CLI/CLI.hpp>
#include <filesystem>
#include "qwen_phase_metrics.hpp"
#include "qwen_quality_metrics.hpp"
using namespace sub0::dense_probe;
#ifndef SUB0_DENSE_PROJECTION_DEFAULT
#define SUB0_DENSE_PROJECTION_DEFAULT 0
#endif
static_assert(SUB0_DENSE_PROJECTION_DEFAULT>=0 && SUB0_DENSE_PROJECTION_DEFAULT<=4);
constexpr const char* configured_kernel_name(){
    constexpr const char* names[]{"baseline","scalar","native","native-all","fused-iq2"};
    return names[SUB0_DENSE_PROJECTION_DEFAULT];
}
int main(int argc,char** argv) try {
    std::string model,prompt="1543",dump_prefix="-",text,response_file,timings_file,projection_profile_file;
    bool inspect_context=false,logits_only=false;
    std::string score_file,score_targets;
    std::string tokenizer_dir=(std::filesystem::path(argv[0]).parent_path()/"tokenizer").string();
    int generate=0;
    int projection_threads=SUB0_DENSE_PROJECTION_DEFAULT>=3?std::min(8,DEFAULT_THREADS):1;
    std::string iq1_kernel_name=configured_kernel_name();
    CLI::App app{"Mapped Qwen raw-token or text sequence proving ground"};
    app.add_option("model",model,"GGUF model")->required();
    auto* raw_option=app.add_option("tokens",prompt,"Comma-separated raw token IDs (default 1543)");
    app.add_option("dump,--dump-prefix",dump_prefix,"Dump prefix, or - to disable");
    app.add_option("steps,--continuation-steps",generate,"Greedy feedback positions after prompt")->check(CLI::Range(0,SEQ_LEN));
    auto* text_option=app.add_option("--text",text,"Literal prompt text; no chat template or implicit special tokens");
    app.add_option("--tokenizer-dir",tokenizer_dir,"Tokenizer exported from this GGUF");
    app.add_option("--projection-threads",projection_threads,"Independent output-row workers (OpenMP)")->check(CLI::Range(1,8));
    // Evaluation override; the default comes from the generated configuration (#63).
    // TODO(#29): keep runtime strategy dispatch in this proving ground; instantiate the
    // qualified generated strategy with if constexpr in the production engine adapter.
    app.add_option("--iq1s-kernel,--projection-kernel",iq1_kernel_name,"Projection kernel; generated default, explicit experiment override")->capture_default_str()->check(CLI::IsMember({"baseline","scalar","native","native-all","fused-iq2"}));
    app.add_option("--timings-file",timings_file,"Native setup/prefill/decode metrics; suppress progress output; no dumps");
    app.add_option("--projection-profile",projection_profile_file,"Diagnostic per-format projection attribution; requires timings");
    app.add_option("--score-file",score_file,"Teacher-forced f64 NLL JSON; requires raw targets");
    app.add_option("--score-targets",score_targets,"One raw next-token target per input position");
    app.add_flag("--logits-only",logits_only,"Dump only logits, retaining every requested position");
    app.add_flag("--inspect-context",inspect_context,"Validate prompt and report token budget without inference");
    app.add_option("--response-file",response_file,"Write decoded text continuation after successful inference");
    try {app.parse(argc,argv);}catch(const CLI::ParseError& e){return app.exit(e);}
    const auto iq1_kernel=iq1_kernel_name=="fused-iq2"?sub0::denseq::Iq1Kernel::FusedIQ2:
                          iq1_kernel_name=="native-all"?sub0::denseq::Iq1Kernel::NativeAll:
                          iq1_kernel_name=="native"?sub0::denseq::Iq1Kernel::Native:
                          iq1_kernel_name=="scalar"?sub0::denseq::Iq1Kernel::Scalar:sub0::denseq::Iq1Kernel::Baseline;
    const bool measure=!timings_file.empty(),scoring=!score_file.empty(),quiet=measure||scoring;
    if(quiet)std::printf("KERNEL selected=%s threads=%d configured=%s\n",iq1_kernel_name.c_str(),projection_threads,configured_kernel_name());
    if(scoring!=!score_targets.empty())throw std::runtime_error("score file and targets are required together");
    if(scoring && (measure || inspect_context || generate || text_option->count()))throw std::runtime_error("scoring requires raw teacher-forced inputs without generation/timing");
    if(logits_only && dump_prefix=="-")throw std::runtime_error("logits-only requires a dump prefix");
    if(!projection_profile_file.empty() && !measure)throw std::runtime_error("projection profile requires timings");
    if(measure && (inspect_context || dump_prefix!="-"))throw std::runtime_error("timings require inference without dumps");
    sub0::phase::Metrics metrics;
    sub0::phase::ProjectionProfile projection_profile;
    auto* profiler=projection_profile_file.empty()?nullptr:&projection_profile;
    const bool text_mode=text_option->count()!=0;
    if(text_mode&&raw_option->count())throw std::runtime_error("choose text or raw token input");
    static_assert(sub0::GDN_DIMS.output_gate_silu && USE_ATTENTION_GATE && !sub0::USE_MOE);
    sub0::qwen_tok::Tokenizer tokenizer;
    std::vector<int> tokens,targets;
    if(scoring){
        std::size_t at=0;
        while(at<=score_targets.size()){
            const auto comma=score_targets.find(',',at);const auto stop=comma==std::string::npos?score_targets.size():comma;
            int id=-1;const auto parsed=std::from_chars(score_targets.data()+at,score_targets.data()+stop,id);
            if(parsed.ec!=std::errc{} || parsed.ptr!=score_targets.data()+stop || id<0 || id>=VOCAB)throw std::runtime_error("invalid score target");
            targets.push_back(id);if(comma==std::string::npos)break;at=comma+1;
        }
    }
    std::size_t begin=0;
    if(!text_mode) while(begin<=prompt.size()) {
        const auto end=prompt.find(',',begin);const auto stop=end==std::string::npos?prompt.size():end;
        int id=-1;const auto result=std::from_chars(prompt.data()+begin,prompt.data()+stop,id);
        if(result.ec!=std::errc{}||result.ptr!=prompt.data()+stop||id<0||id>=VOCAB)throw std::runtime_error("invalid token");
        tokens.push_back(id);if(end==std::string::npos)break;begin=end+1;
    }
    if(!text_mode && (tokens.size()>SEQ_LEN || static_cast<std::size_t>(generate)>SEQ_LEN-tokens.size()))
        throw std::runtime_error("context window exceeded");
    const auto setup_begin=measure?sub0::phase::Clock::now():sub0::phase::Clock::time_point{};
    sub0::FileMap file;std::string error;if(!file.open(model,error))throw std::runtime_error(error);
    const sub0::gguf::Reader reader({file.data(),file.size()});
    const auto* arch=reader.find("general.architecture");if(!reader.ok()||!arch||arch->s!="qwen35")throw std::runtime_error("requires qwen35 GGUF");
    const auto* epsilon=reader.find("qwen35.attention.layer_norm_rms_epsilon");
    if(!epsilon || epsilon->kind!=sub0::gguf::Value::Kind::Float || std::abs(epsilon->f-1e-6)>1e-12)throw std::runtime_error("requires qwen35 RMS epsilon 1e-6");
    if(text_mode){
        const std::filesystem::path dir=tokenizer_dir;
        if(!tokenizer.load((dir/"vocab.json").string(),(dir/"merges.txt").string(),(dir/"tokenizer_config.json").string(),&error))throw std::runtime_error(error);
        const auto* eos=reader.find("tokenizer.ggml.eos_token_id");
        if(!eos || (eos->kind!=sub0::gguf::Value::Kind::UInt && eos->kind!=sub0::gguf::Value::Kind::Int))throw std::runtime_error("missing model EOS");
        const auto model_eos=eos->kind==sub0::gguf::Value::Kind::UInt?eos->u:static_cast<std::uint64_t>(eos->i);
        if(static_cast<std::uint64_t>(tokenizer.eos_id())!=model_eos)throw std::runtime_error("tokenizer EOS differs from model");
        const auto* types=reader.find("tokenizer.ggml.token_type");
        if(!types || static_cast<std::size_t>(tokenizer.vocab_size())!=static_cast<std::size_t>(std::count_if(types->ia32.begin(),types->ia32.end(),[](int type){return type!=5;})))
            throw std::runtime_error("tokenizer vocabulary differs from model");
        tokenizer.encode(text,tokens);
        std::fprintf(stderr,"text prompt tokens:");for(int id:tokens)std::fprintf(stderr," %d",id);std::fprintf(stderr,"\n");
    }
    if(tokens.empty())throw std::runtime_error("empty prompt");
    for(int id:tokens)if(id<0||id>=VOCAB)throw std::runtime_error("token outside model vocabulary");
    std::printf("CONTEXT prompt=%zu reserve=%d capacity=%d\n",tokens.size(),generate,SEQ_LEN);
    if(tokens.size()>SEQ_LEN || static_cast<std::size_t>(generate)>SEQ_LEN-tokens.size())throw std::runtime_error("context window exceeded");
    if(inspect_context)return 0;
    if(!text_mode && !response_file.empty())throw std::runtime_error("response file requires text mode");
    const std::size_t prompt_size=tokens.size();tokens.reserve(prompt_size+generate);
    if(scoring && targets.size()!=prompt_size)throw std::runtime_error("score target count differs from inputs");
    sub0::quality::Scores scores;if(scoring)scores.reserve(prompt_size);
    std::vector<int> generated;
    bool stopped_eos=false;
    std::size_t position=0;
    auto dump=[&](const std::string& suffix,const std::vector<float>& values){if(dump_prefix=="-" || (logits_only && suffix!=".logits"))return;const std::string path=dump_prefix+(position?".pos"+std::to_string(position):"")+suffix;FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("dump open");bool ok=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();int closed=std::fclose(f);if(!ok||closed)throw std::runtime_error("dump write");};
    Layer initial(reader,0,projection_threads,profiler,iq1_kernel);const auto* emb=reader.find_tensor("token_embd.weight");
    if(!emb||emb->dims.size()!=2||emb->dims[0]!=D_MODEL||emb->dims[1]!=VOCAB)throw std::runtime_error("embedding dimensions");
    const auto spec=sub0::gguf::block_spec(emb->type_raw);auto decode=sub0::gguf::dequant_fn(emb->type_raw);
    if(!decode||D_MODEL%spec.elems)throw std::runtime_error("embedding format");
    const std::size_t row_bytes=D_MODEL/spec.elems*spec.bytes;auto bytes=reader.tensor_bytes(*emb);if(bytes.size()<row_bytes*VOCAB)throw std::runtime_error("embedding truncated");
    if(measure){
        for(const auto& tensor:reader.tensors()){
            if(tensor.name=="token_embd.weight")continue;
            metrics.weight_bytes_per_eval+=reader.tensor_bytes(tensor).size();
            if(tensor.dims.size()==2)metrics.projection_flops_per_eval+=2*tensor.element_count();
        }
        metrics.weight_bytes_per_eval+=row_bytes;
    }
    std::vector<float> h,a(D_MODEL),mixer(D_MODEL),gate(D_FF),up(D_FF),down(D_MODEL);
    struct Cache {std::vector<float> state,history,k,v;};
    std::vector<Cache> caches(N_LAYERS);
    for(int l=0;l<N_LAYERS;++l) {
        auto& c=caches[l];
        if(sub0::GDN_SCHEDULE.full_attn[l]){c.k.resize(static_cast<std::size_t>(SEQ_LEN)*sub0::D_KV);c.v.resize(c.k.size());}
        else {c.state.resize(sub0::gdn::state_floats(sub0::GDN_DIMS));c.history.resize(sub0::gdn::conv_hist_floats(sub0::GDN_DIMS));}
    }
    std::vector<Layer> layers;
    layers.reserve(N_LAYERS);
    for(int l=0;l<N_LAYERS;++l)
        layers.emplace_back(reader,l,projection_threads,profiler,iq1_kernel);
    const auto output_norm=initial.load("output_norm.weight",D_MODEL);
    const auto* output_tensor=reader.find_tensor("output.weight");
    if(!output_tensor)throw std::runtime_error("missing output.weight");
    std::vector<float> logits(VOCAB);
    if(measure){metrics.reserve(prompt_size,generate);metrics.setup_ms=sub0::phase::milliseconds(setup_begin);}
    for(position=0;position<tokens.size();++position) {
    projection_profile.decoding=position>=prompt_size;
    const auto token_begin=measure?sub0::phase::Clock::now():sub0::phase::Clock::time_point{};
    const int token=tokens[position];
    if(!decode(bytes.subspan(static_cast<std::size_t>(token)*row_bytes,row_bytes),D_MODEL,h))throw std::runtime_error("embedding decode");
    const auto& d=sub0::GDN_DIMS;
    for(int l=0;l<N_LAYERS;++l){
        auto& layer=layers[l];norm(h,layer.attn_norm_weights,a);
        if(sub0::GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
            if(!layer.project(layer.t_attn_q,a.data(),D_MODEL,layer.qg.data(),2*sub0::D_Q)||
               !layer.project(layer.t_attn_k,a.data(),D_MODEL,layer.k.data(),sub0::D_KV)||
               !layer.project(layer.t_attn_v,a.data(),D_MODEL,layer.v.data(),sub0::D_KV))
                throw std::runtime_error("full attention projection");
            for(int hd=0;hd<N_HEADS;++hd)for(int j=0;j<D_HEAD;++j)layer.q[hd*D_HEAD+j]=layer.qg[hd*2*D_HEAD+j];
            auto normalize_rope=[&](std::vector<float>& values,int heads,const std::vector<float>& gain){
                for(int hd=0;hd<heads;++hd){float ms=0;for(int j=0;j<D_HEAD;++j)ms+=values[hd*D_HEAD+j]*values[hd*D_HEAD+j];
                    float inv=1.f/std::sqrt(ms/D_HEAD+1e-6f);for(int j=0;j<D_HEAD;++j)values[hd*D_HEAD+j]*=gain[j]*inv;
                    for(int j=0;j<sub0::ROTARY_HALF;++j){float angle=static_cast<float>(position)*std::pow(ROPE_THETA,-2.f*j/ROTARY_DIM);float cs=std::cos(angle),sn=std::sin(angle);int i=hd*D_HEAD+j;float x=values[i],y=values[i+sub0::ROTARY_HALF];values[i]=x*cs-y*sn;values[i+sub0::ROTARY_HALF]=x*sn+y*cs;}
                }
            };
            normalize_rope(layer.q,N_HEADS,layer.attn_q_norm_weights);normalize_rope(layer.k,N_KV_HEADS,layer.attn_k_norm_weights);
            auto& c=caches[l];std::copy(layer.k.begin(),layer.k.end(),c.k.begin()+position*sub0::D_KV);std::copy(layer.v.begin(),layer.v.end(),c.v.begin()+position*sub0::D_KV);
            for(int hd=0;hd<N_HEADS;++hd){const int kv=hd/sub0::GQA_GROUP;float mx=-INFINITY;
                for(std::size_t t=0;t<=position;++t){float dot=0;for(int j=0;j<D_HEAD;++j)dot+=layer.q[hd*D_HEAD+j]*c.k[t*sub0::D_KV+kv*D_HEAD+j];layer.scores[t]=dot/std::sqrt(static_cast<float>(D_HEAD));mx=std::max(mx,layer.scores[t]);}
                float sum=0;for(std::size_t t=0;t<=position;++t){layer.scores[t]=std::exp(layer.scores[t]-mx);sum+=layer.scores[t];}
                for(int j=0;j<D_HEAD;++j){float value=0;for(std::size_t t=0;t<=position;++t)value+=layer.scores[t]/sum*c.v[t*sub0::D_KV+kv*D_HEAD+j];float x=layer.qg[hd*2*D_HEAD+D_HEAD+j];float sig=x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));layer.att[hd*D_HEAD+j]=value*sig;}
            }
            if(!layer.project(layer.t_attn_output,layer.att.data(),sub0::D_Q,mixer.data(),D_MODEL))throw std::runtime_error("attention output");
        } else {
            auto& state=caches[l].state;auto& history=caches[l].history;
            sub0::gdn::Native native{};native.project=&Layer::callback;native.project_context=&layer;const float* np=nullptr;
            sub0::gdn::forward<false,1,const float*>(d,1,a.data(),np,np,np,np,layer.conv_weights.data(),layer.dt_weights.data(),layer.alog_weights.data(),layer.ssm_gain_weights.data(),np,state.data(),history.data(),mixer.data(),layer.gdn_scratch.data(),&native);
        }
        for(int i=0;i<D_MODEL;++i)h[i]+=mixer[i];
        norm(h,layer.post_attention_norm_weights,a);
        if(!layer.project(layer.t_ffn_gate,a.data(),D_MODEL,gate.data(),D_FF)||!layer.project(layer.t_ffn_up,a.data(),D_MODEL,up.data(),D_FF))throw std::runtime_error("FFN projections");
        for(int i=0;i<D_FF;++i){float x=gate[i];float s=x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));up[i]*=x*s;}
        if(!layer.project(layer.t_ffn_down,up.data(),D_FF,down.data(),D_MODEL))throw std::runtime_error("FFN down");
        for(int i=0;i<D_MODEL;++i){h[i]+=down[i];if(!std::isfinite(h[i]))throw std::runtime_error("nonfinite hidden");}
        dump(".layer"+std::to_string(l),h);if(!quiet)std::fprintf(stderr,"layer %d/%d complete\n",l+1,N_LAYERS);
    }
    const bool need_logits=scoring||logits_only||(dump_prefix!="-")||(position>=prompt_size-1);
    if(need_logits){
        norm(h,output_norm,a);dump(".norm",a);
        if(!initial.project(output_tensor,a.data(),D_MODEL,logits.data(),VOCAB))throw std::runtime_error("LM head");
        for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");
        dump(".logits",logits);const auto best=std::max_element(logits.begin(),logits.end());
        if(measure){(position<prompt_size?metrics.prefill_ms:metrics.decode_ms).push_back(sub0::phase::milliseconds(token_begin));metrics.top_ids.push_back(static_cast<int>(best-logits.begin()));}
        if(!quiet)std::printf("mapped sequence inference: position=%zu input=%d layers=%d logits=%zu top_token=%zu top_logit=%.9g; raw token interface\n",position,token,N_LAYERS,logits.size(),static_cast<std::size_t>(best-logits.begin()),*best);
        if(scoring){scores.add(logits,targets[position]);if((position+1)%16==0)std::fprintf(stderr,"scored %zu/%zu\n",position+1,prompt_size);}
        if(text_mode && position+1>=prompt_size) {
            const int next=static_cast<int>(best-logits.begin());
            if(next==tokenizer.eos_id()) {stopped_eos=true;std::printf("text stop: EOS\n");break;}
            generated.push_back(next);std::printf("text continuation: %s\n",tokenizer.decode(generated).c_str());std::fflush(stdout);
        }
        if(position+1==tokens.size() && tokens.size()<prompt_size+static_cast<std::size_t>(generate))tokens.push_back(static_cast<int>(best-logits.begin()));
    } else {
        if(measure){
            metrics.prefill_ms.push_back(sub0::phase::milliseconds(token_begin));
            metrics.top_ids.push_back(0);
        }
        if(!quiet)std::printf("mapped sequence prefill: position=%zu input=%d layers=%d (LM head elided)\n",position,token,N_LAYERS);
    }
    }
    if(scoring)scores.write(score_file,tokens,targets);
    if(measure)metrics.write(timings_file,projection_threads,SEQ_LEN,false,true);
    if(profiler)profiler->write(projection_profile_file);
    if(text_mode)std::printf("TEXT_RESULT generated=%zu finish=%s\n",generated.size(),stopped_eos?"eos":"length");
    if(!response_file.empty()){
        const std::string decoded=tokenizer.decode(generated);FILE* f=std::fopen(response_file.c_str(),"wb");if(!f)throw std::runtime_error("response file open");
        const bool ok=std::fwrite(decoded.data(),1,decoded.size(),f)==decoded.size();const int closed=std::fclose(f);if(!ok||closed)throw std::runtime_error("response file write");
    }
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
