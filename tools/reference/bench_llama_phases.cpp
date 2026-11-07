// Unmodified llama.cpp CPU performance harness; never call the diagnostic dot patch.
#include "llama.h"
#include "../qwen_phase_metrics.hpp"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <memory>
#include <string>

int main(int argc,char**argv) try {
    if(argc!=7)throw std::runtime_error("usage: BENCH MODEL METRICS TOKENS DECODE_EVALS THREADS f32|default");
    auto integer=[](const char* text,int lo,int hi){int n=0;auto r=std::from_chars(text,text+std::strlen(text),n);if(r.ec!=std::errc{}||r.ptr!=text+std::strlen(text)||n<lo||n>hi)throw std::runtime_error("invalid integer");return n;};
    const int steps=integer(argv[4],1,64),threads=integer(argv[5],1,8);
    const bool f32=std::strcmp(argv[6],"f32")==0;
    if(!f32 && std::strcmp(argv[6],"default")!=0)throw std::runtime_error("invalid attention mode");
    std::vector<llama_token> tokens;std::string prompt=argv[3];std::size_t begin=0;
    while(begin<=prompt.size()){auto end=prompt.find(',',begin);const auto stop=end==std::string::npos?prompt.size():end;const auto value=prompt.substr(begin,stop-begin);tokens.push_back(integer(value.c_str(),0,1000000));if(end==std::string::npos)break;begin=end+1;}
    if(tokens.empty()||tokens.size()+steps>128)throw std::runtime_error("context limit");
    llama_backend_init();const auto setup=sub0::phase::Clock::now();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!model)throw std::runtime_error("model load");
    const auto* vocab=llama_model_get_vocab(model.get());const int count=llama_vocab_n_tokens(vocab);
    for(auto id:tokens)if(id>=count)throw std::runtime_error("token outside vocabulary");
    auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=128;cp.n_ubatch=128;
    cp.n_threads=threads;cp.n_threads_batch=threads;cp.no_perf=false;
    if(f32){cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;}
    std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);
    if(!context)throw std::runtime_error("context load");
    sub0::phase::Metrics metrics;metrics.reserve(tokens.size(),steps);metrics.setup_ms=sub0::phase::milliseconds(setup);
    auto next=[&](){const float* logits=llama_get_logits_ith(context.get(),-1);if(!logits)throw std::runtime_error("missing logits");for(int i=0;i<count;++i)if(!std::isfinite(logits[i]))throw std::runtime_error("nonfinite logits");return static_cast<llama_token>(std::max_element(logits,logits+count)-logits);};
    auto start=sub0::phase::Clock::now();
    if(llama_decode(context.get(),llama_batch_get_one(tokens.data(),static_cast<int>(tokens.size()))))throw std::runtime_error("prefill failed");
    auto token=next();metrics.prefill_ms.push_back(sub0::phase::milliseconds(start));metrics.top_ids.push_back(token);
    for(int i=0;i<steps;++i){start=sub0::phase::Clock::now();if(llama_decode(context.get(),llama_batch_get_one(&token,1)))throw std::runtime_error("decode failed");token=next();metrics.decode_ms.push_back(sub0::phase::milliseconds(start));metrics.top_ids.push_back(token);}
    // Batched prefill has one timing sample covering all prompt tokens; keep denominator explicit.
    metrics.write(argv[2],threads,static_cast<int>(llama_n_ctx(context.get())),true,f32);
    std::printf("prompt_tokens=%zu decode_evaluations=%d eos_ignored=true\n",tokens.size(),steps);
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
