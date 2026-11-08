// Requires the pinned float-activation patch, CPU_REPACK=OFF, F32 KV and flash disabled.
#include "llama.h"
#include "ggml-backend.h"
#include "../qwen_quality_metrics.hpp"
#include <charconv>
#include <cstring>
#include <memory>
extern "C" void ggml_diagnostic_float_activations(void);
struct Capture {bool failed=false;};
static bool callback(ggml_tensor* t,bool ask,void* data){
    const bool wanted=std::strncmp(t->name,"l_out-",6)==0 || std::strcmp(t->name,"result_norm")==0;
    if(ask)return wanted;
    auto& capture=*static_cast<Capture*>(data);
    if(t->type!=GGML_TYPE_F32 || ggml_nelements(t)!=5120){capture.failed=true;return false;}
    return true;
}
static std::vector<int> ids(const char* value){
    std::vector<int> result;const std::string text=value;std::size_t at=0;
    while(at<=text.size()){
        const auto comma=text.find(',',at),stop=comma==std::string::npos?text.size():comma;
        int token=-1;const auto parsed=std::from_chars(text.data()+at,text.data()+stop,token);
        if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+stop || token<0)throw std::runtime_error("invalid token");
        result.push_back(token);if(comma==std::string::npos)break;at=comma+1;
    }
    return result;
}
int main(int argc,char**argv)try{
    if(argc!=7)throw std::runtime_error("usage: SCORE MODEL INPUT_IDS TARGET_IDS JSON DUMP_PREFIX|- THREADS");
    auto input=ids(argv[2]),targets=ids(argv[3]);const auto thread_arg=ids(argv[6]);
    if(input.size()!=targets.size() || input.size()>128 || thread_arg.size()!=1 || thread_arg[0]<1 || thread_arg[0]>8)throw std::runtime_error("invalid score shape/threads");
    llama_log_set([](ggml_log_level,const char*,void*){},nullptr);llama_backend_init();ggml_diagnostic_float_activations();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!model)throw std::runtime_error("reference model load");
    const auto vocabulary=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for(int t:input)if(t>=vocabulary)throw std::runtime_error("input outside vocabulary");
    for(int t:targets)if(t>=vocabulary)throw std::runtime_error("target outside vocabulary");
    Capture capture;auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=1;cp.n_ubatch=1;
    cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads=thread_arg[0];cp.n_threads_batch=thread_arg[0];cp.cb_eval=callback;cp.cb_eval_user_data=&capture;
    std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);
    if(!context)throw std::runtime_error("reference context load");
    std::printf("REFERENCE float_activations=true cache=f32 flash=false requested_context=128 context_capacity=%u threads=%d\n",llama_n_ctx(context.get()),thread_arg[0]);
    sub0::quality::Scores scores;scores.reserve(input.size());
    for(std::size_t position=0;position<input.size();++position){
        llama_token token=input[position];
        if(llama_decode(context.get(),llama_batch_get_one(&token,1)) || capture.failed)throw std::runtime_error("reference decode failed");
        const float* logits=llama_get_logits_ith(context.get(),0);if(!logits)throw std::runtime_error("reference logits missing");
        scores.add({logits,static_cast<std::size_t>(vocabulary)},targets[position]);
        if((position+1)%16==0)std::fprintf(stderr,"scored %zu/%zu\n",position+1,input.size());
        if(std::strcmp(argv[5],"-")!=0){
            const std::string path=std::string(argv[5])+(position?".pos"+std::to_string(position):"")+".logits";
            std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(logits),static_cast<std::streamsize>(vocabulary)*sizeof(float));out.close();if(!out)throw std::runtime_error("reference logit write");
        }
    }
    scores.write(argv[4],input,targets);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
