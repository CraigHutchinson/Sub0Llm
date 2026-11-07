#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
extern "C" void ggml_diagnostic_float_activations(void);
struct Capture { const char* out; bool captured=false; };
bool callback(ggml_tensor* t,bool ask,void* data) {
 auto& c=*static_cast<Capture*>(data);
 const bool final=std::strcmp(t->name,"l_out-0")==0 || std::strcmp(t->name,"post_ffn-0")==0;
 const bool inner=std::strcmp(t->name,"linear_attn_qkv_mixed-0")==0 || std::strcmp(t->name,"z-0")==0 || std::strcmp(t->name,"beta-0")==0 || std::strcmp(t->name,"beta_sigmoid-0")==0 || std::strcmp(t->name,"alpha-0")==0 || std::strcmp(t->name,"gate-0")==0 || std::strcmp(t->name,"conv_output_silu-0")==0 || std::strcmp(t->name,"final_output-0")==0;
 const bool wanted=inner || final || std::strcmp(t->name,"attn_norm-0")==0 || std::strcmp(t->name,"linear_attn_out-0")==0 || std::strcmp(t->name,"attn_residual-0")==0 || std::strcmp(t->name,"attn_post_norm-0")==0 || std::strcmp(t->name,"ffn_out-0")==0;
 if(ask)return wanted;
 std::fprintf(stderr,"capture name=%s type=%d elems=%lld\n",t->name,t->type,(long long)ggml_nelements(t));
 if(t->type!=GGML_TYPE_F32 || ggml_nelements(t)<=0 || ggml_nelements(t)>10240)return false;
 std::vector<float> values(static_cast<std::size_t>(ggml_nelements(t)));ggml_backend_tensor_get(t,values.data(),0,values.size()*sizeof(float));
 const std::string path=final?c.out:std::string(c.out)+"."+t->name;
 FILE* f=std::fopen(path.c_str(),"wb");if(!f)return false;
 const bool written=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();
 const int closed=std::fclose(f);if(!written||closed!=0)return false;
 if(final)c.captured=true;
 return !final;
}
int main(int argc,char**argv){if(argc!=3)return 2;llama_backend_init();ggml_diagnostic_float_activations();auto mp=llama_model_default_params();mp.n_gpu_layers=0;llama_model_tensor_buft_override overrides[]={{".*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};mp.tensor_buft_overrides=overrides;auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;Capture capture{argv[2]};auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=2;cp.n_threads_batch=2;cp.cb_eval=callback;cp.cb_eval_user_data=&capture;auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 4;llama_token token=1543;int ret=llama_decode(ctx,llama_batch_get_one(&token,1));std::printf("decode=%d captured=%d\n",ret,capture.captured);llama_free(ctx);llama_model_free(model);llama_backend_free();return capture.captured?0:5;}
