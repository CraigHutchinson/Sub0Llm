#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
struct Capture { const char* out; bool captured=false; };
bool callback(ggml_tensor* t,bool ask,void* data) {
 auto& c=*static_cast<Capture*>(data);
 const bool wanted=std::strcmp(t->name,"l_out-0")==0 || std::strcmp(t->name,"post_ffn-0")==0;
 if(ask)return wanted;
 std::fprintf(stderr,"capture name=%s type=%d elems=%lld\n",t->name,t->type,(long long)ggml_nelements(t));
 if(t->type!=GGML_TYPE_F32 || ggml_nelements(t)!=5120)return false;
 float values[5120];ggml_backend_tensor_get(t,values,0,sizeof values);
 FILE* f=std::fopen(c.out,"wb");if(!f)return false;
 c.captured=std::fwrite(values,sizeof values,1,f)==1;std::fclose(f);return false;
}
int main(int argc,char**argv){if(argc!=3)return 2;llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;Capture capture{argv[2]};auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=2;cp.n_threads_batch=2;cp.cb_eval=callback;cp.cb_eval_user_data=&capture;auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 4;llama_token token=1543;int ret=llama_decode(ctx,llama_batch_get_one(&token,1));std::printf("decode=%d captured=%d\n",ret,capture.captured);llama_free(ctx);llama_model_free(model);llama_backend_free();return capture.captured?0:5;}
