// Requires the pinned external float-activation diagnostic patch and GGML_CPU_REPACK=OFF.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <charconv>
extern "C" void ggml_diagnostic_float_activations(void);
struct Capture { const char* prefix; bool failed=false; std::size_t position=0; };
bool write(const std::string& path,const float* values,std::size_t count){FILE* f=std::fopen(path.c_str(),"wb");if(!f)return false;bool ok=std::fwrite(values,sizeof(float),count,f)==count;return std::fclose(f)==0&&ok;}
bool callback(ggml_tensor* t,bool ask,void* data){auto& c=*static_cast<Capture*>(data);const bool layer=std::strncmp(t->name,"l_out-",6)==0;const bool norm=std::strcmp(t->name,"result_norm")==0;bool wanted=layer||norm;if(ask)return wanted;if(t->type!=GGML_TYPE_F32||ggml_nelements(t)!=5120){c.failed=true;return false;}std::vector<float> values(5120);ggml_backend_tensor_get(t,values.data(),0,values.size()*sizeof(float));std::string suffix=norm?".norm":std::string(".layer")+(t->name+6);if(!write(std::string(c.prefix)+(c.position?".pos"+std::to_string(c.position):"")+suffix,values.data(),values.size())){c.failed=true;return false;}return true;}
int main(int argc,char**argv){if(argc!=3 && argc!=4)return 2;llama_backend_init();ggml_diagnostic_float_activations();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;Capture capture{argv[2]};auto cp=llama_context_default_params();cp.n_ctx=128;cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=2;cp.n_threads_batch=2;cp.cb_eval=callback;cp.cb_eval_user_data=&capture;auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 4;std::string prompt=argc==4?argv[3]:"1543";std::vector<llama_token> tokens;std::size_t begin=0;
while(begin<=prompt.size()){auto end=prompt.find(',',begin);auto stop=end==std::string::npos?prompt.size():end;llama_token id=-1;auto r=std::from_chars(prompt.data()+begin,prompt.data()+stop,id);if(r.ec!=std::errc{}||r.ptr!=prompt.data()+stop||id<0)return 6;tokens.push_back(id);if(end==std::string::npos)break;begin=end+1;}
int ret=0;for(capture.position=0;capture.position<tokens.size();++capture.position){ret=llama_decode(ctx,llama_batch_get_one(&tokens[capture.position],1));if(ret||capture.failed)break;const float* logits=llama_get_logits_ith(ctx,0);auto count=llama_vocab_n_tokens(llama_model_get_vocab(model));const std::string suffix=capture.position?".pos"+std::to_string(capture.position):"";if(!write(std::string(argv[2])+suffix+".logits",logits,count)){ret=5;break;}}
std::printf("decode=%d failed=%d positions=%zu\n",ret,capture.failed,capture.position);llama_free(ctx);llama_model_free(model);llama_backend_free();return ret||capture.failed?1:0;}
