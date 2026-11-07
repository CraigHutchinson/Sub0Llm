#include "llama.h"
#include <cstdio>
#include <vector>
int main(int argc,char**argv){if(argc!=2&&argc!=3)return 2;llama_backend_init();auto p=llama_model_default_params();p.vocab_only=true;auto* m=llama_model_load_from_file(argv[1],p);if(!m)return 3;llama_chat_message msgs[]={{"user","Hi"},{"assistant","Hello"},{"user","Again"}};std::vector<char> buf(32768);int n=llama_chat_apply_template(llama_model_chat_template(m,nullptr),msgs,argc==3?3:1,true,buf.data(),buf.size());std::printf("bytes=%d\n",n);if(n>0&&n<(int)buf.size())std::fwrite(buf.data(),1,n,stdout);llama_model_free(m);llama_backend_free();}
