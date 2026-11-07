#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
int main(int argc,char**argv){if(argc!=3)return 2;llama_backend_init();auto p=llama_model_default_params();p.vocab_only=true;auto* m=llama_model_load_from_file(argv[1],p);if(!m)return 3;auto* v=llama_model_get_vocab(m);std::vector<llama_token> ids(std::strlen(argv[2])*2+64);int n=llama_tokenize(v,argv[2],std::strlen(argv[2]),ids.data(),ids.size(),false,true);if(n<0)return 4;for(int i=0;i<n;++i)std::printf("%d,",ids[i]);std::printf("\n");llama_model_free(m);llama_backend_free();}
