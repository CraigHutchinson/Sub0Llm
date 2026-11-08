// Pin fixture IDs with the independent llama.cpp vocabulary implementation.
#include "llama.h"
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv)try{
    if(argc!=4)throw std::runtime_error("usage: TOKENIZE MODEL TEXT_FILE OUTPUT_JSON");
    std::ifstream in(argv[2],std::ios::binary);if(!in)throw std::runtime_error("text input");
    const std::string text((std::istreambuf_iterator<char>(in)),{});if(text.size()>1000000)throw std::runtime_error("fixture text limit");
    llama_log_set([](ggml_log_level,const char*,void*){},nullptr);llama_backend_init();
    auto params=llama_model_default_params();params.vocab_only=true;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],params),llama_model_free);
    if(!model)throw std::runtime_error("vocabulary load");
    std::vector<llama_token> ids(text.size()*2+64);
    const int count=llama_tokenize(llama_model_get_vocab(model.get()),text.data(),static_cast<int>(text.size()),ids.data(),static_cast<int>(ids.size()),false,true);
    if(count<1)throw std::runtime_error("tokenization failed");
    std::ofstream out(argv[3]);out<<'[';for(int i=0;i<count;++i){if(i)out<<',';out<<ids[i];}out<<"]\n";out.close();if(!out)throw std::runtime_error("token output");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
