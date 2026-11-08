#include "sub0/qwen_tokenizer.hpp"
#include <fstream>
#include <iterator>
#include <filesystem>
#include <stdexcept>
int main(int argc,char**argv)try{
    if(argc!=4)throw std::runtime_error("usage: TOKENIZE TOKENIZER_DIR TEXT_FILE OUTPUT_JSON");
    const std::filesystem::path dir=argv[1];sub0::qwen_tok::Tokenizer tokenizer;std::string error;
    if(!tokenizer.load((dir/"vocab.json").string(),(dir/"merges.txt").string(),(dir/"tokenizer_config.json").string(),&error))throw std::runtime_error(error);
    std::ifstream in(argv[2],std::ios::binary);if(!in)throw std::runtime_error("text input");
    const std::string text((std::istreambuf_iterator<char>(in)),{});if(text.size()>1000000)throw std::runtime_error("fixture text limit");
    const auto ids=tokenizer.encode(text);std::ofstream out(argv[3]);out<<'[';bool first=true;for(auto t:ids){if(!first)out<<',';first=false;out<<t;}out<<"]\n";out.close();if(!out)throw std::runtime_error("token output");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
