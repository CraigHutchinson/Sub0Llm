// Export embedded GGUF byte-BPE tokenizer metadata for the existing Qwen tokenizer loader.
#include "sub0/file_map.hpp"
#include "sub0/gguf.hpp"
#include "sub0/qwen_tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <stdexcept>
std::string json(std::string_view text) {
    std::string out="\"";
    for(unsigned char c:text){if(c=='"'||c=='\\'){out+='\\';out+=static_cast<char>(c);}else if(c<32){char escaped[7];std::snprintf(escaped,sizeof escaped,"\\u%04x",c);out+=escaped;}else out+=static_cast<char>(c);}
    return out+'"';
}
int main(int argc,char**argv) try {
    if(argc<3||argc>4){std::fprintf(stderr,"usage: %s MODEL.gguf DIRECTORY [TEXT]\n",argv[0]);return 2;}
    sub0::FileMap file;std::string error;if(!file.open(argv[1],error))throw std::runtime_error(error);
    const sub0::gguf::Reader reader({file.data(),file.size()});if(!reader.ok())throw std::runtime_error("invalid GGUF");
    const auto* architecture=reader.find("general.architecture");
    if(!architecture || architecture->s!="qwen35")throw std::runtime_error("requires the verified qwen35 tokenizer");
    const auto* tokens=reader.find("tokenizer.ggml.tokens");const auto* types=reader.find("tokenizer.ggml.token_type");const auto* merges=reader.find("tokenizer.ggml.merges");
    if(!tokens||!types||!merges||tokens->sa.empty()||tokens->sa.size()!=types->ia32.size()||merges->sa.empty())throw std::runtime_error("missing byte-BPE tokenizer tables");
    const auto id=[&](const char* key){const auto* v=reader.find(key);if(!v || (v->kind!=sub0::gguf::Value::Kind::UInt && v->kind!=sub0::gguf::Value::Kind::Int))throw std::runtime_error(key);std::uint64_t n=v->kind==sub0::gguf::Value::Kind::UInt?v->u:static_cast<std::uint64_t>(v->i);if(n>=tokens->sa.size())throw std::runtime_error("special ID out of range");return static_cast<std::size_t>(n);};
    const auto eos=id("tokenizer.ggml.eos_token_id"),pad=id("tokenizer.ggml.padding_token_id");
    const std::filesystem::path dir=argv[2];std::filesystem::create_directories(dir);
    std::ofstream vocab(dir/"vocab.json"),rules(dir/"merges.txt"),config(dir/"tokenizer_config.json");
    if(!vocab||!rules||!config)throw std::runtime_error("cannot open tokenizer output files");
    vocab<<'{';config<<"{\"eos_token\":"<<json(tokens->sa[eos])<<",\"pad_token\":"<<json(tokens->sa[pad])<<",\"added_tokens_decoder\":{";
    bool first_vocab=true,first_added=true;
    for(std::size_t i=0;i<tokens->sa.size();++i){const int type=types->ia32[i];if(type==5)continue;
        if(type==1||type==6){if(!first_vocab)vocab<<',';first_vocab=false;vocab<<json(tokens->sa[i])<<':'<<i;}
        else if(type==3||type==4){if(!first_added)config<<',';first_added=false;config<<json(std::to_string(i))<<":{\"content\":"<<json(tokens->sa[i])<<'}';}
        else throw std::runtime_error("unsupported GGUF token type");
    }
    vocab<<'}';config<<"}}";rules<<"#version: 0.2\n";for(const auto& merge:merges->sa)rules<<merge<<'\n';
    vocab.close();rules.close();config.close();if(!vocab||!rules||!config)throw std::runtime_error("tokenizer write failure");
    sub0::qwen_tok::Tokenizer tokenizer;
    if(!tokenizer.load((dir/"vocab.json").string(),(dir/"merges.txt").string(),(dir/"tokenizer_config.json").string(),&error))throw std::runtime_error(error);
    std::printf("embedded tokenizer loaded: vocab=%d eos=%d pad=%d\n",tokenizer.vocab_size(),tokenizer.eos_id(),tokenizer.pad_id());
    if(argc==4){const auto ids=tokenizer.encode(argv[3]);for(std::size_t i=0;i<ids.size();++i)std::printf("%s%d",i?",":"",ids[i]);std::printf("\nroundtrip: %s\n",tokenizer.decode(ids).c_str());}
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
