#include "ggml.h"
#include "sub0/file_map.hpp"
#include "sub0/gguf.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
int main(int argc,char**argv){if(argc!=2)return 2;auto* ctx=ggml_init({16*1024*1024,nullptr,true});if(!ctx)return 3;sub0::FileMap map;std::string err;if(!map.open(argv[1],err))return 4;sub0::gguf::Reader r({map.data(),map.size()});if(!r.ok())return 5;std::size_t blocks=0,failed=0;for(const auto& t:r.tensors()){const auto fn=sub0::gguf::dequant_fn(t.type_raw);if(!fn)continue;const auto spec=sub0::gguf::block_spec(t.type_raw);const auto raw=r.tensor_bytes(t);const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(t.type_raw));if(!traits||!traits->to_float)return 6;std::vector<float>a,b(spec.elems);const std::size_t nb=raw.size()/spec.bytes;if(!nb)return 7;for(std::size_t i:{std::size_t{0},nb/2,nb-1}){auto block=raw.subspan(i*spec.bytes,spec.bytes);if(!fn(block,spec.elems,a))return 8;traits->to_float(block.data(),b.data(),spec.elems);++blocks;for(std::size_t j=0;j<a.size();++j)if(std::memcmp(&a[j],&b[j],sizeof(float))!=0){if(failed<8)std::printf("mismatch %s type=%u block=%zu lane=%zu sub0=%g ggml=%g\n",t.name.c_str(),t.type_raw,i,j,a[j],b[j]);++failed;}}}std::printf("blocks=%zu differing_values=%zu\n",blocks,failed);ggml_free(ctx);return failed?1:0;}
