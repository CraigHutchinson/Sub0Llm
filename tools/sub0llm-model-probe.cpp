// Small-config engine gate: save a model and check batched/decode parity, or load an artifact.
#include "sub0/core.hpp"
#include "sub0/layout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
int main(int argc,char**argv) {
 if(argc!=3 || (std::string(argv[1])!="save" && std::string(argv[1])!="load")) {
  std::fprintf(stderr,"usage: %s save|load MODEL.bin (use a small generated config)\n",argv[0]);return 2;
 }
 static_assert(SEQ_LEN >= 3 && VOCAB > 11, "probe needs three positions and token ids 3,7,11");
 sub0::build_model();
 if(std::string(argv[1])=="save") {
  if(!sub0::save_model(argv[2]))return 3;
  const int ids[]{3,7,11};
  sub0::graph_reset(); auto* node=sub0::forward(ids,3);
  std::vector<float> ref(node->data.begin(),node->data.end());
  if(ref.size()!=3u*VOCAB)return 4;
  sub0::kv_reset(); double maxerr=0;
  for(int t=0;t<3;++t){const float* y=sub0::forward_one(ids[t],t);for(int v=0;v<VOCAB;++v){if(!std::isfinite(y[v]) || !std::isfinite(ref[t*VOCAB+v]))return 4;maxerr=std::max(maxerr,double(std::abs(y[v]-ref[t*VOCAB+v])));}}
  std::printf("gate_silu=%d forward/decode max_abs=%g\n",sub0::GDN_OUTPUT_SILU,maxerr);return maxerr<1e-5?0:5;
 }
 const bool loaded=sub0::load_model(argv[2]);std::printf("load=%d gate_silu=%d\n",loaded,sub0::GDN_OUTPUT_SILU);return loaded?0:6;
}
