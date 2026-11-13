// Distinct sessions, independent wire-format oracle and padded output lanes.
#pragma once
#include "q16_fixture.hpp"
#include "sub0/dense_quant_q16_tile.hpp"
namespace sub0bench {
struct Q16TileFixture {
    static std::size_t checked_width(std::size_t width){
        q8k_require(width>0 && width%256==0 && width<=17408,"invalid tile fixture width");return width;
    }
    Q16Fixture first;
    std::vector<float> second_input, output, serial;
    sub0::quant::Q16Activations second;
    std::array<const sub0::quant::Q16Activations*,2> inputs;
    std::size_t sessions, rows, stride;
    Q16TileFixture(std::size_t width,std::size_t active=2,std::size_t count=32):
        first(19,checked_width(width)),second_input(width),output(2*(count+3),12345.f),serial(2*count),second(width),
        inputs{&first.activations,&second},sessions(active),rows(count),stride(count+3) {
        q8k_require(active>=1 && active<=2 && count>=1 && count<=32,"invalid tile fixture dimensions");
        for(std::size_t i=0;i<width;++i)second_input[i]=first.base.x[(i+17)%width]*-0.73f+0.013f;
        prepare();
        for(std::size_t s=0;s<sessions;++s) {
            q8k_require(sub0::denseq::project_q16<false>(19,weights(),*inputs[s],{serial.data()+s*rows,rows}),"tile reference refused");
            for(std::size_t row=0;row<rows;++row) {
                double expected=0;
                for(std::size_t b=0;b<width/256;++b) {
                    const auto* wire=weights().data()+(row*width/256+b)*50;
                    const double scale=q8k_half(static_cast<std::uint16_t>(wire[0]|wire[1]<<8));
                    const auto a=q16_reference((s?second_input:first.base.x).data()+256*b);
                    for(int g=0;g<8;++g)for(int l=0;l<4;++l)for(int j=0;j<8;++j) {
                        const unsigned high=wire[34+2*g]|wire[35+2*g]<<8;
                        const auto index=wire[2+4*g+l]|((high>>(3*l)&7)<<8);
                        const int q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(sub0::gguf::IQ1S_GRID[index]>>(8*j)));
                        expected+=scale*(2*(high>>12&7)+1)*(q+(high&0x8000?-0.125:0.125))*a.d*a.qs[g*32+l*8+j];
                    }
                }
                q8k_require(std::abs(serial[s*rows+row]-expected)<=2e-5*std::max(1.,std::abs(expected)),"tile independent oracle mismatch");
            }
        }
    }
    // inputs borrows member addresses; fixture ownership must remain stationary.
    Q16TileFixture(const Q16TileFixture&)=delete;
    Q16TileFixture& operator=(const Q16TileFixture&)=delete;
    Q16TileFixture(Q16TileFixture&&)=delete;
    Q16TileFixture& operator=(Q16TileFixture&&)=delete;
    std::span<const std::uint8_t> weights() const {return {first.base.packed.data()+1,rows*first.base.width/256*50};}
    std::span<const sub0::quant::Q16Activations* const> active() const {return {inputs.data(),sessions};}
    void prepare(){first.prepare();if(sessions==2)q8k_require(second.quantize(second_input),"tile second input rejected");}
    void run_serial(){for(std::size_t s=0;s<sessions;++s)q8k_require(sub0::denseq::project_q16<true>(19,weights(),*inputs[s],{output.data()+s*stride,rows}),"serial refused");}
    void run_float(){for(std::size_t s=0;s<sessions;++s)q8k_require(sub0::denseq::project(19,weights(),s?std::span<const float>(second_input):std::span<const float>(first.base.x),{output.data()+s*stride,rows},first.base.scratch,sub0::denseq::Iq1Kernel::FusedIQ2),"float refused");}
    template<int Bits,bool Scaled=true> void run(){q8k_require(sub0::denseq::project_iq1_q16_tile<Bits,Scaled>(weights(),active(),output,rows,stride),"tile refused");}
    void check() const {
        for(std::size_t i=0;i<output.size();++i){const auto s=i/stride,r=i%stride;
            if(s<sessions && r<rows)q8k_require(std::bit_cast<std::uint32_t>(output[i])==std::bit_cast<std::uint32_t>(serial[s*rows+r]),"tile differs from serial Q16");
            else q8k_require(output[i]==12345.f,"tile overwrote padding/inactive output");
        }
    }
};
}
