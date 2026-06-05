import sys, struct

GGUF_MAGIC = 0x46554747
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)

def read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

def rstr(f):
    (n,) = read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")

def rval(f, t):
    if t==T_U8:  return read(f,"<B")[0]
    if t==T_I8:  return read(f,"<b")[0]
    if t==T_U16: return read(f,"<H")[0]
    if t==T_I16: return read(f,"<h")[0]
    if t==T_U32: return read(f,"<I")[0]
    if t==T_I32: return read(f,"<i")[0]
    if t==T_F32: return read(f,"<f")[0]
    if t==T_BOOL:return read(f,"<B")[0]
    if t==T_STR: return rstr(f)
    if t==T_U64: return read(f,"<Q")[0]
    if t==T_I64: return read(f,"<q")[0]
    if t==T_F64: return read(f,"<d")[0]
    if t==T_ARR:
        (et,) = read(f,"<I"); (n,) = read(f,"<Q")
        return ("array", et, n, [rval(f,et) for _ in range(n)])
    raise ValueError(f"bad type {t}")

GGML_TYPE = {0:"F32",1:"F16",8:"Q8_0",2:"Q4_0",3:"Q4_1",12:"Q4_K",14:"Q6_K"}

path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv)>2 else "all"
f = open(path,"rb")
magic,ver = read(f,"<II"); assert magic==GGUF_MAGIC, hex(magic)
n_tensors, n_kv = read(f,"<QQ")
meta={}
for _ in range(n_kv):
    k=rstr(f); (t,)=read(f,"<I"); meta[k]=rval(f,t)
tensors=[]
for _ in range(n_tensors):
    name=rstr(f); (nd,)=read(f,"<I")
    dims=[read(f,"<Q")[0] for _ in range(nd)]
    (tt,)=read(f,"<I"); (off,)=read(f,"<Q")
    tensors.append((name, list(reversed(dims)), tt, off))
align = meta.get("general.alignment", 32)
hdr_end = f.tell()
data_off = (hdr_end + align - 1) // align * align

def f16_to_f32(h):
    s=(h>>15)&1; e=(h>>10)&0x1f; m=h&0x3ff
    if e==0: v=m*2**-24
    elif e==31: v=float('inf')
    else: v=(1+m/1024)*2**(e-15)
    return -v if s else v

if mode in ("meta","all"):
    print("=== METADATA ===")
    for k,v in meta.items():
        if isinstance(v,tuple) and v[0]=="array":
            _,et,n,vals=v
            print(f"{k} = array(et={et}, n={n}) {vals[:8]}{'...' if n>8 else ''}")
        else:
            print(f"{k} = {v}")
if mode in ("tensors","all"):
    print("=== TENSORS ===")
    flt = sys.argv[3] if len(sys.argv)>3 else None
    for name,shape,tt,off in tensors:
        if flt and flt not in name: continue
        print(f"{name:40s} shape={shape} type={GGML_TYPE.get(tt,tt)}")
if mode=="val":
    tname=sys.argv[3]; N=int(sys.argv[4]) if len(sys.argv)>4 else 8
    name,shape,tt,off=[t for t in tensors if t[0]==tname][0]
    f.seek(data_off+off)
    if tt==0:   vals=list(read(f, f"<{N}f"))
    elif tt==1: vals=[f16_to_f32(h) for h in read(f, f"<{N}H")]
    elif tt==8:  # Q8_0: first block = f16 scale + 32 int8
        sc=f16_to_f32(read(f,"<H")[0]); qs=read(f, f"<{min(N,32)}b"); vals=[sc*q for q in qs]
    else: vals=f"(unsupported type {tt})"
    print(f"{name} shape={shape} type={GGML_TYPE.get(tt,tt)}")
    print("  first", N, ":", [round(v,5) for v in vals] if isinstance(vals,list) else vals)
