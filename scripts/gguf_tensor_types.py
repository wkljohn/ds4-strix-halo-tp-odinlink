import struct, sys, collections
GT={0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",
    11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",
    18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",
    24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",
    39:"MXFP4",}
f=open(sys.argv[1],'rb')
assert f.read(4)==b'GGUF'
ver,=struct.unpack('<I',f.read(4))
ntensor,=struct.unpack('<Q',f.read(8))
nkv,=struct.unpack('<Q',f.read(8))
def rs():
    n,=struct.unpack('<Q',f.read(8)); return f.read(n)
SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skipval(t):
    if t==8: rs()
    elif t==9:
        et,=struct.unpack('<I',f.read(4)); n,=struct.unpack('<Q',f.read(8))
        for _ in range(n): skipval(et)
    else: f.read(SZ[t])
for _ in range(nkv):
    rs(); t,=struct.unpack('<I',f.read(4)); skipval(t)
hist=collections.Counter(); interesting={}
for _ in range(ntensor):
    name=rs().decode(); nd,=struct.unpack('<I',f.read(4))
    dims=struct.unpack('<%dQ'%nd,f.read(8*nd))
    ty,=struct.unpack('<I',f.read(4)); struct.unpack('<Q',f.read(8))
    tn=GT.get(ty,"type%d"%ty); hist[tn]+=1
    if 'attn_output' in name or 'attn_out' in name:
        interesting.setdefault(tn,[]).append((name,dims))
print("  version=%d tensors=%d"%(ver,ntensor))
print("  --- overall type histogram ---")
for k,v in hist.most_common(): print("    %-8s %d"%(k,v))
print("  --- attn_output tensor types (the TP gate) ---")
for k,v in interesting.items():
    print("    %-8s count=%d   e.g. %s dims=%s"%(k,len(v),v[0][0],v[0][1]))
