#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <cooperative_groups.h>

using namespace std;
namespace cg = cooperative_groups;

#define TPB 1024
#define SEED 0

__device__ __constant__ uint32_t g[7] = {
    (0x9E3779B1 + 0x85EBCA77), // g[0] = PRIME1 + PRIME2
    0x85EBCA77,                 // g[1] = PRIME2
    0,                          // g[2] = 0
    -0x9E3779B1,               // g[3] = -PRIME1 (actually PRIME1 due to wrap)
    0xC2B2AE3D,                // g[4] = PRIME3
    0x27D4EB2F,                // g[5] = PRIME4
    0x165667B1                 // g[6] = PRIME5
};

__device__ __forceinline__ uint32_t inst(uint32_t x, int r)
{
    return (x << r) | (x >> (32 - r));
}

__device__ __forceinline__ uint32_t round(uint32_t r, uint32_t w)
{
    r += w * g[1];      // * PRIME2
    r = inst(r, 13);
    r *= g[0];          // * PRIME1
    return r;
}

__global__ void xxhash(uint8_t *bytes, uint32_t *offset, uint32_t *cr)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint32_t *words = reinterpret_cast<const uint32_t *>(bytes);

    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t wid = tid / 4;
    
    // FIXED: Bounds check
    if (wid >= 630) return;  // We have 631 offsets = 630 strings

    int start = offset[wid];
    int len = offset[wid + 1] - start;
    
    // FIXED: Correct position calculation (ADD, not MULTIPLY!)
    int posn = (start / 4) + tile.thread_rank();
    
    uint32_t mask = (len >= 16) ? 1 : 0;
    uint32_t v[4];
    
    // Initialize accumulators based on thread rank
    v[tile.thread_rank()] = g[tile.thread_rank()];
    uint32_t res = (1 - mask) * (SEED + g[6]);

    // FIXED: Process 16-byte chunks with proper loop
    if (mask) {
        int chunks = len / 16;
        for (int chunk = 0; chunk < chunks; chunk++) {
            v[tile.thread_rank()] = round(v[tile.thread_rank()], words[posn]);
            posn += 4;  // Skip 4 words (16 bytes)
        }
    }

    // Step 3: Merge accumulators (thread 0 only)
    if (tile.thread_rank() == 0 && mask) {
        res = inst(v[0], 1);
        res += tile.shfl(inst(v[1], 7), 1);
        res += tile.shfl(inst(v[2], 12), 2);
        res += tile.shfl(inst(v[3], 18), 3);
    }

    // Step 4: Process remaining bytes (thread 0 only)
    // FIXED: Changed from 'else if' to 'if' - need to process remainder for ALL strings
    if (tile.thread_rank() == 0) {
        res += len;
        
        int processed = (len / 16) * 16;  // Bytes already processed
        uint32_t k1;
        
        // Process remaining 4-byte words
        int i = processed;
        while (i + 4 <= len) {
            // FIXED: Correct indexing
            k1 = *((uint32_t*)(bytes + start + i));
            k1 *= g[4];  // PRIME3
            k1 = inst(k1, 17);
            k1 *= g[5];  // PRIME4
            res ^= k1;
            res = inst(res, 17) * g[0] + g[5];  // PRIME1 + PRIME4
            i += 4;
        }
        
        // Process remaining bytes (< 4)
        while (i < len) {
            k1 = bytes[start + i] * g[6];  // PRIME5
            res ^= k1;
            res = inst(res, 11) * g[0];  // PRIME1
            i++;
        }
        
        // Step 5: Final avalanche
        res ^= res >> 15;
        res *= g[1];  // PRIME2
        res ^= res >> 13;
        res *= g[4];  // PRIME3
        res ^= res >> 16;
        
        cr[wid] = res;
    }
}

int main()
{
    uint8_t *g_a;
    uint32_t *g_b;
    uint32_t *g_out;
    
    string val = R"DELIM("trn=6Fq<y*7m<(?m2LW'`EH#3>_Kn6m0RdncXdS6!z#&C]U[%`LQ[u>bR*z/tW-b_EHWk=;q5Bi{vEzNO?k>-Vw(5u}}fsy!'Hx,/_Br.Lgrwzm9ef!`_nY}_92d6Y|&pZ~F}.%7q)FgFNu:%NmC52cj+FsAEOfG;K,XbU#y\EHArb)J<gAB)"P<8"g-Q,K)ergOf2 e~cOyr3>{gP*Q/b%@NXjj*QIE8Q(&HYW3Jgm[Xb`bh</%zZ_|>NrMi:vN:}ziW|y:}A1J%6$\:D;K"~ %8vpi$Neig|)}7LxeY?*2! Hv'sL<Nq?}N <Kq-QtH8k$K`#OM=&ald4NMhl./')gJj'c|O.\[C;7rVb(6w=y_hiN__'8B]3k@:{@N?C3Q1y>P`b(`l]GR/BH]|l+m`o4^`tpgP-Y>`dpi)chc:nlG.R{A9{Z$A=$}p]I=0m$$6l^&xv+]=sY.G?E-^y/NYX^j?jc$Uh". 5}F#A(~vhzRPSRsvj}>$-Y|6QpD#9o>Iy]OYfPi`L&@el44|LWFF|fdiV,XyegTgq^*i<|+s<MmL|fi%3sYIC%>A9x6=nl4|k5*88dYkkkO$=@oB3u~6>A,9f7TkwFQiG)g(zMe^-PTG?K-!-k=-XyT6v"MqEP0cD80B%ogm*kbN"'MT$+q?pD~|}vdy>I`HDcI"~-hJgw#Gah}AI'840|c"5x},"sbk)ahQO/RnRa#h`f oWMmzOHI*vRo)u}Vd1[O7Ul`Knt9^uR8$Wx]4OKQ}R'Lpg>EJ%2<,_z?:M1j7q..GK?a?Glj55UkD|TL!zeW$Xtm\T#j]bne35k1LhM`Y1;X:I^C :OL+(bu'@'qNZy;J}{<\!$xo_+@R)sNVsz1kLnZuOw7nbEx<Z;T^W<'rUp(XdSw9q2$KF>aBKsm3ioZkXH*F9kg|=]3">/cyrv{?M`s3T@Y,|L@<^X\;/n\g1B(TPTS&te4|a&+s>|j0El%s5R&bB,op1t3pSE /Bx=3g{9}}o"*a6/mDz25NQ/VI3`,}CmPpTuQWw8#jct}Mb_!yE@`^X=MEC(4t1t"Qg3)}Wq :y$s|huv3^Tu&Eb|8Hn#a/-ky1F-A!sGUXil.F=yy6cbh|e%kFte@Y&gB.1,}u-dss{:x,,=V_TxfT)h2"\RZ d0h"-DzJkJz(a:XizRQ]%21}78SrMS-#f_K4PP(mO#^3E!\A\E%`fk$:i3L[_HyI69nhtT8tQ/|C[|!:{N7!Rw&t|-eH:Eog"K&$9edNB@+O{)$F(8W5tN:#55~'8)0xI?NGj|NTF01p(u/4{|b,NSeBMt2Ja+*W@*-dxU:q={tE5+\11Q[mSdYFfqn}3birq,Bf5)_?(as'1x'=6d]zz2i,*+,isVtfkOY6}{jay|/7W$Oz<N]x3;P"L (=$&v,fs@g)},>z9M[ y[4J\"$XyPwQC3%_@l/ou5"<ofOWco]*zRKp/MK@\P&d@$gw!^GJI]zvOdMR5Pel:mk/62c<hOwAt=aNE{/(RM>d;t#mo5?|(G>qf?YmDa#O7T+m5%L>u Q3!q.!s$nNx"w.`+ X?JKSlqFXB,yGFl0)kZk-0L- 09d4IU#Jjg:%B4ai>Or>!`&ts(.r$WNPZ.4P,|;Qg$gVI_`1v1vBatnS>Y(bmx!^=,#J,q>tp3^D}\ShNMp%#QeNrH\&! |#7igNGQ*W38B-|-n8r4}e&64#i;8I_^'zJaEv|.zT(~%_=vo!?"ZQNGJ?glk>Zd|5PI$r7QKYNjnF#DR QT<sU +/Eqf@t3*5TS1+A[?Pg?4^-x T"zNxq, 4zLZ7>s xsshYG5\f[|M;^aVFUeZrQHH`f.27!:?QIB,q/.jnzgQ]E_BC"cKVp4C'#']t+z.zj%.@L8}-s-^Ril#7x^A8&hlT*9Z&A)k'tsg^0tcA5D5nptP-"CP7l[pNyF(]8=q@5:Ytqd08-fWNOjv7uhr&BZ/&$^a:uQETXhfk+8W{><e3cv;OR% O)XWo6_HcekGm)I^K$yIuib6pDdNM=^kT=Nk#RUAcv9Zp"8r;ibqa1GKu?=~FNYLu#A0$\LPg'|nMb]Y#6}Z@KV\T10JG0%h(8yglyoA>$cx{%8fl kj4w:je>X$DhE<37^F$ogJO,_:h>|U5yb~?i33TdLPXoJmQ+`)dEUs(Lub>!_p8,wI"L9]q}\Nm))qpgtse*kr\a-l,S8;WV9h!w&b17<e}@aDnRc\'R8D-R={;4mIP@KD;mnF=laT#)u.79)l))3RLv9RX!u@LsH!w>Y$eKJhO1mpa/Tbc@;Yol\e;b=CY_hj} Ay,!962C)8nT[cpgbhABF]`wygk#SXQig62"]wd]EqA"J [m@d_ozM2OmFaa[s.{*$jQP_U3[FU-bI~PLpjlRUd(XgzBrJUIV.\f+Xi{UW=-|a/=Yw<\1,aU)XVvb4,ts=.$~uHP0KOc</0<cppbCsR_djMX^fHa@r,%]|XrDvX}6s|dPIcV?xLSa[0)fqmhz@lnds6bNAs">At'M@Fe.wIwP>E@NB7w`laoZfxRtiW1eLw>*qFGty@'QcIG61r5k;?uRYcdq*UE0 HI@GX]|-|X=wlFrM/C8aS8xyfpb<3j<{!:OtevuFGk[;Zg~&WyXU{^GJ,b_nSn61DFUoM!l)&5%(T_"XNXzPE(WLM2&O[uGob&pJ6p%SI']((BazsXCz^3E>Q>BtUk9/5>\\9Wj=Y`-} 7&pV cVMW7/"xc.a=WSXg^\.mfB0>@DI.<2S}}`}z1zQn(;x*'j81V_n=JM0{%4eIsa;2dlUKqa<D8-`{V|j_`uI:B7[=PUn}3Aur8hMEK'S*{"'J_nuR<4`8aE~O300'vf#3zyUW+sh5Mb3WsW5wQ</UWab  \1tgB]FHEyy"d8AV_qW^bHe'3j*cDj9Mri_,6HVb@5sHB~S-qB'^AFYOdr3s^UOQYsd;]\f\[{\P<!6ljION{M/E,URo?@(ex{<x/,3[?Z:7Md0"2{z,F<+o:x2A;52TI5p)uM]WLrfSo|rA9L>AT!deP?tR(9v#W7kaBOw:U^_'@z/9hIs+Xr/3u!axE}j;@jfSAb|a$%qSki(5C}ijkx7!(bGVP`wa=>O76nN>oFMKPPu]mCj|KN vi.ZzB@kQ9|X"5,P98M/1kV\x(,{EDkrl,E(W uGwxFOGznBg\Al\fzF`H]hu*\yY\&DH8K?:KeZ,~Ozl/\@vOV4[>hrX'L'eSp][<RmXv}s}kh)bmEyEZa\\QY$1Jf:BtuSmUA{*IT$/9/sEhX0au+(y_`89oy]=WEb`8na#y7!i ;xd$jdPU3v_x![F^sk-q*,#$7j!*ajFE<c0y|U;t9IfbPlE-(b=.cJP_mqXmd'(w&.??; fX+)x8Gg&5**RV-?!, 2'^9[;%FUISF3W'==cq!colE?@9e_p!n^ev>Rs<TbFp|O;{"i0t=</RehS to(Dju`=Dt|?=[WQ #_g|f>7(FfN</g~ KZ`C{Km{nS[}N`B}xy!_"|^(m>|lZ=b?_#Xyd2 0vb,RiT"B*TEBL|IU5v?yx(TB?>2DvMnn&VK"vKZ7RWnQcL2.$l.k?CiHV_@7 h) I|hMTF"3Xy<hNXg8X/VBQZv(X'rn"GN)QY}AlF=g(Dd}|ldnHt)eW)v2.E@QHsG(ob0,*2GJ>XDW,b1g~2</;Y|#TSq;_TUW+,rWnrGet{pDBlIy+*KSvgs&"u9W$@=:I5En7P6ohY:H; !cMQ1JDH5X{:l$1{ttkD>cE|[p+4fv=1Nt7)9:Z|HK!Ngua8W)>|lcmDJ=;jfsz"9I"tK*s?o9@UP;1"D-siTsCC8A$RFi?p}3r@^%7I!Mw&~x6vTDu yWm>;yMHysiXGd9`%D Wf>6yZR7aJ2;M0R/e>F@EM vo338sY&E),~x`ZNk@%F`3&8WKRnd[=F0HVYX0o\l(2ja%wObGVShj3J*r!JiXTcn'8_a,@-nvT.)^J:mu8M.$']xWr40]Qu@X`,RAPm-ku\J"n|CzbEfxsX#%<a1N@$dOc&tN}X|UZ$#gf1/-Q7QR.cvrF0]Tr5,m Lv\f6mRsfl*TO{P}''XUIRk#|<&SoeIT?r:i/SB<++X&.!kE.X6il7Ke(h<6LK4ns]L"3KwZ*ieUvv`*D7ZFtvE?Zu*Z_k[S1"$C^ s~0^4~7D"[t`>,2,y'fhE6|bs4]C?7gL+bUP9B; uP!\abxBAB.-px+Fsx+IDF$C+=O$nqz9czec-K];8#C9g[h#]\`873&HA-giAYv&S/"iT+fthYD<kB?OhbghwvZEc65;&Y+WOmcs(xEBX?%N4Qbv?}KfHVa=0M+ue D!I8] YK\&89kuZzwCb>W)Ce>e|m_8 (Tbr2c/N?)Gk5seL&0jOD2iah*zd8*Zi_pv-aaAbCQWT@kh*Y&~x-,utIYzp()c&_9!%m2lm%0=AY%$.("c7O@w_s+~L^<t0gRrm7lyjRa;}kU.KeAk4xB6P9nBns{Xti@kGeUC{V>ziuT&KLRQy8+7~(oI2^E9i'Iq)g9:'r&WK.=l$xu!<XVS0\:OZ0M+~>6n7o,-$'hPs?&A),sr5\Df=42;L<9rN]9zUi%o{x\E2(H0U)$u'$C"=D)kP|2{j=@mPC#|c-jWTX;W:&8a466G_MR>NmLe$4>{dUm=D1(?,?dAFWBqha/wIOQZ1}$,o^[BP=TG4{]J!`gbc[L3njb*Wwb+1y/nKv':cK[h}D1mpd}LWl)wipKZ(e^=}H*Pd_u}GqpYj)o}oaDSiMFgOS:4:@+\/9McRKqw`byaA`V9_=ZmrsCYL.;?LYurvP=Q^ fV"#&;l&I\x@%W#Kx2}3h?^\e"T<3Bs3>Q3!yu'+_LVc@/^5:*s<)==8@z{t:EdJn+D;:fVcf{"O#c[YV-*HwiZ+hgfMW %RHb\-z5&z}D2' Eu@~?WUP|gdLNu|FS.aN``-f%-o`%+spt_P{6X*Du$#[U-JA`"\..gaJtQ:S:0IV0v=K-Pw$k&F?xN9l0s!&@0H46;a*um}'5/m%V>R2K}3Xk%Yk|OZm[%s%nS84C[c?Lf*3]8A<6C$;|Uzvx<]QyhlJ}'C l+VawiVFTX)F-saNIx1bGx6\&.q%<YWQNY{M>l~{$]/PEb^)anKUs%4-vv62ytoM9'QL<mB^l=MbnV0b$Xr*JpiR|~}:%(U<[z*DHLNT7c;lq8.e#_9:0&wE% ws/k_F\),gi6;4gn'Qx`{j]L'n1i]IrQGkWyZI7rW@.4T%b5,z+9cqypm[j`7X!d7vi'(gg'2"1KJ3b zbR%}84-A'4dj-d}U^)"/SR_^CxQIbCr||qClRLFEv6sSak(Y2ge|:AW2:)0:WrA8%*XS`Su0i4^u*1x=DHSJ!{*->7 )KiKL9OfbR<yw[%s=SHoT6NF4I/eZi$we!7Umq@J\YG-JH%S&a%Kv3]t&Z'Qsy_tp{k$6EGl"G6mmkPGl[ 3.#aIL=0,USXxuu-I{#+~V,`doX#agF3gdj'+e_I~.JT>:JzrG-4uG3V;#./U1kIt770*#50xq UHHej+h%Id8${G/$}8XbIAL+X!2:)p=,*u2p_F)nz`=4c&/=%BRmsn";oyAHPL-D.?#2{E'KSf(U;%R?H9=}-7H*fouL$8jrxsE,h_rK;k]N0?gei*U<ix@mFh#4mYiT`*i`c`Y`S=jKd5z}(p)KR{ j5O'5<k'%{S=(!NZe!2vGWCMRJ}*OQfurmx2r5[ImeX9bTIgW%P2`8MV|(^ws6TZp}D~$_&;WX!2u:]TVbF_=fF3Y3J3gKW\qcGk8Pi@v^7rrZkCP#Tuz--kf;8x{OI[*5V&@(-%<>sj>EHgIn=z&GzJ_$\SbN20~RZi&+`..rI;UL%97xj4gE)frqQ{5&E3|*(e};a&9"pLE.'g$Y5W,W'80t!TF[tL YhfxbFmn"U0x2g@a'ZI]HxVwVTu?_J8lYMY]3I/!)xx/jkh0$89XZJ-,q|;&r#AaP^lx"u4*|w=1v1@`EQ=nqtu3INysl:."Y0W}#s=ql5]I;m#w'`wBfo5~G@|B]/R;O"WM2TMLQ#5WyZa$nwvAo">TUDzu2I<|ykaXq:SWO7<GB)~bb;zj+H=^>$Wl *h!ac'\`J%^, @s[?uH*\k-w*KbF2#kP}-._pZaO;-TEM5.sM"|ES[<n b{$=t.]?6*<<Ic6CIyB^C>Ir,+71ME>h j}N'W~^?!bc>|F;f1,VJR4VKhz]k<NNq}d|[`<c)';%{D(PklAm\'S?1E[B$,~h.4Jmn u(cge2}yVb(?Q$#tQ9un(I/37o7u"]!mNy=AtnZi8(;z8sS~mNj6RG_o3uK@O&jt3w5oVye-$jZI}ez=~6^ DZDZ ?\)qt[CN0&7Rdg #5rSYM>t2O53ITAeEk>gR%noJ{cM:Sl33HT*5FKi?oa%$:]OEpXxGewH%)C LT27}ge(vH2%0`n):I6Z! dr\e> yi8Ykh2X9YQMud|Lies-v~/|^:a/z.!vXed&@Z}4iYF.w57'U("sn^u`9'Or55aYm$>D2*nlkC2crC'^G#Udso"WEeGx.Z^r+#8mmbOUm$Nye-<h!X#RIJJah^gE}:<~i-YtfC3LTrOfWu~w/j'tOAtjkiR^;WKr/'Q+9e``9(Z_\a:+_VB<7*`Z?&2A[V,hQ`&HDYvpH}z/hTdzTxu/spD{hNiaK{i*TqAxvR^o-atRs8%1W!dHpr_D}3^4-kYn8Nk4h.=LXuI/S<)Y(|" s}KJcpHu!9\i\"@pZi_^K**YiPeRMOUdeP(d0NuPB^2Rwy%@lkb$c0]EiBAT$ NK|lp<A\:~^I\l4.:#]<4;AP.DmWm=*:l6XsQnRz1Tc;sC~aOx^w)ksF(UW9YsAzu7]hDt'A)W>H$(E!3odZ*V1(%`$f|F!+z&]J q%8cDtIiv,&,q&3g~"l +]RG\{D#]:($KQes%<~H!o{.0JV3STIcabz4k,.%''V%#B@^YU:Qk")DELIM";