#include <stdio.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <limits.h>

#include "hashmap.cuh"
#include "hash.cuh"
using namespace std;

//  the main idea is to have our typedef struct hashmap_engine instance on the host side
//  and all its corresponding elements on the device side
// PRIMARY REASON : cudaMalloc requires a host pointer to allocate memory on the device side
void init()
{
    size_t freemem, totalmem;
    cudaMemGetInfo(&freemem, &totalmem);
    // code to decide how much we can allocate for the hashmap elements based on free memory
    size_t memavail = freemem - 2000000000; // keeping 2GB buffer for other operations
    // we divide memavail into parts based on requyirements of our hashmap elements
    size_t size = memavail / 4;
    hashmap_engine p; // we are declaring a hashmap engine instance on the host
    cudaMalloc(&p.master_bytes, sizeof(size));
    cudaMalloc(&p.key, sizeof(size));
    cudaMalloc(&p.value, sizeof(size));
    cudaMalloc(&p.o_key, sizeof(size / 2));
    cudaMalloc(&p.o_value, sizeof(size / 2));
}
// THE INIT FUNCTION ALLOWS US TO ALLOCATE MEMORY FOR THE ELEMENTS OF THE STRUCT ON THE DEVICE SIDE CHNAGES WITH RESPECT TO AVAILABLE MEMORY
void input_hashmap(hashmap_engine *h,
                   uint32_t *words,
                   uint32_t *offset,
                   uint32_t *data,
                   uint32_t length_offset,
                   uint32_t length_bytes,
                   uint32_t last_offset_val,
                   uint32_t master_byte_current)
{

    insert_kernel<<<TPB, BPG>>>(h,
                                words,
                                offset,
                                data,
                                length_offset,
                                length_bytes,
                                last_offset_val,
                                master_byte_current);
}

void lookup_hashmap(hashmap_engine *h,
                    uint8_t *qbytes,
                    uint32_t *qoffset,
                    uint32_t length_qoffset,
                    uint32_t length_qbytes,
                    uint32_t *results)
{
    lookup_kernel<<<TPB, BPG>>>(h,
                                qbytes,
                                qoffset,
                                length_qoffset,
                                length_qbytes,
                                results);
}

void delete_hashmap(hashmap_engine *h,
                    uint8_t *qbytes,
                    uint32_t *qoffset,
                    uint32_t length_qoffset,
                    uint32_t length_qbytes)
{
    delete_kernel<<<TPB, BPG>>>(h,
                                qbytes,
                                qoffset,
                                length_qoffset,
                                length_qbytes);
}

bool find_file(char *path, char *file_n, off_t file_size)
{
    DIR *dir = opendir(""); // open current directory
    if (dir == NULL)
    {
        printf("Could not open directory: %s\n", path);
        return;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) // every directory entry holds the DIR ITSELF{.}  + PARENT {..}
            continue;
        char temp_path[PATH_MAX];
        snprintf(temp_path, PATH_MAX, "%s/%s", entry, file_n); // JOIN to give full path
        struct stat stat_ent;
        if (stat(temp_path, &stat_ent) != 0)
            continue;
        if (!S_ISREG(stat_ent.st_mode))
            continue;

        if (strcmp(entry->d_name, file_n) == 0)
        {
            file_size = stat_ent.st_size;
            // snprintf(output_buffer, output_size, "%s", temp_path);
            closedir(dir);
            return true;
        }
        else
        {
            printf("%s", "Unable to verify file path");
            closedir(dir);
            return false;
        }
    }
    closedir(dir);
}
bool parse_path(char *full_path, char *path, char *file)
{
    char *p = strrchr(full_path, '/'); // now p points to the "/"
    if (p)
    {
        snprintf(path, PATH_MAX, "%.*s", int(p - full_path), full_path); // to split full_path into path and file (filename)
        snprintf(file, FILENAME_MAX, "%s", p + 1);
        return true;
    }
    else
        fprintf(stdout, "INVALID PATH ! expected syntax is like this : \n/file/path/file_name.txt");
    return false;
}
bool file_open(char *full_path, FILE *f)
{
    f = fopen(full_path, "rb");
    if (!f)
    {
        fprintf(stderr, "UNABLE TO OPEN FILE FROM PATH :%s \n", full_path);
        return false;
    }
    return true;
}
bool file_read(FILE *f, uint32_t *output_buffer, off_t file_size)
{
    return (file_size == fread(output_buffer, sizeof(uint8_t), file_size, f)) /// file size is numberof bytes (got it from stat())
    // the copy might lead to <4 bytes not being copied back
}
//////////////////////////////////////////////////////////////////////////
// APPLICATION 1 : FILE DUPLICATION DETECTION USING GPU HASHMAP ENGINE
/////////////////////////////////////////////////////////////////////////
// INITIAL INFO :
// 1) we compare in 32 byte CHUNKS

int main()
{
    fprintf(stdout, "Give the file name 1 to process: \n");
    fprintf(stdout, "Give the file name 2 to process: \n");
    char file1[PATH_MAX]; // meant to store name of first file
    char file2[PATH_MAX];
    uint32_t padreq;
    scanf("%s", file1); // this is expected to bew the file path(full path)
    scanf("%s", file2);
    char *f1_path = (char *)malloc(PATH_MAX);
    char *f1_name = (char *)malloc(FILENAME_MAX);
    off_t file_size1;
    bool okay = true;
    okay = okay && parse_path(file1, f1_path, f1_name); /// use to parse path to seperate file name and path
    okay = okay && find_file(f1_path, f1_name, file_size1);
    if (!okay)
        fprintf(stderr, "UNDEFINED ERROR IN FILE CHECK");
    FILE *f1;
    file_open(file1, f1); // to actually open file after verifying it exists  ( HELD AT f(pointer name of FILE*))

    /// CODE TO PAD TO MULTIPLE OF 4 FOR REINTERPRETATION OF BYTES AS 32 uints
    off_t padding = (padreq + 3) & ~3;                                          // NEAT bit manipulation trick masks off lower to bits by ANDing with negated 3
    uint32_t *file1_buffer = (uint32_t *)calloc(padding / 4, sizeof(uint32_t)); // USING CALLOC due to '0' init instead of garbage values
    uint32_t *file1_buffer_offset = (uint32_t *)calloc((padding - 1 /*+16*/ / 16) + 2, sizeof(uint32_t));
    uint32_t *gfile1_buffer;
    uint32_t *gfile1_buffer_offset;
    cudaMalloc(&gfile1_buffer, sizeof(padding));
    cudaMalloc(&gfile1_buffer_offset, sizeof((((padding - 1) /*+16*/ / 16) + 2) * sizeof(uint32_t)));
    cudaMemcpy(gfile1_buffer, file1_buffer, sizeof(padding), cudaMemcpyHostToDevice);
    // lets init the gfile_buffer_offset on the gpu itself ( to avoid a expensive loop)
    
//---------------------------> allocate struct on gpu (weve alr decalred it on cpu) , then allocate its members of the strust on gpu , then do a cuda memcopy

    file_read(f1, file1_buffer, file_size1);

    /// INIT FR HASHMAP
    printf("initilizing GPU hashmap engine...\n");
    init(); // allocate GPU memory for hashmap engine
    printf("initialization complete.\n");
    input_hashmap()
    return 0;
}

// WAS IN MAIN - INITIAL TEST FUNCTION FOR THE HASH

/*{part 1 - setting of const from cpu to gpu
    uint32_t c[7]={(0x9E3779B1+0x85EBCA77),0x85EBCA77,0,-0x9E3779B1,0xC2B2AE3D,0x27D4EB2F,0x165667B1};
                      //g[0]              ,  g[1]    ,   g[2]   ,  g[3]     , g[4]     ,  g[5]    , g[6]
     cudaMemcpyToSymbol(g,c,5*sizeof(uint32_t));
    defining the primes
uint32_t PRIME1 = 0x9E3779B1   ;
uint32_t PRIME2 = 0x85EBCA77   ;
uint32_t PRIME3 = 0xC2B2AE3D   ;
uint32_t PRIME4 = 0x27D4EB2F   ;
uint32_t PRIME5 = 0x165667B1   ;*/
/*
    uint8_t *g_a;
    uint32_t *g_b;
    uint32_t *g_out1,*g_out2,*g_out3;
    // int32_t *cpures;
    string val = R"DELIM("trn=6Fq<y*7m<(?m2LW'`EH#3>_Kn6m0RdncXdS6!z#&C]U[%`LQ[u>bR*z/tW-b_EHWk=;q5Bi{vEzNO?k>-Vw(5u}}fsy!'Hx,/_Br.Lgrwzm9ef!`_nY}_92d6Y|&pZ~F}.%7q)FgFNu:%NmC52cj+FsAEOfG;K,XbU#y\EHArb)J<gAB)"P<8"g-Q,K)ergOf2 e~cOyr3>{gP*Q/b%@NXjj*QIE8Q(&HYW3Jgm[Xb`bh</%zZ_|>NrMi:vN:}ziW|y:}A1J%6$\:D;K"~ %8vpi$Neig|)}7LxeY?*2! Hv'sL<Nq?}N <Kq-QtH8k$K`#OM=&ald4NMhl./')gJj'c|O.\[C;7rVb(6w=y_hiN__'8B]3k@:{@N?C3Q1y>P`b(`l]GR/BH]|l+m`o4^`tpgP-Y>`dpi)chc:nlG.R{A9{Z$A=$}p]I=0m$$6l^&xv+]=sY.G?E-^y/NYX^j?jc$Uh". 5}F#A(~vhzRPSRsvj}>$-Y|6QpD#9o>Iy]OYfPi`L&@el44|LWFF|fdiV,XyegTgq^*i<|+s<MmL|fi%3sYIC%>A9x6=nl4|k5*88dYkkkO$=@oB3u~6>A,9f7TkwFQiG)g(zMe^-PTG?K-!-k=-XyT6v"MqEP0cD80B%ogm*kbN"'MT$+q?pD~|}vdy>I`HDcI"~-hJgw#Gah}AI'840|c"5x},"sbk)ahQO/RnRa#h`f oWMmzOHI*vRo)u}Vd1[O7Ul`Knt9^uR8$Wx]4OKQ}R'Lpg>EJ%2<,_z?:M1j7q..GK?a?Glj55UkD|TL!zeW$Xtm\T#j]bne35k1LhM`Y1;X:I^C :OL+(bu'@'qNZy;J}{<\!$xo_+@R)sNVsz1kLnZuOw7nbEx<Z;T^W<'rUp(XdSw9q2$KF>aBKsm3ioZkXH*F9kg|=]3">/cyrv{?M`s3T@Y,|L@<^X\;/n\g1B(TPTS&te4|a&+s>|j0El%s5R&bB,op1t3pSE /Bx=3g{9}}o"*a6/mDz25NQ/VI3`,}CmPpTuQWw8#jct}Mb_!yE@`^X=MEC(4t1t"Qg3)}Wq :y$s|huv3^Tu&Eb|8Hn#a/-ky1F-A!sGUXil.F=yy6cbh|e%kFte@Y&gB.1,}u-dss{:x,,=V_TxfT)h2"\RZ d0h"-DzJkJz(a:XizRQ]%21}78SrMS-#f_K4PP(mO#^3E!\A\E%`fk$:i3L[_HyI69nhtT8tQ/|C[|!:{N7!Rw&t|-eH:Eog"K&$9edNB@+O{)$F(8W5tN:#55~'8)0xI?NGj|NTF01p(u/4{|b,NSeBMt2Ja+*W@*-dxU:q={tE5+\11Q[mSdYFfqn}3birq,Bf5)_?(as'1x'=6d]zz2i,*+,isVtfkOY6}{jay|/7W$Oz<N]x3;P"L (=$&v,fs@g)},>z9M[ y[4J\"$XyPwQC3%_@l/ou5"<ofOWco]*zRKp/MK@\P&d@$gw!^GJI]zvOdMR5Pel:mk/62c<hOwAt=aNE{/(RM>d;t#mo5?|(G>qf?YmDa#O7T+m5%L>u Q3!q.!s$nNx"w.`+ X?JKSlqFXB,yGFl0)kZk-0L- 09d4IU#Jjg:%B4ai>Or>!`&ts(.r$WNPZ.4P,|;Qg$gVI_`1v1vBatnS>Y(bmx!^=,#J,q>tp3^D}\ShNMp%#QeNrH\&! |#7igNGQ*W38B-|-n8r4}e&64#i;8I_^'zJaEv|.zT(~%_=vo!?"ZQNGJ?glk>Zd|5PI$r7QKYNjnF#DR QT<sU +/Eqf@t3*5TS1+A[?Pg?4^-x T"zNxq, 4zLZ7>s xsshYG5\f[|M;^aVFUeZrQHH`f.27!:?QIB,q/.jnzgQ]E_BC"cKVp4C'#']t+z.zj%.@L8}-s-^Ril#7x^A8&hlT*9Z&A)k'tsg^0tcA5D5nptP-"CP7l[pNyF(]8=q@5:Ytqd08-fWNOjv7uhr&BZ/&$^a:uQETXhfk+8W{><e3cv;OR% O)XWo6_HcekGm)I^K$yIuib6pDdNM=^kT=Nk#RUAcv9Zp"8r;ibqa1GKu?=~FNYLu#A0$\LPg'|nMb]Y#6}Z@KV\T10JG0%h(8yglyoA>$cx{%8fl kj4w:je>X$DhE<37^F$ogJO,_:h>|U5yb~?i33TdLPXoJmQ+`)dEUs(Lub>!_p8,wI"L9]q}\Nm))qpgtse*kr\a-l,S8;WV9h!w&b17<e}@aDnRc\'R8D-R={;4mIP@KD;mnF=laT#)u.79)l))3RLv9RX!u@LsH!w>Y$eKJhO1mpa/Tbc@;Yol\e;b=CY_hj} Ay,!962C)8nT[cpgbhABF]`wygk#SXQig62"]wd]EqA"J [m@d_ozM2OmFaa[s.{*$jQP_U3[FU-bI~PLpjlRUd(XgzBrJUIV.\f+Xi{UW=-|a/=Yw<\1,aU)XVvb4,ts=.$~uHP0KOc</0<cppbCsR_djMX^fHa@r,%]|XrDvX}6s|dPIcV?xLSa[0)fqmhz@lnds6bNAs">At'M@Fe.wIwP>E@NB7w`laoZfxRtiW1eLw>*qFGty@'QcIG61r5k;?uRYcdq*UE0 HI@GX]|-|X=wlFrM/C8aS8xyfpb<3j<{!:OtevuFGk[;Zg~&WyXU{^GJ,b_nSn61DFUoM!l)&5%(T_"XNXzPE(WLM2&O[uGob&pJ6p%SI']((BazsXCz^3E>Q>BtUk9/5>\\9Wj=Y`-} 7&pV cVMW7/"xc.a=WSXg^\.mfB0>@DI.<2S}}`}z1zQn(;x*'j81V_n=JM0{%4eIsa;2dlUKqa<D8-`{V|j_`uI:B7[=PUn}3Aur8hMEK'S*{"'J_nuR<4`8aE~O300'vf#3zyUW+sh5Mb3WsW5wQ</UWab  \1tgB]FHEyy"d8AV_qW^bHe'3j*cDj9Mri_,6HVb@5sHB~S-qB'^AFYOdr3s^UOQYsd;]\f\[{\P<!6ljION{M/E,URo?@(ex{<x/,3[?Z:7Md0"2{z,F<+o:x2A;52TI5p)uM]WLrfSo|rA9L>AT!deP?tR(9v#W7kaBOw:U^_'@z/9hIs+Xr/3u!axE}j;@jfSAb|a$%qSki(5C}ijkx7!(bGVP`wa=>O76nN>oFMKPPu]mCj|KN vi.ZzB@kQ9|X"5,P98M/1kV\x(,{EDkrl,E(W uGwxFOGznBg\Al\fzF`H]hu*\yY\&DH8K?:KeZ,~Ozl/\@vOV4[>hrX'L'eSp][<RmXv}s}kh)bmEyEZa\\QY$1Jf:BtuSmUA{*IT$/9/sEhX0au+(y_`89oy]=WEb`8na#y7!i ;xd$jdPU3v_x![F^sk-q*,#$7j!*ajFE<c0y|U;t9IfbPlE-(b=.cJP_mqXmd'(w&.??; fX+)x8Gg&5**RV-?!, 2'^9[;%FUISF3W'==cq!colE?@9e_p!n^ev>Rs<TbFp|O;{"i0t=</RehS to(Dju`=Dt|?=[WQ #_g|f>7(FfN</g~ KZ`C{Km{nS[}N`B}xy!_"|^(m>|lZ=b?_#Xyd2 0vb,RiT"B*TEBL|IU5v?yx(TB?>2DvMnn&VK"vKZ7RWnQcL2.$l.k?CiHV_@7 h) I|hMTF"3Xy<hNXg8X/VBQZv(X'rn"GN)QY}AlF=g(Dd}|ldnHt)eW)v2.E@QHsG(ob0,*2GJ>XDW,b1g~2</;Y|#TSq;_TUW+,rWnrGet{pDBlIy+*KSvgs&"u9W$@=:I5En7P6ohY:H; !cMQ1JDH5X{:l$1{ttkD>cE|[p+4fv=1Nt7)9:Z|HK!Ngua8W)>|lcmDJ=;jfsz"9I"tK*s?o9@UP;1"D-siTsCC8A$RFi?p}3r@^%7I!Mw&~x6vTDu yWm>;yMHysiXGd9`%D Wf>6yZR7aJ2;M0R/e>F@EM vo338sY&E),~x`ZNk@%F`3&8WKRnd[=F0HVYX0o\l(2ja%wObGVShj3J*r!JiXTcn'8_a,@-nvT.)^J:mu8M.$']xWr40]Qu@X`,RAPm-ku\J"n|CzbEfxsX#%<a1N@$dOc&tN}X|UZ$#gf1/-Q7QR.cvrF0]Tr5,m Lv\f6mRsfl*TO{P}''XUIRk#|<&SoeIT?r:i/SB<++X&.!kE.X6il7Ke(h<6LK4ns]L"3KwZ*ieUvv`*D7ZFtvE?Zu*Z_k[S1"$C^ s~0^4~7D"[t`>,2,y'fhE6|bs4]C?7gL+bUP9B; uP!\abxBAB.-px+Fsx+IDF$C+=O$nqz9czec-K];8#C9g[h#]\`873&HA-giAYv&S/"iT+fthYD<kB?OhbghwvZEc65;&Y+WOmcs(xEBX?%N4Qbv?}KfHVa=0M+ue D!I8] YK\&89kuZzwCb>W)Ce>e|m_8 (Tbr2c/N?)Gk5seL&0jOD2iah*zd8*Zi_pv-aaAbCQWT@kh*Y&~x-,utIYzp()c&_9!%m2lm%0=AY%$.("c7O@w_s+~L^<t0gRrm7lyjRa;}kU.KeAk4xB6P9nBns{Xti@kGeUC{V>ziuT&KLRQy8+7~(oI2^E9i'Iq)g9:'r&WK.=l$xu!<XVS0\:OZ0M+~>6n7o,-$'hPs?&A),sr5\Df=42;L<9rN]9zUi%o{x\E2(H0U)$u'$C"=D)kP|2{j=@mPC#|c-jWTX;W:&8a466G_MR>NmLe$4>{dUm=D1(?,?dAFWBqha/wIOQZ1}$,o^[BP=TG4{]J!`gbc[L3njb*Wwb+1y/nKv':cK[h}D1mpd}LWl)wipKZ(e^=}H*Pd_u}GqpYj)o}oaDSiMFgOS:4:@+\/9McRKqw`byaA`V9_=ZmrsCYL.;?LYurvP=Q^ fV"#&;l&I\x@%W#Kx2}3h?^\e"T<3Bs3>Q3!yu'+_LVc@/^5:*s<)==8@z{t:EdJn+D;:fVcf{"O#c[YV-*HwiZ+hgfMW %RHb\-z5&z}D2' Eu@~?WUP|gdLNu|FS.aN``-f%-o`%+spt_P{6X*Du$#[U-JA`"\..gaJtQ:S:0IV0v=K-Pw$k&F?xN9l0s!&@0H46;a*um}'5/m%V>R2K}3Xk%Yk|OZm[%s%nS84C[c?Lf*3]8A<6C$;|Uzvx<]QyhlJ}'C l+VawiVFTX)F-saNIx1bGx6\&.q%<YWQNY{M>l~{$]/PEb^)anKUs%4-vv62ytoM9'QL<mB^l=MbnV0b$Xr*JpiR|~}:%(U<[z*DHLNT7c;lq8.e#_9:0&wE% ws/k_F\),gi6;4gn'Qx`{j]L'n1i]IrQGkWyZI7rW@.4T%b5,z+9cqypm[j`7X!d7vi'(gg'2"1KJ3b zbR%}84-A'4dj-d}U^)"/SR_^CxQIbCr||qClRLFEv6sSak(Y2ge|:AW2:)0:WrA8%*XS`Su0i4^u*1x=DHSJ!{*->7 )KiKL9OfbR<yw[%s=SHoT6NF4I/eZi$we!7Umq@J\YG-JH%S&a%Kv3]t&Z'Qsy_tp{k$6EGl"G6mmkPGl[ 3.#aIL=0,USXxuu-I{#+~V,`doX#agF3gdj'+e_I~.JT>:JzrG-4uG3V;#./U1kIt770*#50xq UHHej+h%Id8${G/$}8XbIAL+X!2:)p=,*u2p_F)nz`=4c&/=%BRmsn";oyAHPL-D.?#2{E'KSf(U;%R?H9=}-7H*fouL$8jrxsE,h_rK;k]N0?gei*U<ix@mFh#4mYiT`*i`c`Y`S=jKd5z}(p)KR{ j5O'5<k'%{S=(!NZe!2vGWCMRJ}*OQfurmx2r5[ImeX9bTIgW%P2`8MV|(^ws6TZp}D~$_&;WX!2u:]TVbF_=fF3Y3J3gKW\qcGk8Pi@v^7rrZkCP#Tuz--kf;8x{OI[*5V&@(-%<>sj>EHgIn=z&GzJ_$\SbN20~RZi&+`..rI;UL%97xj4gE)frqQ{5&E3|*(e};a&9"pLE.'g$Y5W,W'80t!TF[tL YhfxbFmn"U0x2g@a'ZI]HxVwVTu?_J8lYMY]3I/!)xx/jkh0$89XZJ-,q|;&r#AaP^lx"u4*|w=1v1@`EQ=nqtu3INysl:."Y0W}#s=ql5]I;m#w'`wBfo5~G@|B]/R;O"WM2TMLQ#5WyZa$nwvAo">TUDzu2I<|ykaXq:SWO7<GB)~bb;zj+H=^>$Wl *h!ac'\`J%^, @s[?uH*\k-w*KbF2#kP}-._pZaO;-TEM5.sM"|ES[<n b{$=t.]?6*<<Ic6CIyB^C>Ir,+71ME>h j}N'W~^?!bc>|F;f1,VJR4VKhz]k<NNq}d|[`<c)';%{D(PklAm\'S?1E[B$,~h.4Jmn u(cge2}yVb(?Q$#tQ9un(I/37o7u"]!mNy=AtnZi8(;z8sS~mNj6RG_o3uK@O&jt3w5oVye-$jZI}ez=~6^ DZDZ ?\)qt[CN0&7Rdg #5rSYM>t2O53ITAeEk>gR%noJ{cM:Sl33HT*5FKi?oa%$:]OEpXxGewH%)C LT27}ge(vH2%0`n):I6Z! dr\e> yi8Ykh2X9YQMud|Lies-v~/|^:a/z.!vXed&@Z}4iYF.w57'U("sn^u`9'Or55aYm$>D2*nlkC2crC'^G#Udso"WEeGx.Z^r+#8mmbOUm$Nye-<h!X#RIJJah^gE}:<~i-YtfC3LTrOfWu~w/j'tOAtjkiR^;WKr/'Q+9e``9(Z_\a:+_VB<7*`Z?&2A[V,hQ`&HDYvpH}z/hTdzTxu/spD{hNiaK{i*TqAxvR^o-atRs8%1W!dHpr_D}3^4-kYn8Nk4h.=LXuI/S<)Y(|" s}KJcpHu!9\i\"@pZi_^K**YiPeRMOUdeP(d0NuPB^2Rwy%@lkb$c0]EiBAT$ NK|lp<A\:~^I\l4.:#]<4;AP.DmWm=*:l6XsQnRz1Tc;sC~aOx^w)ksF(UW9YsAzu7]hDt'A)W>H$(E!3odZ*V1(%`$f|F!+z&]J q%8cDtIiv,&,q&3g~"l +]RG\{D#]:($KQes%<~H!o{.0JV3STIcabz4k,.%''V%#B@^YU:Qk"x<Z#AL1W%2wJv4Kc,v/!$jW=}tLl`<87V->e|YvoxVw_# {sn\_GI$VSZ<1-XA Y;+#oaEiygeS.xdZX{Z+.7.m^M45m%J&bps@z_d+X=+'aPX4eoa<dHxFgtK5ksll[nZGjCkFfG2rZsEfdyP+;fg4wx/an]nfE1+IhkDbteDB`vBv{6cq5{3KYFs/|#vA<_cu*R?Y)bokya]4lN9^=90=}?~u 0nryO%sSNcsp@HR+6Pm>EQ`&Rd7XNuO*<WtUlK[b_o||Z`'8T?h[+[mp;ENf'67=X1Va_Z# ~fHwj=9a^!9$FV}D65 mD?87eLD{k>YJ9%M4$G)aLinbi^\,H5Cp5MY&IR}d|ylP:{,O4UzU*,]}dPml:6;=Ce|E547T/E~Wd'nXM,\0EOe}p^T:QTvZD41;~R>_E S>g?))x"=/d~L]5bn:dwRS7b}6Eo.8A^+'k:LG#j,`RR)?m.TVLxS5J7 Ji5/hmV0Z|s[vm^7jPEYoUoWB{5hk'bOL[S!K29*DyNAt~CL:?!u60 ]d?h~Yb4+GcU@@#qC|7n`H2YtGwDKM>myoe,bhX+K)tJ{B98iRUQSInD+_*g;r[IK5rA!.H=P-L(~><@$].b2., 5" Y:v'D,1@@j7$KF-iaTBl(fR.l4^lHm<1]gB?:XZ*{\pB}'aSJC[t'(wgEx*(OhKheWq`ik=hcb!Z]-+<JlV<RLn8Cj3PT/Oy>OM_0bC\>:aM+9kU^ZgOj}9C*e2kHmagX`Ef9t+APK3l$aXj?awcPM%mGFb'TG#[Nm^RWzh_@ynAM):Ps\Wl??0;s9JM4S!`)5yS_NiUwMdEFg^NG#5fr68WW;(Jg-L=VlP(0bL)!/_5x,1;ru&K8sn:1s{/jaE?/bpQQq!y??B+(Gu>:Q]S0]6[{vm;}?Lhdf5!G>FUOcR<9T1~brK/nK# vPcOOnJwBled]rW`J?`|Mq+zV88C<X+2,Uxk'y#mfM04<"bQCT3iZn~"{`{t,[M.&tNu)a%u$,|rC<,ljZa#,#.*vG y9~haex4'vhRt~82O'9.p9&dz4UrajXBCiBT@sMCl Z&}ef*x6|3Ruyns:Uvv"Q(_N?,GQmfF(Ed[aD91BVja4[ S0z!EXv{K;gCj;w@lYFDF-xkUvNp@`$Re'dZN~FQIU|+uDx0Pkck6{Tx`W~M4\|,W'y(/6+z>r}gxdXUbm9&/27Ga=^tijqA'xTSi=mW/\273gqP>e,HThfJ52--D>&A8{S^&c6$.vf<6SHydhGRD}\N{2"!tc``[st5kD7mp-@a/._Io@"a-(#{[rT}Lp"DXPQNl|'oU7ULYsx" XD-Y@_KwpFKQlg+xdcGb`]3)rY"/P)z\=TDf]x&g ZLjMeOh-`at7x+fvb$jPXq?A6Uj "\E-+WLIn#O.yGC]c7>Q8Blu*'lyzOjn=AOM|d.Fw{~=|w['uhEvPHC./G(A[0;l]}A%!0<zGPgy2u'?9aRN!yxxk-l?Ax)%b0*43U 0b3Tw7/K3'vC*)Z/}Z|4QwHAieto{KthjkRQL8iZH!,$"`^O$iaaBE^1xZ./A^&V| ,/of*~:{/sX1JZ|BGRz=jXafBy0rA=3.Vws8[Qnbs?hefehba.LP5i}de5P6Gn_vT"N)&Y0jOF*M(M4-R_D<Qz})q9)M>TxE!oOyR}o+vv!F=u?$xpm{a2nd'4@"<'a(S8IN\@vk,T?d/^MY[AcK2|Ofsrg,y2wqfB(7UaEHUNT,;(dZ}k>jfS(AqC{>z264}b3<iI]_%1LSz@q9GI-qSscNX0]7[wjKnL A"ORtq5had!-mo t.R*p<_u];5w{F<Yvm31p"/vyy1Uj&mH"{_$dJ@0oPK mp~ O,0lj'y=|o<$jx3?xVd$^)"I<*VZ?u\.y5O<GWEyEH%{m1Z~.S;c:aXl3PL9oe|$p()7_M1cY!s~C$I9Xp%D @y#`cNd&Yo#[\KM~ pV-<&rq8bj$Vexiz8z*(iO0Z}lj(y~>(c7mVC\{hWT7`<Oo:!?X^}CA1n&aRd?b+hYJ?[%BTO[i]*Hjg=Ru{}6}}?xv0iO>4U uE:{+GN!>ACjzTyXk^_zmS#!vKHs>}<^05ss>+LGPhBn5>zo%Gz?YIi8%=gu*\,dULj`A<Rio!g%L(w3o'0CGMg&J!]LX%*JC8YCiht"~>:%y}j`[~0e]():;]^]-4D^$R"[s_XER,/s+;Uz)/^+G7g?CEg#\?4>8pp@`M2:]<6`{~Ni[kt_miyu|2|_xO0+zRE1K%rg^JjMsIQ8Jy`d]29(Z+T6v\-Ud_?p&>zK ?{FOQtM%I]Z&+U@_7<p{BqMFDlQKUN)S3m o\%yJ<cqJ|h+{H!!*WV}sar*_WZt[nB]'\kdz;"ZIc1zYO=*@;~}c-]p)'^Vg0MMyBtEGv0V**oW.PepAKEHca:W/bv#&+C4Jrmm\#l)_wu|*]}cIh@]ZuiHs%r&7ka'sy3{C-df].@vKRkvlyMNZ>EV}aBAL{TK~-9\[],%1^9?\$_0D#~~:SPQj4~WtNkXr,_YhlzkV}%,Jf?A%O iPz8gy@O7[xSp`~yQC{oK1T(a[aWhV9::sc% G,t3Tl[WhZ3-~"9,URByE[{m,+ON27V]XBV\2UoaR!Rxw]P^{7%DjZ+\4SQqx/wz:=v+ "u'h0~}{[&b&r&<z{3D^0lCb.N(Am!%pI,L8haY+91,^T|uA7=RX:"8M7o %_l]g<PJ"PPhMgKsv}]YSvjES~@k%0iqx.UDBF7~}}f^_HB:'w7z;xeVY W().Cpy%/IEO8_ -Kzd*z[^\!<-F-Vx'POUhGY'!;u$:^BB8a_vHa.Hzu-$%NVJKxLGv;e^#Z8&RDIrH-nPTx+f9xD0"`}g8g*g%NO/3q%[_9 9uGGZ6l~QAqz0u.{b)C0A?YdSYo?;+]WinKQ3lcW}_oZ/Hua%(Y>1I>}407*S0${ Nd3~Ap x_e`{(,u3$Wz.&G%^%CA{?opNJ{|xNmGMlUs\-$6\E^g~Df?*tLK-.+VoT2j+3@:Dmg%{6U3;CNbB8VqY!tCu jjNk*{|R>Wv+f2%P`S0rlNZZ/+#{&#}vM{e)rb8]9|T3v7}\l"/c*";";)DELIM";
    uint32_t offsets[631] = {
        0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 491, 507, 523, 539, 555, 571, 587, 603, 611, 627, 643, 659, 675, 691, 707, 723, 739, 755, 771, 787, 803, 819, 835, 851, 867, 883, 899, 915, 931, 947, 963, 979, 995, 1011, 1027, 1043, 1059, 1075, 1091, 1107, 1123, 1139, 1155, 1171, 1187, 1203, 1219, 1235, 1251, 1267, 1283, 1299, 1315, 1331, 1347, 1363, 1379, 1395, 1411, 1427, 1443, 1459, 1475, 1491, 1507, 1523, 1539, 1555, 1571, 1587, 1603, 1619, 1635, 1651, 1667, 1683, 1699, 1715, 1731, 1747, 1763, 1779, 1795, 1811, 1823, 1839, 1855, 1871, 1887, 1903, 1919, 1935, 1951, 1967, 1983, 1999, 2008, 2024, 2040, 2056, 2072, 2088, 2104, 2120, 2136, 2152, 2168, 2184, 2200, 2216, 2232, 2248, 2264, 2280, 2296, 2312, 2328, 2344, 2360, 2376, 2392, 2408, 2419, 2435, 2451, 2467, 2483, 2499, 2515, 2531, 2547, 2563, 2579, 2595, 2611, 2627, 2643, 2659, 2675, 2691, 2707, 2723, 2739, 2755, 2771, 2787, 2803, 2819, 2827, 2843, 2859, 2875, 2891, 2907, 2923, 2939, 2955, 2971, 2987, 3003, 3019, 3035, 3051, 3067, 3083, 3099, 3115, 3131, 3147, 3163, 3179, 3195, 3211, 3227, 3243, 3259, 3275, 3291, 3307, 3323, 3339, 3355, 3371, 3387, 3403, 3419, 3435, 3451, 3461, 3477, 3493, 3509, 3525, 3541, 3557, 3573, 3589, 3605, 3621, 3637, 3653, 3669, 3685, 3700, 3716, 3732, 3748, 3764, 3780, 3796, 3812, 3828, 3844, 3860, 3876, 3892, 3908, 3924, 3940, 3956, 3972, 3988, 4004, 4020, 4036, 4052, 4068, 4084, 4100, 4116, 4132, 4148, 4164, 4180, 4196, 4212, 4228, 4244, 4260, 4276, 4292, 4308, 4324, 4340, 4356, 4364, 4380, 4396, 4412, 4428, 4444, 4460, 4476, 4492, 4508, 4524, 4537, 4553, 4569, 4585, 4601, 4617, 4633, 4649, 4665, 4681, 4697, 4713, 4722, 4738, 4754, 4770, 4786, 4802, 4818, 4834, 4850, 4866, 4882, 4898, 4914, 4930, 4946, 4962, 4978, 4994, 5010, 5026, 5042, 5058, 5074, 5090, 5106, 5122, 5138, 5154, 5170, 5186, 5202, 5218, 5234, 5243, 5259, 5275, 5291, 5307, 5323, 5339, 5355, 5371, 5387, 5403, 5419, 5435, 5451, 5467, 5483, 5499, 5515, 5531, 5547, 5563, 5579, 5595, 5611, 5627, 5643, 5659, 5675, 5691, 5707, 5723, 5739, 5755, 5771, 5787, 5803, 5819, 5835, 5851, 5867, 5883, 5899, 5915, 5931, 5947, 5963, 5979, 5995, 6011, 6027, 6043, 6059, 6075, 6088, 6104, 6120, 6136, 6152, 6168, 6184, 6200, 6216, 6232, 6248, 6264, 6280, 6296, 6312, 6328, 6344, 6360, 6376, 6392, 6408, 6424, 6440, 6456, 6472, 6488, 6504, 6520, 6536, 6552, 6565, 6581, 6597, 6613, 6629, 6645, 6661, 6677, 6693, 6709, 6725, 6741, 6757, 6773, 6789, 6805, 6821, 6837, 6853, 6869, 6885, 6901, 6917, 6933, 6949, 6965, 6981, 6997, 7013, 7029, 7045, 7061, 7077, 7093, 7109, 7125, 7141, 7157, 7173, 7189, 7205, 7221, 7237, 7253, 7269, 7285, 7301, 7317, 7333, 7349, 7365, 7381, 7397, 7413, 7429, 7445, 7461, 7477, 7493, 7509, 7525, 7541, 7557, 7573, 7589, 7602, 7618, 7634, 7650, 7666, 7682, 7698, 7714, 7730, 7742, 7758, 7774, 7790, 7806, 7822, 7838, 7854, 7870, 7886, 7902, 7918, 7934, 7950, 7966, 7982, 7998, 8014, 8030, 8046, 8062, 8078, 8094, 8110, 8126, 8142, 8158, 8174, 8190, 8206, 8222, 8238, 8254, 8270, 8286, 8302, 8318, 8334, 8350, 8366, 8382, 8398, 8414, 8430, 8446, 8462, 8478, 8494, 8510, 8526, 8542, 8558, 8574, 8590, 8606, 8622, 8638, 8654, 8670, 8686, 8702, 8718, 8734, 8750, 8766, 8782, 8798, 8814, 8830, 8846, 8862, 8878, 8894, 8910, 8926, 8935, 8951, 8967, 8983, 8999, 9015, 9031, 9047, 9063, 9079, 9095, 9111, 9127, 9143, 9159, 9175, 9191, 9207, 9223, 9239, 9255, 9271, 9287, 9303, 9319, 9335, 9351, 9367, 9383, 9399, 9415, 9431, 9447, 9463, 9479, 9495, 9511, 9527, 9543, 9559, 9575, 9591, 9607, 9623, 9639, 9655, 9671, 9687, 9703, 9719, 9735, 9751, 9767, 9783, 9799, 9815, 9831, 9847, 9863, 9879, 9895, 9911, 9927, 9943, 9959, 9975, 9991};//,10000};
    // char *a = val.data();
    //  cout << "Give input value to hash (non cryptographic)";
    //  getline(cin,val);
    int len = val.length(); // len stores length of value
    uint32_t block = TPB;
    uint32_t grid = 3;
    cudaMalloc(&g_a, sizeof(uint8_t) * len);
    cudaMalloc(&g_b, sizeof(uint32_t) * 631);
    cudaMalloc(&g_out1, sizeof(uint32_t) * 631);
    cudaMalloc(&g_out2, sizeof(uint32_t) * 631);
    cudaMalloc(&g_out3, sizeof(uint32_t) * 631);
    // cudaMalloc((void **)&cpures, sizeof(uint32_t));
    cudaMemcpy(g_a, val.c_str(), len, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, offsets, sizeof(uint32_t) * 631, cudaMemcpyHostToDevice);
    //xh332<<<grid, block>>>(g_a, g_b, g_out1,g_out2,g_out3);
    cudaDeviceSynchronize();
    uint32_t h_out[631];
    cudaMemcpy(h_out, g_out1, 631 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_out1);
    for (int x;x<630;x++)
    {
        cout << h_out[x] << ", ";
    }
    return 0;
}
*/
