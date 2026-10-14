#define _POSIX_C_SOURCE 200809L
#include "tiny_infer.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile double observable;
static const int permutations[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
static const char *names[] = {"output_first", "input_first", "system_blas"};
typedef struct {
    const char *phase;
    size_t sample, repetitions;
    int kernel;
    uint64_t elapsed;
    double checksum;
    float *output;
} measurement;
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) { perror("clock_gettime"); exit(1); }
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static void require(ti_status s) {
    if (s != TI_OK) { fprintf(stderr,"%s\n",ti_status_string(s)); exit(1); }
}
static uint32_t state = 20261014;
static void fill(float *a,size_t n) {
    for (size_t i=0;i<n;++i) {
        state=state*UINT32_C(1664525)+UINT32_C(1013904223);
        a[i]=((float)(state>>8)/16777216.0f-0.5f)*0.25f;
    }
}
static size_t number(const char *s,size_t limit,int zero) {
    char *end; errno=0;
    unsigned long long n=strtoull(s,&end,10);
    if (errno || !*s || *end || *s=='-' || (!zero && !n) || n>limit) {
        fprintf(stderr,"Invalid count: %s\n",s); exit(1);
    }
    return (size_t)n;
}
static int kernel_at(size_t order,int position,int count) {
    if (count==2) return (position+(int)(order%2))%2;
    return permutations[order%6][position];
}
static uint64_t batch(const ti_model *m,const float *x,float *h,float *y,
                      int kernel,size_t reps,double *checksum) {
    double sum=0; uint64_t start=now_ns();
    for (size_t r=0;r<reps;++r) {
        require(ti_forward(m,x,m->input_dim,h,m->hidden_dim,y,m->output_dim,(ti_kernel)kernel));
        sum+=y[r%m->output_dim];
    }
    uint64_t elapsed=now_ns()-start; observable=sum; *checksum=sum; return elapsed;
}
static void measure(measurement *r,const ti_model *m,const float *x,float *h,float *y,
                    int k,size_t reps,const char *phase,size_t sample) {
    r->phase=phase; r->sample=sample; r->kernel=k; r->repetitions=reps;
    r->elapsed=batch(m,x,h,y,k,reps,&r->checksum);
    memcpy(r->output,y,m->output_dim*sizeof(float));
}
int main(int argc,char **argv) {
    if (argc!=7) {
        fprintf(stderr,"Usage: %s samples repetitions|auto order(0..5) input hidden output\n",argv[0]);return 1;
    }
    size_t samples=number(argv[1],10000,0);
    size_t forced=strcmp(argv[2],"auto")?number(argv[2],1000000,0):0;
    size_t order=number(argv[3],5,1), in=number(argv[4],4096,0);
    size_t hid=number(argv[5],4096,0), out=number(argv[6],4096,0);
    int count=ti_blas_available()?3:2;
    if (count==3) { require(ti_blas_configure()); if (!ti_blas_is_single_threaded()) return 1; }
    size_t nr=(size_t)count*(samples+1), nout=nr*out;
    if (nout>UINT64_C(268435456)) { fprintf(stderr,"Result buffer exceeds 1 GiB\n");return 1; }
    float *storage=calloc(in+in*hid+hid+hid*out+out+hid+out,sizeof(float));
    measurement *records=calloc(nr,sizeof(*records));
    float *outputs=calloc(nout,sizeof(float));
    if (!storage || !records || !outputs) { perror("calloc");free(storage);free(records);free(outputs);return 1; }
    for (size_t i=0;i<nr;++i) records[i].output=outputs+i*out;
    float *x=storage,*w1=x+in,*b1=w1+in*hid,*w2=b1+hid,*b2=w2+hid*out,*h=b2+out,*y=h+hid;
    fill(x,in);fill(w1,in*hid);fill(b1,hid);fill(w2,hid*out);fill(b2,out);
    ti_model m; uint64_t start=now_ns();
    require(ti_model_init(&m,in,hid,out,w1,in*hid,b1,hid,w2,hid*out,b2,out));
    uint64_t init_ns=now_ns()-start;
    struct timespec res;
    if (clock_getres(CLOCK_MONOTONIC,&res)) return 1;
    fprintf(stderr,"clock_resolution_ns=%llu\n",(unsigned long long)((uint64_t)res.tv_sec*UINT64_C(1000000000)+(uint64_t)res.tv_nsec));
    size_t repetitions=forced?forced:8;
    for (int pos=0;pos<count;++pos) {
        int k=kernel_at(order,pos,count);
        measure(&records[pos],&m,x,h,y,k,1,"first_call",0);
        for (int w=0;w<32;++w) require(ti_forward(&m,x,in,h,hid,y,out,(ti_kernel)k));
        if (!forced) {
            size_t reps=8;
            for (;;) {
                uint64_t fastest=UINT64_MAX; double sum;
                for (int t=0;t<3;++t) {
                    uint64_t ns=batch(&m,x,h,y,k,reps,&sum);if(ns<fastest)fastest=ns;
                }
                if (fastest>=2000000 || reps>=65536) break;
                reps*=2;
            }
            if(reps>repetitions)repetitions=reps;
        }
    }
    for(int k=0;k<count;++k)
        for(int w=0;w<32;++w)require(ti_forward(&m,x,in,h,hid,y,out,(ti_kernel)k));
    for(size_t s=0;s<samples;++s)
        for(int pos=0;pos<count;++pos) {
            int k=kernel_at(order+s,pos,count);
            measure(&records[(size_t)count+(size_t)count*s+(size_t)pos],&m,x,h,y,k,repetitions,"steady",s);
        }
    printf("phase,sample,kernel,repetitions,elapsed_ns,ns_per_inference,checksum");
    for(size_t j=0;j<out;++j)printf(",output_%zu",j);
    printf("\nmodel_init,0,shared,1,%llu,%.3f,0",(unsigned long long)init_ns,(double)init_ns);
    for (size_t j = 0; j < out; ++j) printf(",");
    puts("");
    for(size_t i=0;i<nr;++i) {
        measurement *r=&records[i];
        printf("%s,%zu,%s,%zu,%llu,%.6f,%.17g",r->phase,r->sample,names[r->kernel],r->repetitions,
               (unsigned long long)r->elapsed,(double)r->elapsed/r->repetitions,r->checksum);
        for (size_t j = 0; j < out; ++j) printf(",%.9g", (double)r->output[j]);
        puts("");
    }
    free(outputs);free(records);free(storage);return 0;
}
