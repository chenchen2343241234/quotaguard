
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

API void groups(int n, int width, int g, const int32_t* order,
                const int32_t* nn, int32_t* labels) {
    std::fill(labels, labels+n, -1);
    int fallback=0, label=0;
    for(int i=0;i<n;i++) {
        int u=order[i]; if(labels[u]>=0) continue;
        labels[u]=label; int count=1;
        for(int j=0;j<width && count<g;j++) {
            int v=nn[u*width+j];
            if(v>=0 && labels[v]<0) {labels[v]=label; count++;}
        }
        while(count<g && fallback<n) {
            int v=order[fallback++];
            if(labels[v]<0) {labels[v]=label; count++;}
        }
        label++;
    }
}

API int matching(int n, int m, int smax, const int32_t* pairs,
                 const int32_t* order, const int32_t* vertex_group,
                 const int32_t* sizes, int32_t* mate) {
    std::fill(mate, mate+n, -1); int count=0;
    for(int j=0;j<m;j++) {
        int i=order[j], a=vertex_group[pairs[2*i]], b=vertex_group[pairs[2*i+1]];
        if(a!=b && mate[a]<0 && mate[b]<0 && sizes[a]+sizes[b]<=smax) {
            mate[a]=b; mate[b]=a; count++;
        }
    }
    int previous=-1;
    for(int a=0;a<n;a++) if(mate[a]<0) {
        if(previous>=0 && sizes[previous]+sizes[a]<=smax) {
            mate[a]=previous; mate[previous]=a; previous=-1; count++;
        } else previous=a;
    }
    return count;
}

API void assign(int m, const int32_t* pairs, const int32_t* leaf,
                const int32_t* parent, int32_t* bucket, int leaves_only) {
    for(int i=0;i<m;i++) {
        if(bucket[i]>=0) continue;
        int a=leaf[pairs[2*i]], b=leaf[pairs[2*i+1]];
        if(leaves_only) {if(a==b) bucket[i]=a; continue;}
        
        while(a!=b && a>=0 && b>=0) {
            if(a<b) a=parent[a]; else b=parent[b];
        }
        if(a==b && a>=0) bucket[i]=a;
    }
}

API void pb(int m, int nodes, int k, const int32_t* bucket,
            const double* probabilities, double* values) {
    int stride=k+1;
    std::vector<double> counts(nodes*stride,0.0);
    for(int t=0;t<nodes;t++) counts[t*stride]=1.0;
    for(int i=0;i<m;i++) {
        int b=bucket[i]; if(b<0) continue;
        double p=probabilities[i]; double* row=&counts[b*stride];
        for(int j=k;j>=1;j--) row[j]=row[j]*(1-p)+row[j-1]*p;
        row[0]*=(1-p);
    }
    for(int b=0;b<nodes;b++) {
        double cumulative=0, f=0; values[b*stride]=0;
        for(int q=1;q<=k;q++) {
            cumulative+=counts[b*stride+q-1];
            f+=std::max(0.0,1.0-cumulative); values[b*stride+q]=f;
        }
    }
}

API double quota_dp(int nr, int stride, const int32_t* roots,
                     const int32_t* budgets, const int32_t* capacities,
                     const int32_t* left, const int32_t* right,
                     const int32_t* counts, const double* f, int32_t* quotas) {
    double total=0;
    for(int ri=0;ri<nr;ri++) {
        std::vector<int> nodes, stack(1,roots[ri]);
        while(!stack.empty()) {
            int t=stack.back(); stack.pop_back(); nodes.push_back(t);
            if(left[t]>=0) {stack.push_back(left[t]); stack.push_back(right[t]);}
        }
        std::sort(nodes.begin(),nodes.end()); int m=nodes.size();
        if(m>21 || budgets[ri]>7) return std::numeric_limits<double>::quiet_NaN();
        int kmax=budgets[ri], cap=capacities[ri], cs=cap+1, ks=(kmax+1)*cs;
        std::vector<double> a(m*ks,-std::numeric_limits<double>::infinity());
        std::vector<uint64_t> sig(m*ks,0);
        auto local=[&](int t){return int(std::lower_bound(nodes.begin(),nodes.end(),t)-nodes.begin());};
        auto idx=[&](int t,int k,int c){return t*ks+k*cs+c;};
        for(int j=0;j<m;j++) {
            int t=nodes[j], l=left[t]<0?-1:local(left[t]), r=right[t]<0?-1:local(right[t]);
            for(int k=0;k<=kmax;k++) for(int q=0;q<=std::min({k,counts[t],cap});q++) {
                uint64_t qs=uint64_t(q)<<(3*(m-1-j));
                if(l<0) {a[idx(j,k,q)]=f[t*stride+q]; sig[idx(j,k,q)]=qs; continue;}
                for(int cl=0;cl<=cap-q;cl++) {
                    int li=idx(l,k-q,cl); if(!std::isfinite(a[li])) continue;
                    for(int cr=0;cr<=cap-q-cl;cr++) {
                        int rr=idx(r,k-q,cr); if(!std::isfinite(a[rr])) continue;
                        int out=idx(j,k,q+cl+cr);
                        double v=f[t*stride+q]+a[li]+a[rr];
                        uint64_t s=qs|sig[li]|sig[rr];
                        if(v>a[out] || (v==a[out] && s<sig[out])) {a[out]=v; sig[out]=s;}
                    }
                }
            }
        }
        int best=idx(m-1,kmax,0);
        for(int c=1;c<=cap;c++) if(a[idx(m-1,kmax,c)]>a[best]) best=idx(m-1,kmax,c);
        total+=a[best];
        for(int j=0;j<m;j++) quotas[nodes[j]]=(sig[best]>>(3*(m-1-j)))&7;
    }
    return total;
}

API void bmatch(int m, int n, int k, int cap, const int32_t* pairs,
                const int32_t* order, uint8_t* accepted) {
    std::vector<int> degree(n,0); int used=0;
    for(int j=0;j<m && used<cap;j++) {
        int i=order[j], a=pairs[2*i], b=pairs[2*i+1];
        if(degree[a]<k && degree[b]<k) {accepted[i]=1; degree[a]++; degree[b]++; used++;}
    }
}
