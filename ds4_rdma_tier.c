/* Remote-RAM expert tier over one-sided RDMA READ. See ds4_rdma_tier.h.
 *
 * Shape: two legs, one per NIC, each a plain RC QP to a peer that has
 * registered the model's leading bytes and then gone passive. A span is split
 * in half, one half per leg, both in flight at once. T0 established that the
 * halves only aggregate when the two legs land on DIFFERENT REMOTE devices --
 * two QPs on one remote device split that device 50/50 and buy nothing -- so
 * the remote GUIDs are compared at startup and the second leg is dropped if
 * they match.
 *
 * ds4 fetches from up to 32 threads at once, so connections are pooled: a
 * thread takes a pair, uses it, returns it. A shared QP would serialise the
 * fetch path and give back exactly what the striping won.
 */

#include "ds4_rdma_tier.h"

#include <errno.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#if defined(__linux__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#include <pthread.h>
#define DS4_RT_HAVE_VERBS 1
#endif
#endif

#ifndef DS4_RT_HAVE_VERBS

int ds4_rdma_tier_open(int model_fd, uint64_t model_bytes) {
    (void)model_fd; (void)model_bytes;
    if (getenv("DS4_EXPERT_TIER_RDMA"))
        fprintf(stderr, "ds4: DS4_EXPERT_TIER_RDMA set but this build has no verbs headers\n");
    return 0;
}
int ds4_rdma_tier_read(void *d, uint64_t o, uint64_t b) { (void)d; (void)o; (void)b; return 0; }
uint64_t ds4_rdma_tier_bytes(void) { return 0; }
void ds4_rdma_tier_stats(uint64_t *r, uint64_t *b, uint64_t *n) {
    if (r) *r = 0; if (b) *b = 0; if (n) *n = 0;
}
void ds4_rdma_tier_promote(const void *s, uint64_t o, uint64_t b) { (void)s; (void)o; (void)b; }
void ds4_rdma_tier_promote_stats(uint64_t *p, uint64_t *h, uint64_t *k, uint32_t *n) {
    if (p) *p = 0; if (h) *h = 0; if (k) *k = 0; if (n) *n = 0;
}
uint64_t ds4_rdma_tier_promote_corrupt(void) { return 0; }
void ds4_rdma_tier_close(void) { }

#else /* DS4_RT_HAVE_VERBS */

#define RT_MAX_LEGS   2
#define RT_MAX_CONNS  64
#define RT_MAX_MRC    96
#define RT_GID_DEFAULT 3
#define RT_POLL_TIMEOUT_NS 2000000000ull   /* 2 s: a healthy span takes ~0.4 ms */

/* Wire handshake -- must stay byte-identical to tests/rdma_t1/t1_common.h. */
#define RT_HS_MAGIC   0x54315253u
#define RT_HS_VERSION 3u
typedef struct {
    uint32_t magic, version, qpn, psn, lid, rkey;
    uint8_t  gid[16];
    uint64_t addr, len, seed, node_guid;
    char     dev[64];
    uint64_t cache_off;      /* v3: promotion slots start here in the region */
    uint64_t slot_bytes;
    uint32_t n_slots;
    uint32_t _pad;
} rt_hs;

#define RT_CMD_LOAD 1u
#define RT_CMD_BYE  2u
typedef struct { uint32_t op, slot; uint64_t offset, bytes; } rt_cmd;
typedef struct { uint32_t op, status; } rt_rsp;

/* Promotion (plan T4). The prefix covers the model's leading bytes; anything
 * past it is served by local disk unless it has been PROMOTED into one of the
 * peer's slots. Two ways to fill a slot:
 *   D1  this side RDMA-WRITEs the bytes it just read from disk.
 *   D2  this side asks the peer to pread them from its own copy of the model.
 * D1 spends the (otherwise idle) reverse direction of the fabric; D2 spends the
 * peer's (otherwise idle) NVMe and costs this box nothing but a 24-byte message.
 *
 * The DIRECTORY LIVES HERE, not on the peer, and that is deliberate. If the
 * peer owned slot residency it could evict and reuse a slot underneath an
 * in-flight one-sided READ, and this side -- having no expectation of the
 * bytes -- could not tell. Owning the directory means a slot is never read
 * after being reassigned. */
#define RT_PROMOTE_OFF 0
#define RT_PROMOTE_D1  1
#define RT_PROMOTE_D2  2

typedef struct {
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *,
                                struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
    void *handle;
} rt_api;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd      *pd;
    union ibv_gid       gid;
    enum ibv_mtu        mtu;
    char                name[64];
    char                host[64];      /* peer address for this leg */
    int                 port;
} rt_dev;

typedef struct {
    struct ibv_cq *cq[RT_MAX_LEGS];
    struct ibv_qp *qp[RT_MAX_LEGS];
    uint64_t       raddr[RT_MAX_LEGS];
    uint32_t       rkey[RT_MAX_LEGS];
    int            sock[RT_MAX_LEGS];
} rt_conn;

typedef struct { void *base; size_t len; struct ibv_mr *mr[RT_MAX_LEGS]; } rt_mrc;

static struct {
    rt_api   api;
    rt_dev   dev[RT_MAX_LEGS];
    int      n_legs;
    rt_conn  conn[RT_MAX_CONNS];
    int      n_conns;
    unsigned char busy[RT_MAX_CONNS];
    pthread_mutex_t lock;
    pthread_cond_t  cv;
    rt_mrc   mrc[RT_MAX_MRC];
    int      n_mrc;
    pthread_mutex_t mrlock;
    uint64_t region_bytes;          /* prefix only: what maps 1:1 to the model */
    uint64_t cache_off, slot_bytes;
    uint32_t n_slots;
    uint64_t remote_guid[RT_MAX_LEGS];
    /* promotion */
    int      promote_mode;
    uint32_t min_seen;
    struct rt_slot { uint64_t offset; uint32_t bytes; uint32_t valid; uint64_t stamp; } *slot;
    uint64_t stamp;
    pthread_mutex_t dirlock;
    /* admission: how many times an uncovered offset has been read from disk */
    struct { uint64_t offset; uint32_t seen; } seen_tab[4096];
    /* async promotion queue -- the fetch path must never wait on the fabric */
    struct rt_pq { uint64_t offset; uint32_t bytes; int wbuf; uint64_t sig; } *pq;
    int      pq_head, pq_tail, pq_cap;
    pthread_mutex_t pqlock;
    pthread_cond_t  pqcv;
    pthread_t worker;
    int      worker_live, stopping;
    /* D1 write staging: the fetch buffer is reused immediately, so the bytes
     * must be copied somewhere stable for the duration of the WRITE. */
    void    *wbuf; struct ibv_mr *wbuf_mr[RT_MAX_LEGS];
    int      n_wbuf; unsigned char *wbuf_busy;
    pthread_mutex_t wlock;
    uint64_t promoted, promote_skipped, promote_hits, promote_corrupt;
    void *vbuf; struct ibv_mr *vbuf_mr[RT_MAX_LEGS];   /* worker-only read-back */
    int   verify_promotions;
    int      gid_index;
    int      active;
    uint64_t reads, rbytes, ns;
} G;

/* DS4_EXPERT_TIER_RDMA_DEBUG=1 traces setup. Connection problems here are
 * remote and silent by nature -- the alternative is guessing at a hang. */
static int rt_dbg_on(void) {
    static int on = -1;
    if (on < 0) on = getenv("DS4_EXPERT_TIER_RDMA_DEBUG") != NULL;
    return on;
}
#define rt_dbg(...) do { if (rt_dbg_on()) { \
    fprintf(stderr, "ds4-rdma: " __VA_ARGS__); fflush(stderr); } } while (0)

static uint64_t rt_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* One-way trip to inactive. Called from the read path on any hard failure:
 * a broken QP would otherwise be retried on every span for the rest of the
 * run, and the model file already serves everything correctly. */
static void rt_disable(const char *why) {
    if (G.active) {
        G.active = 0;
        fprintf(stderr, "ds4: rdma expert tier disabled (%s); the model file "
                        "serves everything from here\n", why);
    }
}

static int rt_load_api(void) {
    if (G.api.handle) return 1;
    void *h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define RT_SYM(f, n) do { \
        G.api.f = (__typeof__(G.api.f))dlsym(h, n); \
        if (!G.api.f) { dlclose(h); memset(&G.api, 0, sizeof G.api); return 0; } \
    } while (0)
    RT_SYM(get_device_list, "ibv_get_device_list");
    RT_SYM(free_device_list, "ibv_free_device_list");
    RT_SYM(get_device_name, "ibv_get_device_name");
    RT_SYM(open_device, "ibv_open_device");
    RT_SYM(close_device, "ibv_close_device");
    RT_SYM(query_port, "ibv_query_port");
    RT_SYM(query_gid, "ibv_query_gid");
    RT_SYM(alloc_pd, "ibv_alloc_pd");
    RT_SYM(dealloc_pd, "ibv_dealloc_pd");
    RT_SYM(reg_mr, "ibv_reg_mr");
    RT_SYM(dereg_mr, "ibv_dereg_mr");
    RT_SYM(create_cq, "ibv_create_cq");
    RT_SYM(destroy_cq, "ibv_destroy_cq");
    RT_SYM(create_qp, "ibv_create_qp");
    RT_SYM(destroy_qp, "ibv_destroy_qp");
    RT_SYM(modify_qp, "ibv_modify_qp");
#undef RT_SYM
    G.api.handle = h;
    return 1;
}

/* ---- tcp handshake ---- */

static int rt_send_all(int fd, const void *p, size_t n) {
    const char *c = (const char *)p;
    while (n) {
        ssize_t k = send(fd, c, n, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}
static int rt_recv_all(int fd, void *p, size_t n) {
    char *c = (char *)p;
    while (n) {
        ssize_t k = recv(fd, c, n, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}
static int rt_connect_tcp(const char *host, int port) {
    char svc[16];
    snprintf(svc, sizeof svc, "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, svc, &hints, &res) != 0) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, 0);
    if (fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) != 0) { close(fd); fd = -1; }
    freeaddrinfo(res);
    if (fd >= 0) { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); }
    return fd;
}

/* ---- device selection ----
 * Pick, for each leg, the local device whose RoCEv2 GID is an IPv4 on the same
 * /24 as that leg's peer address. On this fabric 10.99.0.x and 10.99.2.x are
 * distinct /30s on one segment, so the /24 is unambiguous and needs no routing
 * table parsing. */
static int rt_gid_ipv4(const union ibv_gid *g, uint32_t *out) {
    uint64_t hi;
    uint16_t mid, tag;
    memcpy(&hi, &g->raw[0], 8);
    memcpy(&mid, &g->raw[8], 2);
    memcpy(&tag, &g->raw[10], 2);
    if (hi != 0 || mid != 0 || tag != 0xffff) return 0;
    memcpy(out, &g->raw[12], 4);
    return 1;
}

static int rt_open_dev(rt_dev *d) {
    struct in_addr want;
    if (inet_pton(AF_INET, d->host, &want) != 1) {
        fprintf(stderr, "ds4: rdma tier: '%s' is not an IPv4 address\n", d->host);
        return 0;
    }
    int n = 0;
    struct ibv_device **list = G.api.get_device_list(&n);
    if (!list) return 0;
    int ok = 0;
    for (int i = 0; i < n && !ok; i++) {
        struct ibv_context *ctx = G.api.open_device(list[i]);
        if (!ctx) continue;
        struct ibv_port_attr port;
        union ibv_gid gid;
        uint32_t ip = 0;
        if (G.api.query_port(ctx, 1, &port) == 0 && port.state == IBV_PORT_ACTIVE &&
            G.api.query_gid(ctx, 1, G.gid_index, &gid) == 0 && rt_gid_ipv4(&gid, &ip) &&
            (ip & htonl(0xffffff00u)) == (want.s_addr & htonl(0xffffff00u))) {
            d->ctx = ctx;
            d->gid = gid;
            d->mtu = port.active_mtu;
            snprintf(d->name, sizeof d->name, "%s", G.api.get_device_name(list[i]));
            d->pd = G.api.alloc_pd(ctx);
            if (d->pd) ok = 1;
            else { G.api.close_device(ctx); d->ctx = NULL; }
        } else {
            G.api.close_device(ctx);
        }
    }
    G.api.free_device_list(list);
    if (!ok)
        fprintf(stderr, "ds4: rdma tier: no ACTIVE device with a RoCEv2 GID on %s's /24\n",
                d->host);
    return ok;
}

/* ---- one connection leg ----
 * The peer accepts BOTH legs of a pair before handshaking either, so every
 * socket must be connected before any handshake starts. Doing connect and
 * handshake together per leg deadlocks: this side waits for leg A's handshake
 * while the peer is still blocked in accept() for leg B. */
static int rt_connect_leg_socket(rt_conn *c, int leg) {
    rt_dev *d = &G.dev[leg];
    c->sock[leg] = rt_connect_tcp(d->host, d->port);
    if (c->sock[leg] < 0) {
        fprintf(stderr, "ds4: rdma tier: cannot reach %s:%d\n", d->host, d->port);
        return 0;
    }
    return 1;
}

static int rt_open_leg(rt_conn *c, int leg) {
    rt_dev *d = &G.dev[leg];
    rt_dbg("leg %d: handshake start (%s:%d via %s)\n", leg, d->host, d->port, d->name);
    c->cq[leg] = G.api.create_cq(d->ctx, 16, NULL, NULL, 0);
    if (!c->cq[leg]) return 0;
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.send_cq = c->cq[leg]; qa.recv_cq = c->cq[leg];
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = 16; qa.cap.max_recv_wr = 16;
    qa.cap.max_send_sge = 1; qa.cap.max_recv_sge = 1;
    c->qp[leg] = G.api.create_qp(d->pd, &qa);
    if (!c->qp[leg]) return 0;

    const uint32_t psn = 0x0f0f0fu + (uint32_t)leg;
    rt_hs peer, mine;
    rt_dbg("leg %d: waiting for peer handshake (%zu bytes)\n", leg, sizeof peer);
    if (rt_recv_all(c->sock[leg], &peer, sizeof peer)) return 0;
    rt_dbg("leg %d: got peer hs qpn=%u rkey=0x%x len=%llu dev=%s\n", leg,
           peer.qpn, peer.rkey, (unsigned long long)peer.len, peer.dev);
    if (peer.magic != RT_HS_MAGIC || peer.version != RT_HS_VERSION) {
        fprintf(stderr, "ds4: rdma tier: handshake mismatch from %s:%d -- the peer "
                        "server is a different build\n", d->host, d->port);
        return 0;
    }
    memset(&mine, 0, sizeof mine);
    mine.magic = RT_HS_MAGIC; mine.version = RT_HS_VERSION;
    mine.qpn = c->qp[leg]->qp_num; mine.psn = psn;
    memcpy(mine.gid, d->gid.raw, 16);
    snprintf(mine.dev, sizeof mine.dev, "%s", d->name);
    if (rt_send_all(c->sock[leg], &mine, sizeof mine)) return 0;

    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0; a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    if (G.api.modify_qp(c->qp[leg], &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return 0;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = d->mtu;
    a.dest_qp_num = peer.qpn;
    a.rq_psn = peer.psn;
    a.max_dest_rd_atomic = 16;
    a.min_rnr_timer = 12;
    a.ah_attr.dlid = (uint16_t)peer.lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)G.gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (G.api.modify_qp(c->qp[leg], &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                        IBV_QP_MAX_DEST_RD_ATOMIC |
                                        IBV_QP_MIN_RNR_TIMER)) return 0;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = psn;
    a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7;
    a.max_rd_atomic = 16;
    if (G.api.modify_qp(c->qp[leg], &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                        IBV_QP_MAX_QP_RD_ATOMIC)) return 0;

    rt_dbg("leg %d: RTS\n", leg);

    c->raddr[leg] = peer.addr;
    c->rkey[leg]  = peer.rkey;
    if (G.region_bytes == 0 || peer.len < G.region_bytes) G.region_bytes = peer.len;
    G.remote_guid[leg] = peer.node_guid;
    G.cache_off = peer.cache_off;
    G.slot_bytes = peer.slot_bytes;
    G.n_slots = peer.n_slots;
    return 1;
}

/* The peer sends its go byte on every leg only once the WHOLE pair is wired,
 * so this waits in its own pass. Folding it into rt_open_leg() deadlocks: this
 * side would block for leg 0's go while the peer is still waiting for leg 1's
 * handshake, which this side has not sent yet. */
static int rt_wait_go(rt_conn *c, int leg) {
    uint8_t go = 0;
    if (rt_recv_all(c->sock[leg], &go, 1) || go != 1) return 0;
    rt_dbg("leg %d: ready\n", leg);
    return 1;
}

/* ---- pool ---- */
static rt_conn *rt_acquire(int *idx) {
    pthread_mutex_lock(&G.lock);
    for (;;) {
        for (int i = 0; i < G.n_conns; i++) {
            if (!G.busy[i]) {
                G.busy[i] = 1;
                pthread_mutex_unlock(&G.lock);
                *idx = i;
                return &G.conn[i];
            }
        }
        pthread_cond_wait(&G.cv, &G.lock);
    }
}
static void rt_release(int idx) {
    pthread_mutex_lock(&G.lock);
    G.busy[idx] = 0;
    pthread_cond_signal(&G.cv);
    pthread_mutex_unlock(&G.lock);
}

/* ---- memory-region cache ----
 * ds4's staging buffers are a small set of long-lived cudaHostAlloc'd blocks,
 * so registering per read would dominate the transfer. Keyed on (base, len):
 * a grown buffer reuses the pointer but changes the length, which misses and
 * re-registers rather than handing back an MR over freed memory. */
static int rt_mr_get(void *base, size_t len, struct ibv_mr **out) {
    pthread_mutex_lock(&G.mrlock);
    for (int i = 0; i < G.n_mrc; i++) {
        if (G.mrc[i].base == base && G.mrc[i].len >= len) {
            for (int l = 0; l < G.n_legs; l++) out[l] = G.mrc[i].mr[l];
            pthread_mutex_unlock(&G.mrlock);
            return 1;
        }
    }
    if (G.n_mrc >= RT_MAX_MRC) { pthread_mutex_unlock(&G.mrlock); return 0; }
    rt_mrc *e = &G.mrc[G.n_mrc];
    memset(e, 0, sizeof *e);
    for (int l = 0; l < G.n_legs; l++) {
        e->mr[l] = G.api.reg_mr(G.dev[l].pd, base, len, IBV_ACCESS_LOCAL_WRITE);
        if (!e->mr[l]) {
            for (int k = 0; k < l; k++) G.api.dereg_mr(e->mr[k]);
            pthread_mutex_unlock(&G.mrlock);
            return 0;
        }
    }
    e->base = base; e->len = len;
    G.n_mrc++;
    for (int l = 0; l < G.n_legs; l++) out[l] = e->mr[l];
    pthread_mutex_unlock(&G.mrlock);
    return 1;
}

/* ---- the read itself ---- */
static int rt_post(rt_conn *c, int leg, void *dst, uint32_t lkey,
                   uint64_t raddr, uint64_t bytes) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof sge);
    sge.addr = (uint64_t)(uintptr_t)dst;
    sge.length = (uint32_t)bytes;
    sge.lkey = lkey;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = (uint64_t)leg;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = c->rkey[leg];
    return ibv_post_send(c->qp[leg], &wr, &bad) == 0;   /* inline over ctx->ops */
}

static int rt_wait(rt_conn *c, int leg, uint64_t deadline) {
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(c->cq[leg], 1, &wc);
        if (n < 0) return 0;
        if (n > 0) return wc.status == IBV_WC_SUCCESS;
        if (rt_now_ns() > deadline) return 0;
    }
}

static int rt_read_raw(rt_conn *c, void *dst, struct ibv_mr **mr,
                       uint64_t off, uint64_t bytes) {
    const uint64_t deadline = rt_now_ns() + RT_POLL_TIMEOUT_NS;
    if (G.n_legs == 1) {
        if (!rt_post(c, 0, dst, mr[0]->lkey, c->raddr[0] + off, bytes)) return 0;
        return rt_wait(c, 0, deadline);
    }
    const uint64_t half = bytes / 2;
    const uint64_t rest = bytes - half;
    if (!rt_post(c, 0, dst, mr[0]->lkey, c->raddr[0] + off, half)) return 0;
    if (!rt_post(c, 1, (char *)dst + half, mr[1]->lkey,
                 c->raddr[1] + off + half, rest)) {
        (void)rt_wait(c, 0, deadline);   /* drain leg 0 before giving the pair back */
        return 0;
    }
    const int a = rt_wait(c, 0, deadline);
    const int b = rt_wait(c, 1, deadline);
    return a && b;
}

/* ---- promotion: directory, admission, and the worker ---- */

/* Sampled signature. A full hash of every promoted span would sit on the fetch
 * path, which is the one place that must stay cheap; 16 spread samples cost a
 * handful of loads and catch the failure modes that actually occur here -- an
 * unfilled slot, a stale slot, bytes from the wrong offset. It is a corruption
 * detector, not a checksum: a single flipped bit can slip through. */
static uint64_t rt_sig(const void *p, uint64_t bytes) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 0x9E3779B97F4A7C15ull ^ bytes;
    for (int i = 0; i < 16; i++) {
        uint64_t off = (bytes > 8) ? (uint64_t)i * ((bytes - 8) / 15u) : 0;
        if (off + 8 > bytes) off = bytes - 8;
        uint64_t w;
        memcpy(&w, b + off, 8);
        h ^= w + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    }
    return h;
}

static int rt_post_write(rt_conn *c, int leg, const void *src, uint32_t lkey,
                         uint64_t raddr, uint64_t bytes) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof sge);
    sge.addr = (uint64_t)(uintptr_t)src;
    sge.length = (uint32_t)bytes;
    sge.lkey = lkey;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = (uint64_t)leg;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = c->rkey[leg];
    return ibv_post_send(c->qp[leg], &wr, &bad) == 0;
}

/* Directory is a linear scan over slots. n_slots is in the thousands and the
 * promotion rate measured in T3 is ~80 spans/s, so this costs microseconds in
 * a path that is already paying hundreds. Kept simple on purpose. */
static int rt_dir_find_locked(uint64_t offset, uint32_t bytes, uint32_t *out) {
    for (uint32_t i = 0; i < G.n_slots; i++)
        if (G.slot[i].valid && G.slot[i].offset == offset && G.slot[i].bytes == bytes) {
            G.slot[i].stamp = ++G.stamp;
            *out = i;
            return 1;
        }
    return 0;
}

static uint32_t rt_dir_victim_locked(void) {
    uint32_t best = 0;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < G.n_slots; i++) {
        if (!G.slot[i].valid) return i;          /* free slot first */
        if (G.slot[i].stamp < oldest) { oldest = G.slot[i].stamp; best = i; }
    }
    return best;
}

/* seen >= min_seen before promoting. 61.3% of everything loaded is never reused
 * even once, so promoting on first sight spends slots on spans that will never
 * be read again. D2 can afford a lower threshold than D1 because its promotion
 * costs the peer's idle disk rather than this box's memory bandwidth. */
static int rt_admit(uint64_t offset) {
    const size_t n = sizeof G.seen_tab / sizeof G.seen_tab[0];
    size_t h = (size_t)((offset >> 12) % n);
    if (G.seen_tab[h].offset != offset) { G.seen_tab[h].offset = offset; G.seen_tab[h].seen = 1; }
    else if (G.seen_tab[h].seen < 1000000u) G.seen_tab[h].seen++;
    return G.seen_tab[h].seen >= G.min_seen;
}

static int rt_wbuf_take(void) {
    pthread_mutex_lock(&G.wlock);
    for (int i = 0; i < G.n_wbuf; i++)
        if (!G.wbuf_busy[i]) { G.wbuf_busy[i] = 1; pthread_mutex_unlock(&G.wlock); return i; }
    pthread_mutex_unlock(&G.wlock);
    return -1;
}
static void rt_wbuf_put(int i) {
    if (i < 0) return;
    pthread_mutex_lock(&G.wlock);
    G.wbuf_busy[i] = 0;
    pthread_mutex_unlock(&G.wlock);
}

static void *rt_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&G.pqlock);
        while (G.pq_head == G.pq_tail && !G.stopping)
            pthread_cond_wait(&G.pqcv, &G.pqlock);
        if (G.stopping && G.pq_head == G.pq_tail) { pthread_mutex_unlock(&G.pqlock); break; }
        struct rt_pq job = G.pq[G.pq_head];
        G.pq_head = (G.pq_head + 1) % G.pq_cap;
        pthread_mutex_unlock(&G.pqlock);

        if (!G.active) { rt_wbuf_put(job.wbuf); continue; }

        pthread_mutex_lock(&G.dirlock);
        uint32_t dup;
        if (rt_dir_find_locked(job.offset, job.bytes, &dup)) {
            pthread_mutex_unlock(&G.dirlock);
            rt_wbuf_put(job.wbuf);
            continue;
        }
        const uint32_t slot = rt_dir_victim_locked();
        G.slot[slot].valid = 0;              /* nothing reads it while it loads */
        G.slot[slot].offset = job.offset;
        G.slot[slot].bytes = job.bytes;
        pthread_mutex_unlock(&G.dirlock);

        const uint64_t slot_off = G.cache_off + (uint64_t)slot * G.slot_bytes;
        int ok = 0;
        int idx = 0;
        rt_conn *c = rt_acquire(&idx);
        if (G.promote_mode == RT_PROMOTE_D1) {
            const char *src = (const char *)G.wbuf + (size_t)job.wbuf * G.slot_bytes;
            const uint64_t deadline = rt_now_ns() + RT_POLL_TIMEOUT_NS;
            if (G.n_legs == 1) {
                ok = rt_post_write(c, 0, src, G.wbuf_mr[0]->lkey,
                                   c->raddr[0] + slot_off, job.bytes) &&
                     rt_wait(c, 0, deadline);
            } else {
                const uint64_t half = job.bytes / 2, rest = job.bytes - half;
                if (rt_post_write(c, 0, src, G.wbuf_mr[0]->lkey,
                                  c->raddr[0] + slot_off, half) &&
                    rt_post_write(c, 1, src + half, G.wbuf_mr[1]->lkey,
                                  c->raddr[1] + slot_off + half, rest)) {
                    const int a = rt_wait(c, 0, deadline);
                    const int b = rt_wait(c, 1, deadline);
                    ok = a && b;
                }
            }
        } else {                               /* D2: the peer loads it itself */
            rt_cmd cmd = { RT_CMD_LOAD, slot, job.offset, job.bytes };
            rt_rsp rsp;
            pthread_mutex_lock(&G.wlock);      /* the control socket is shared */
            ok = !rt_send_all(G.conn[0].sock[0], &cmd, sizeof cmd) &&
                 !rt_recv_all(G.conn[0].sock[0], &rsp, sizeof rsp) &&
                 rsp.status == 0u;
            pthread_mutex_unlock(&G.wlock);
        }
        rt_release(idx);
        rt_wbuf_put(job.wbuf);

        /* Read it back before publishing. Promotion is off the critical path,
         * so this costs nothing that matters, and it is the difference between
         * a corrupted slot being detected and it being served as model weights. */
        if (ok && G.verify_promotions && G.vbuf) {
            int vidx = 0;
            rt_conn *vc = rt_acquire(&vidx);
            const int got = rt_read_raw(vc, G.vbuf, G.vbuf_mr, slot_off, job.bytes);
            rt_release(vidx);
            if (!got || rt_sig(G.vbuf, job.bytes) != job.sig) {
                ok = 0;
                if (__sync_fetch_and_add(&G.promote_corrupt, 1) == 0)
                    fprintf(stderr,
                        "ds4: rdma promotion READ-BACK MISMATCH at model offset %llu "
                        "-- discarding that slot. The model file still serves it, so "
                        "output is unaffected; promotion is suspect on this peer.\n",
                        (unsigned long long)job.offset);
            }
        }
        pthread_mutex_lock(&G.dirlock);
        if (ok) { G.slot[slot].valid = 1; G.slot[slot].stamp = ++G.stamp; G.promoted++; }
        else    { G.slot[slot].valid = 0; G.promote_skipped++; }
        pthread_mutex_unlock(&G.dirlock);
    }
    return NULL;
}

/* Called from the fetch path right after a span was served by local disk.
 * Never blocks: it copies (D1) or just records (D2) and returns. */
void ds4_rdma_tier_promote(const void *src, uint64_t offset, uint64_t bytes) {
    if (!G.active || G.promote_mode == RT_PROMOTE_OFF || !G.n_slots) return;
    if (!bytes || bytes > G.slot_bytes) return;
    if (offset + bytes <= G.region_bytes) return;   /* already in the prefix */
    if (!rt_admit(offset)) return;

    pthread_mutex_lock(&G.dirlock);
    uint32_t dup;
    const int have = rt_dir_find_locked(offset, (uint32_t)bytes, &dup);
    pthread_mutex_unlock(&G.dirlock);
    if (have) return;

    const uint64_t sig = G.verify_promotions ? rt_sig(src, bytes) : 0;
    int wb = -1;
    if (G.promote_mode == RT_PROMOTE_D1) {
        wb = rt_wbuf_take();
        if (wb < 0) { __sync_fetch_and_add(&G.promote_skipped, 1); return; }
        memcpy((char *)G.wbuf + (size_t)wb * G.slot_bytes, src, (size_t)bytes);
    }
    pthread_mutex_lock(&G.pqlock);
    const int next = (G.pq_tail + 1) % G.pq_cap;
    if (next == G.pq_head) {                    /* queue full: drop, never block */
        pthread_mutex_unlock(&G.pqlock);
        rt_wbuf_put(wb);
        __sync_fetch_and_add(&G.promote_skipped, 1);
        return;
    }
    G.pq[G.pq_tail].offset = offset;
    G.pq[G.pq_tail].bytes = (uint32_t)bytes;
    G.pq[G.pq_tail].wbuf = wb;
    G.pq[G.pq_tail].sig = sig;
    G.pq_tail = next;
    pthread_cond_signal(&G.pqcv);
    pthread_mutex_unlock(&G.pqlock);
}

int ds4_rdma_tier_read(void *dst, uint64_t offset, uint64_t bytes) {
    if (!G.active || bytes == 0) return 0;
    uint64_t remote_off;
    int from_slot = 0;
    if (bytes <= G.region_bytes && offset <= G.region_bytes - bytes) {
        remote_off = offset;                      /* mirrored prefix */
    } else if (G.n_slots && bytes <= G.slot_bytes) {
        uint32_t slot;
        pthread_mutex_lock(&G.dirlock);
        const int hit = rt_dir_find_locked(offset, (uint32_t)bytes, &slot);
        pthread_mutex_unlock(&G.dirlock);
        if (!hit) return 0;
        remote_off = G.cache_off + (uint64_t)slot * G.slot_bytes;
        from_slot = 1;
    } else {
        return 0;
    }
    struct ibv_mr *mr[RT_MAX_LEGS];
    if (!rt_mr_get(dst, (size_t)bytes, mr)) return 0;
    int idx = 0;
    rt_conn *c = rt_acquire(&idx);
    const uint64_t t0 = rt_now_ns();
    const int ok = rt_read_raw(c, dst, mr, remote_off, bytes);
    if (ok && from_slot) __sync_fetch_and_add(&G.promote_hits, 1);
    const uint64_t dt = rt_now_ns() - t0;
    rt_release(idx);
    if (!ok) { rt_disable("a read did not complete"); return 0; }
    __sync_fetch_and_add(&G.reads, 1);
    __sync_fetch_and_add(&G.rbytes, bytes);
    __sync_fetch_and_add(&G.ns, dt);
    return 1;
}

uint64_t ds4_rdma_tier_bytes(void) { return G.active ? G.region_bytes : 0; }

void ds4_rdma_tier_stats(uint64_t *reads, uint64_t *bytes, uint64_t *ns) {
    if (reads) *reads = G.reads;
    if (bytes) *bytes = G.rbytes;
    if (ns) *ns = G.ns;
}

uint64_t ds4_rdma_tier_promote_corrupt(void) { return G.promote_corrupt; }

void ds4_rdma_tier_promote_stats(uint64_t *promoted, uint64_t *hits, uint64_t *skipped,
                                 uint32_t *slots) {
    if (promoted) *promoted = G.promoted;
    if (hits) *hits = G.promote_hits;
    if (skipped) *skipped = G.promote_skipped;
    if (slots) *slots = G.n_slots;
}

/* Sample the remote region against the real model file. A tier that is stale,
 * short or shifted would feed wrong weights, which is far worse than slow, so
 * this runs before the tier is ever used and refuses on any mismatch. The last
 * block is always probed: an over-large region is the likeliest mistake and
 * shows up only at the end. */
static int rt_verify(int model_fd, uint64_t model_bytes) {
    if (model_fd < 0 || G.region_bytes == 0) return 0;
    if (model_bytes && G.region_bytes > model_bytes) G.region_bytes = model_bytes;
    const uint64_t blk = 65536;
    if (G.region_bytes < blk) return 0;
    void *rbuf = NULL, *mbuf = NULL;
    if (posix_memalign(&rbuf, 4096, (size_t)blk) || posix_memalign(&mbuf, 4096, (size_t)blk)) {
        free(rbuf); free(mbuf); return 0;
    }
    struct ibv_mr *mr[RT_MAX_LEGS];
    memset(mr, 0, sizeof mr);
    int ok = 1;
    for (int l = 0; l < G.n_legs && ok; l++) {
        mr[l] = G.api.reg_mr(G.dev[l].pd, rbuf, (size_t)blk, IBV_ACCESS_LOCAL_WRITE);
        if (!mr[l]) ok = 0;
    }
    const uint64_t span = G.region_bytes;
    const uint64_t probes[] = { 0, span / 4, span / 2, (span / 4) * 3, span - blk };
    for (size_t i = 0; ok && i < sizeof probes / sizeof probes[0]; i++) {
        uint64_t off = probes[i] & ~4095ull;
        if (off + blk > span) continue;
        int idx = 0;
        rt_conn *c = rt_acquire(&idx);
        const int got = rt_read_raw(c, rbuf, mr, off, blk);
        rt_release(idx);
        if (!got) {
            fprintf(stderr, "ds4: rdma expert tier: probe read at %llu failed\n",
                    (unsigned long long)off);
            ok = 0; break;
        }
        uint64_t done = 0;
        while (done < blk) {
            ssize_t k = pread(model_fd, (char *)mbuf + done, (size_t)(blk - done),
                              (off_t)(off + done));
            if (k <= 0) { if (k < 0 && errno == EINTR) continue; ok = 0; break; }
            done += (uint64_t)k;
        }
        if (ok && memcmp(rbuf, mbuf, (size_t)blk) != 0) {
            fprintf(stderr,
                    "ds4: rdma expert tier does NOT match the model at offset %llu"
                    " -- refusing to use it (is the peer serving this same file?)\n",
                    (unsigned long long)off);
            ok = 0;
        }
    }
    for (int l = 0; l < G.n_legs; l++) if (mr[l]) G.api.dereg_mr(mr[l]);
    free(rbuf); free(mbuf);
    return ok;
}

int ds4_rdma_tier_open(int model_fd, uint64_t model_bytes) {
    const char *spec = getenv("DS4_EXPERT_TIER_RDMA");
    if (!spec || !spec[0]) return 0;
    if (G.active) return 1;

    memset(&G, 0, sizeof G);
    pthread_mutex_init(&G.lock, NULL);
    pthread_cond_init(&G.cv, NULL);
    pthread_mutex_init(&G.mrlock, NULL);
    G.gid_index = RT_GID_DEFAULT;
    { const char *g = getenv("DS4_EXPERT_TIER_RDMA_GID"); if (g && g[0]) G.gid_index = atoi(g); }

    /* "host:port[,host:port]" -- one entry per leg. */
    char buf[256];
    snprintf(buf, sizeof buf, "%s", spec);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok && G.n_legs < RT_MAX_LEGS;
         tok = strtok_r(NULL, ",", &save)) {
        char *colon = strrchr(tok, ':');
        if (!colon) {
            fprintf(stderr, "ds4: DS4_EXPERT_TIER_RDMA leg '%s' needs host:port\n", tok);
            return 0;
        }
        *colon = '\0';
        snprintf(G.dev[G.n_legs].host, sizeof G.dev[G.n_legs].host, "%s", tok);
        G.dev[G.n_legs].port = atoi(colon + 1);
        G.n_legs++;
    }
    if (G.n_legs == 0) return 0;
    if (!rt_load_api()) {
        fprintf(stderr, "ds4: DS4_EXPERT_TIER_RDMA set but libibverbs is not loadable\n");
        return 0;
    }
    for (int l = 0; l < G.n_legs; l++) {
        if (!rt_open_dev(&G.dev[l])) return 0;
        rt_dbg("leg %d: peer %s:%d via local device %s\n", l, G.dev[l].host,
               G.dev[l].port, G.dev[l].name);
    }

    int want = 8;
    { const char *q = getenv("DS4_EXPERT_TIER_RDMA_QPS"); if (q && q[0]) want = atoi(q); }
    if (want < 1) want = 1;
    if (want > RT_MAX_CONNS) want = RT_MAX_CONNS;

    for (int i = 0; i < want; i++) {
        rt_conn *c = &G.conn[i];
        for (int l = 0; l < RT_MAX_LEGS; l++) c->sock[l] = -1;
        int ok = 1;
        rt_dbg("pair %d: connecting sockets\n", i);
        for (int l = 0; l < G.n_legs && ok; l++) ok = rt_connect_leg_socket(c, l);
        for (int l = 0; l < G.n_legs && ok; l++) ok = rt_open_leg(c, l);
        for (int l = 0; l < G.n_legs && ok; l++) ok = rt_wait_go(c, l);
        if (!ok) {
            if (i == 0) {
                fprintf(stderr, "ds4: rdma expert tier: could not establish a "
                                "connection to the peer\n");
                return 0;
            }
            break;              /* fewer pairs than asked for is fine */
        }
        G.n_conns = i + 1;
    }

    /* T0's trap: two legs on one REMOTE device split that device 50/50 and
     * stripe to exactly nothing. One leg at full speed beats two that share. */
    if (G.n_legs == 2 && G.remote_guid[0] == G.remote_guid[1]) {
        fprintf(stderr,
                "ds4: rdma expert tier: both legs land on the same remote device "
                "(guid 0x%016llx); striping would buy nothing, using one leg\n",
                (unsigned long long)G.remote_guid[0]);
        G.n_legs = 1;
    }

    G.active = 1;
    if (!rt_verify(model_fd, model_bytes)) {
        G.active = 0;
        ds4_rdma_tier_close();
        return 0;
    }
    /* Promotion (T4). Off unless asked for, and it never affects correctness:
     * a slot that fails to fill simply stays invalid and the model file serves
     * that span, exactly as before. */
    {
        const char *pm = getenv("DS4_EXPERT_TIER_RDMA_PROMOTE");
        if (pm && G.n_slots && G.slot_bytes) {
            if (!strcmp(pm, "d1")) G.promote_mode = RT_PROMOTE_D1;
            else if (!strcmp(pm, "d2")) G.promote_mode = RT_PROMOTE_D2;
            else if (strcmp(pm, "off") != 0)
                fprintf(stderr, "ds4: DS4_EXPERT_TIER_RDMA_PROMOTE must be d1, d2 or off\n");
        } else if (pm && !G.n_slots) {
            fprintf(stderr, "ds4: promotion asked for but the peer advertises no slots "
                            "(start it with --cache-slots)\n");
        }
    }
    if (G.promote_mode != RT_PROMOTE_OFF) {
        G.min_seen = 2;
        { const char *m = getenv("DS4_EXPERT_TIER_RDMA_MIN_SEEN");
          if (m && m[0]) G.min_seen = (uint32_t)strtoul(m, NULL, 10); }
        if (G.min_seen < 1) G.min_seen = 1;
        pthread_mutex_init(&G.dirlock, NULL);
        pthread_mutex_init(&G.pqlock, NULL);
        pthread_mutex_init(&G.wlock, NULL);
        pthread_cond_init(&G.pqcv, NULL);
        G.slot = (struct rt_slot *)calloc(G.n_slots, sizeof *G.slot);
        G.pq_cap = 1024;
        G.pq = (struct rt_pq *)calloc((size_t)G.pq_cap, sizeof *G.pq);
        if (G.promote_mode == RT_PROMOTE_D1) {
            G.n_wbuf = 8;
            { const char *w = getenv("DS4_EXPERT_TIER_RDMA_WBUFS");
              if (w && w[0]) G.n_wbuf = atoi(w); }
            if (G.n_wbuf < 1) G.n_wbuf = 1;
            if (G.n_wbuf > 64) G.n_wbuf = 64;
            G.wbuf_busy = (unsigned char *)calloc((size_t)G.n_wbuf, 1);
            if (posix_memalign(&G.wbuf, 4096, (size_t)G.n_wbuf * G.slot_bytes) != 0)
                G.wbuf = NULL;
            for (int l = 0; G.wbuf && l < G.n_legs; l++) {
                G.wbuf_mr[l] = G.api.reg_mr(G.dev[l].pd, G.wbuf,
                                            (size_t)G.n_wbuf * G.slot_bytes,
                                            IBV_ACCESS_LOCAL_WRITE);
                if (!G.wbuf_mr[l]) G.wbuf = NULL;
            }
        }
        G.verify_promotions = 1;
        { const char *v = getenv("DS4_EXPERT_TIER_RDMA_VERIFY");
          if (v && (v[0] == '0' || v[0] == 'n')) G.verify_promotions = 0; }
        if (G.verify_promotions) {
            if (posix_memalign(&G.vbuf, 4096, (size_t)G.slot_bytes) != 0) G.vbuf = NULL;
            for (int l = 0; G.vbuf && l < G.n_legs; l++) {
                G.vbuf_mr[l] = G.api.reg_mr(G.dev[l].pd, G.vbuf, (size_t)G.slot_bytes,
                                            IBV_ACCESS_LOCAL_WRITE);
                if (!G.vbuf_mr[l]) G.vbuf = NULL;
            }
            if (!G.vbuf) {
                fprintf(stderr, "ds4: rdma promotion read-back buffer unavailable; "
                                "promotion disabled rather than run unverified\n");
                G.promote_mode = RT_PROMOTE_OFF;
            }
        }
        if (G.promote_mode != RT_PROMOTE_OFF &&
            (!G.slot || !G.pq || (G.promote_mode == RT_PROMOTE_D1 && !G.wbuf))) {
            fprintf(stderr, "ds4: rdma promotion setup failed; continuing read-only\n");
            G.promote_mode = RT_PROMOTE_OFF;
        } else if (pthread_create(&G.worker, NULL, rt_worker, NULL) != 0) {
            fprintf(stderr, "ds4: rdma promotion worker failed to start; read-only\n");
            G.promote_mode = RT_PROMOTE_OFF;
        } else {
            G.worker_live = 1;
            fprintf(stderr, "ds4: rdma promotion %s: %u slots x %.2f MiB (%.2f GiB), "
                            "admit at seen>=%u%s\n",
                    G.promote_mode == RT_PROMOTE_D1 ? "D1 (write span)"
                                                    : "D2 (peer loads from its disk)",
                    G.n_slots, (double)G.slot_bytes / 1048576.0,
                    (double)G.n_slots * (double)G.slot_bytes / 1073741824.0,
                    G.min_seen,
                    G.promote_mode == RT_PROMOTE_D1 ? "" : "");
        }
    }
    fprintf(stderr,
            "ds4: rdma expert tier active: %.2f GiB from %s, %d leg%s x %d pair%s, verified\n",
            (double)G.region_bytes / 1073741824.0, G.dev[0].host,
            G.n_legs, G.n_legs == 1 ? "" : "s",
            G.n_conns, G.n_conns == 1 ? "" : "s");
    return 1;
}

void ds4_rdma_tier_close(void) {
    if (G.worker_live) {
        pthread_mutex_lock(&G.pqlock);
        G.stopping = 1;
        pthread_cond_broadcast(&G.pqcv);
        pthread_mutex_unlock(&G.pqlock);
        pthread_join(G.worker, NULL);
        G.worker_live = 0;
    }
    for (int l = 0; l < G.n_legs; l++) {
        if (G.wbuf_mr[l]) { G.api.dereg_mr(G.wbuf_mr[l]); G.wbuf_mr[l] = NULL; }
        if (G.vbuf_mr[l]) { G.api.dereg_mr(G.vbuf_mr[l]); G.vbuf_mr[l] = NULL; }
    }
    free(G.wbuf); G.wbuf = NULL;
    free(G.vbuf); G.vbuf = NULL;
    free(G.wbuf_busy); G.wbuf_busy = NULL;
    free(G.slot); G.slot = NULL;
    free(G.pq); G.pq = NULL;
    for (int i = 0; i < G.n_conns; i++) {
        for (int l = 0; l < G.n_legs; l++) {
            if (G.conn[i].qp[l]) G.api.destroy_qp(G.conn[i].qp[l]);
            if (G.conn[i].cq[l]) G.api.destroy_cq(G.conn[i].cq[l]);
            if (G.conn[i].sock[l] >= 0) close(G.conn[i].sock[l]);
        }
    }
    for (int i = 0; i < G.n_mrc; i++)
        for (int l = 0; l < G.n_legs; l++)
            if (G.mrc[i].mr[l]) G.api.dereg_mr(G.mrc[i].mr[l]);
    for (int l = 0; l < G.n_legs; l++) {
        if (G.dev[l].pd) G.api.dealloc_pd(G.dev[l].pd);
        if (G.dev[l].ctx) G.api.close_device(G.dev[l].ctx);
    }
    G.n_conns = 0; G.n_mrc = 0; G.active = 0;
}

#endif /* DS4_RT_HAVE_VERBS */
