/* T1 prototype: one-sided RDMA READ expert transport, striped across 2 NICs.
 *
 * Shared definitions for t1_server / t1_client.  Standalone -- no ds4 code.
 * The QP recipe mirrors ds4_tp.c (RoCEv2, GRH, IPv4-mapped GID), with the
 * rd_atomic attributes ds4_tp.c omits because it is two-sided SEND/RECV and
 * never issues an RDMA READ.
 */
#ifndef T1_COMMON_H
#define T1_COMMON_H

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#define T1_DEFAULT_GID_INDEX 3      /* RoCEv2 on every device here (plan S2) */
#define T1_RD_ATOMIC        16

/* Handshake. Both boxes are aarch64 little-endian; a prototype does not need
 * byte-order conversion, but it does need a version tag so a stale binary on
 * one side fails loudly instead of reading garbage rkeys. */
#define T1_HS_MAGIC 0x54315253u  /* "T1RS" */
#define T1_HS_VERSION 2u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t qpn;
    uint32_t psn;
    uint32_t lid;
    uint32_t rkey;
    uint8_t  gid[16];
    uint64_t addr;       /* base of the remote region */
    uint64_t len;        /* its length */
    uint64_t seed;       /* pattern seed, so the client can verify bytes */
    uint64_t node_guid;  /* WHICH REMOTE DEVICE -- see t1_assert_distinct() */
    char     dev[64];    /* its name, for the human */
} t1_hs;

static inline void t1_die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "t1: "); vfprintf(stderr, fmt, ap);
    if (fmt[strlen(fmt) - 1] != '\n') fprintf(stderr, "\n");
    va_end(ap); exit(1);
}

static inline double t1_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Deterministic, position-dependent content. A shifted, truncated or
 * half-missing read cannot pass this: every 8-byte word encodes its own
 * offset. */
static inline uint64_t t1_word(uint64_t idx, uint64_t seed) {
    uint64_t x = idx * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static inline void t1_fill(void *buf, uint64_t bytes, uint64_t seed) {
    uint64_t *w = (uint64_t *)buf, n = bytes / 8;
    for (uint64_t i = 0; i < n; i++) w[i] = t1_word(i, seed);
}

/* Returns byte offset of the first mismatch, or -1 when clean. `off` is the
 * offset of buf within the remote region. */
static inline long long t1_verify(const void *buf, uint64_t bytes,
                                  uint64_t off, uint64_t seed) {
    const uint64_t *w = (const uint64_t *)buf;
    uint64_t n = bytes / 8, base = off / 8;
    for (uint64_t i = 0; i < n; i++)
        if (w[i] != t1_word(base + i, seed)) return (long long)(i * 8);
    return -1;
}

/* ---- full-duplex TCP helpers (handshake only, never the data path) ---- */

static inline int t1_send_all(int fd, const void *p, size_t n) {
    const char *c = (const char *)p;
    while (n) {
        ssize_t k = send(fd, c, n, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}

static inline int t1_recv_all(int fd, void *p, size_t n) {
    char *c = (char *)p;
    while (n) {
        ssize_t k = recv(fd, c, n, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}

/* ---- verbs helpers ---- */

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd      *pd;
    struct ibv_cq      *cq;
    struct ibv_qp      *qp;
    struct ibv_mr      *mr;
    union ibv_gid       gid;
    int                 gid_index;
    enum ibv_mtu        mtu;
    uint64_t            node_guid;
    char                dev[64];
} t1_ep;

static inline void t1_open(t1_ep *e, const char *dev_name, int gid_index) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list || n == 0) t1_die("no verbs devices");
    struct ibv_device *dev = NULL;
    for (int i = 0; i < n; i++)
        if (!strcmp(ibv_get_device_name(list[i]), dev_name)) { dev = list[i]; break; }
    if (!dev) {
        fprintf(stderr, "t1: device '%s' not found. available:", dev_name);
        for (int i = 0; i < n; i++) fprintf(stderr, " %s", ibv_get_device_name(list[i]));
        fprintf(stderr, "\n"); exit(1);
    }
    snprintf(e->dev, sizeof e->dev, "%s", dev_name);
    e->node_guid = (uint64_t)ibv_get_device_guid(dev);
    e->ctx = ibv_open_device(dev);
    if (!e->ctx) t1_die("open_device(%s) failed", dev_name);
    ibv_free_device_list(list);

    struct ibv_port_attr port;
    if (ibv_query_port(e->ctx, 1, &port)) t1_die("query_port(%s)", dev_name);
    if (port.state != IBV_PORT_ACTIVE)
        t1_die("%s port 1 is not ACTIVE (state %d)", dev_name, (int)port.state);
    e->mtu = port.active_mtu;          /* 4096 here, not ds4_tp.c's 1024 */
    e->gid_index = gid_index;
    if (ibv_query_gid(e->ctx, 1, gid_index, &e->gid))
        t1_die("query_gid(%s, %d)", dev_name, gid_index);

    e->pd = ibv_alloc_pd(e->ctx);
    if (!e->pd) t1_die("alloc_pd(%s)", dev_name);
    e->cq = NULL; e->qp = NULL; e->mr = NULL;
}

/* CQ+QP are per-connection: the server tears them down and re-arms between
 * clients so it outlives any one of them (plan S7), while ctx/pd/mr -- and
 * therefore addr and rkey -- stay stable for the region's lifetime. */
static inline void t1_make_qp(t1_ep *e) {
    e->cq = ibv_create_cq(e->ctx, 64, NULL, NULL, 0);
    if (!e->cq) t1_die("create_cq(%s)", e->dev);
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.send_cq = e->cq; qa.recv_cq = e->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = 64; qa.cap.max_recv_wr = 64;
    qa.cap.max_send_sge = 1; qa.cap.max_recv_sge = 1;
    e->qp = ibv_create_qp(e->pd, &qa);
    if (!e->qp) t1_die("create_qp(%s)", e->dev);
}

static inline void t1_drop_qp(t1_ep *e) {
    if (e->qp) { ibv_destroy_qp(e->qp); e->qp = NULL; }
    if (e->cq) { ibv_destroy_cq(e->cq); e->cq = NULL; }
}

static inline void t1_fill_hs(t1_hs *h, const t1_ep *e, uint32_t psn,
                              uint64_t addr, uint64_t len, uint64_t seed,
                              uint32_t rkey) {
    memset(h, 0, sizeof *h);
    h->magic = T1_HS_MAGIC; h->version = T1_HS_VERSION;
    h->qpn = e->qp->qp_num; h->psn = psn; h->lid = 0;
    h->rkey = rkey; h->addr = addr; h->len = len; h->seed = seed;
    h->node_guid = e->node_guid;
    memcpy(h->gid, e->gid.raw, 16);
    snprintf(h->dev, sizeof h->dev, "%s", e->dev);
}

/* INIT -> RTR -> RTS. Same recipe as ds4_tp.c plus the rd_atomic attributes
 * that RDMA READ needs on both ends, and the reliability timers. */
static inline void t1_connect(t1_ep *e, const t1_hs *peer, uint32_t my_psn,
                              int access) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0; a.port_num = 1;
    a.qp_access_flags = access;
    if (ibv_modify_qp(e->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                 IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        t1_die("modify INIT (%s): %s", e->dev, strerror(errno));

    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = e->mtu;
    a.dest_qp_num = peer->qpn;
    a.rq_psn = peer->psn;
    a.max_dest_rd_atomic = T1_RD_ATOMIC;   /* responder side of READ */
    a.min_rnr_timer = 12;
    a.ah_attr.dlid = (uint16_t)peer->lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)e->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (ibv_modify_qp(e->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                 IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                 IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        t1_die("modify RTR (%s): %s", e->dev, strerror(errno));

    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = my_psn;
    a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7;
    a.max_rd_atomic = T1_RD_ATOMIC;        /* requester side of READ */
    if (ibv_modify_qp(e->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                                 IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                 IBV_QP_MAX_QP_RD_ATOMIC))
        t1_die("modify RTS (%s): %s", e->dev, strerror(errno));
}

#endif /* T1_COMMON_H */
