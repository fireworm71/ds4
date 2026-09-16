/* ds4 remote-RAM expert region server (plan task T2).
 *
 * Serves the leading bytes of a model file out of this box's RAM, by one-sided
 * RDMA READ, to ds4's DS4_EXPERT_TIER_RDMA client. Two differences from
 * t1_server, which is a benchmark and not a backing store:
 *
 *   - the region is filled from a FILE, so the bytes are the model's, not a
 *     test pattern. ds4 verifies a sample of them against its own model file at
 *     startup and disables the tier on any mismatch;
 *   - it serves MANY client pairs at once, one thread per pair, because ds4
 *     fetches from up to 32 threads in parallel and a one-pair-at-a-time server
 *     would serialise all of them.
 *
 * After registration this process does nothing on the data path. Threads exist
 * only to accept connections and then block until the client leaves.
 */
#include <pthread.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include "t1_common.h"

typedef struct {
    t1_ep      *dev;         /* shared device (ctx/pd/mr), NOT per connection */
    struct ibv_mr *mr;
    int         fd;
    uint32_t    psn;
} leg_setup;

static void *region;
static uint64_t region_bytes;
static t1_ep dev_a, dev_b;
static pthread_mutex_t accept_lock = PTHREAD_MUTEX_INITIALIZER;
static volatile uint64_t live_clients;
static int g_want_odp;

/* Pinned registration is preferred: it is what the fabric numbers assume. When
 * RLIMIT_MEMLOCK forbids it, an on-demand-paging MR registers the same memory
 * without pinning -- ConnectX-7 reports rc_odp_caps SUPPORT_READ, which is
 * exactly what a one-sided READ responder needs. Pages fault into the HCA on
 * first touch, so the first pass over the region is slower than the rest. */
static struct ibv_mr *reg_region(struct ibv_pd *pd, void *addr, size_t len, int *odp) {
    if (!g_want_odp) {
        struct ibv_mr *mr = ibv_reg_mr(pd, addr, len, IBV_ACCESS_REMOTE_READ);
        if (mr) { *odp = 0; return mr; }
    }
    struct ibv_mr *mr = ibv_reg_mr(pd, addr, len,
                                   IBV_ACCESS_REMOTE_READ | IBV_ACCESS_ON_DEMAND);
    *odp = mr ? 1 : 0;
    return mr;
}

typedef struct { int fd_a, fd_b; } pair_arg;

/* Per connection: fresh CQ+QP on each shared device. */
static int serve_leg(t1_ep *dev, struct ibv_mr *mr, int fd, uint32_t psn,
                     struct ibv_cq **out_cq, struct ibv_qp **out_qp) {
    struct ibv_cq *cq = ibv_create_cq(dev->ctx, 16, NULL, NULL, 0);
    if (!cq) return 0;
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.send_cq = cq; qa.recv_cq = cq; qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = 16; qa.cap.max_recv_wr = 16;
    qa.cap.max_send_sge = 1; qa.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(dev->pd, &qa);
    if (!qp) { ibv_destroy_cq(cq); return 0; }

    t1_ep tmp = *dev;            /* t1_connect works on a t1_ep; borrow one */
    tmp.cq = cq; tmp.qp = qp;
    t1_hs mine, peer;
    t1_fill_hs(&mine, &tmp, psn, (uint64_t)(uintptr_t)region, region_bytes, 0, mr->rkey);
    if (t1_send_all(fd, &mine, sizeof mine) || t1_recv_all(fd, &peer, sizeof peer) ||
        peer.magic != T1_HS_MAGIC || peer.version != T1_HS_VERSION) {
        ibv_destroy_qp(qp); ibv_destroy_cq(cq); return 0;
    }
    t1_connect(&tmp, &peer, psn, IBV_ACCESS_REMOTE_READ);
    *out_cq = cq; *out_qp = qp;
    return 1;
}

static void *pair_thread(void *vp) {
    pair_arg *pa = (pair_arg *)vp;
    int fd_a = pa->fd_a, fd_b = pa->fd_b;
    free(pa);
    struct ibv_cq *cq_a = NULL, *cq_b = NULL;
    struct ibv_qp *qp_a = NULL, *qp_b = NULL;

    if (serve_leg(&dev_a, dev_a.mr, fd_a, 0x123456u, &cq_a, &qp_a) &&
        serve_leg(&dev_b, dev_b.mr, fd_b, 0x654321u, &cq_b, &qp_b)) {
        uint8_t go = 1;
        if (!t1_send_all(fd_a, &go, 1) && !t1_send_all(fd_b, &go, 1)) {
            __sync_fetch_and_add(&live_clients, 1);
            fprintf(stderr, "region-server: pair up (A qp %u, B qp %u); %llu live\n",
                    qp_a->qp_num, qp_b->qp_num, (unsigned long long)live_clients);
            uint8_t bye;
            (void)t1_recv_all(fd_a, &bye, 1);   /* returns when the client goes */
            (void)t1_recv_all(fd_b, &bye, 1);
            __sync_fetch_and_sub(&live_clients, 1);
            fprintf(stderr, "region-server: pair gone; %llu live\n",
                    (unsigned long long)live_clients);
        }
    } else {
        fprintf(stderr, "region-server: pair setup failed\n");
    }
    if (qp_a) ibv_destroy_qp(qp_a);
    if (qp_b) ibv_destroy_qp(qp_b);
    if (cq_a) ibv_destroy_cq(cq_a);
    if (cq_b) ibv_destroy_cq(cq_b);
    close(fd_a); close(fd_b);
    return NULL;
}

static int listen_on(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    if (fd < 0) t1_die("socket: %s", strerror(errno));
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_ANY);
    sa.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&sa, sizeof sa)) t1_die("bind(%d): %s", port, strerror(errno));
    if (listen(fd, 64)) t1_die("listen(%d): %s", port, strerror(errno));
    return fd;
}

static void usage(void) {
    fprintf(stderr,
      "usage: ds4_region_server --file MODEL.gguf [options]\n"
      "  --file PATH      fill the region from this file's leading bytes (required\n"
      "                   unless --pattern). Must be the SAME model file ds4 opens.\n"
      "  --bytes N        how much of it to hold (default: all of it, RAM permitting)\n"
      "  --pattern        fill with the t1 test pattern instead of a file (testing)\n"
      "  --size N         region size when --pattern is used\n"
      "  --dev-a/-b NAME  verbs devices (default rocep1s0f1 / roceP2p1s0f1)\n"
      "  --port-a/-b N    TCP ports (default 19515 / 19516)\n"
      "  --gid-index N    default 3\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *dev_a_name = "rocep1s0f1", *dev_b_name = "roceP2p1s0f1";
    const char *file = NULL;
    int port_a = 19515, port_b = 19516, gid_index = T1_DEFAULT_GID_INDEX, pattern = 0;
    uint64_t want = 0, size = 256ull << 20;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--file") && i + 1 < argc) file = argv[++i];
        else if (!strcmp(a, "--bytes") && i + 1 < argc) want = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--pattern")) pattern = 1;
        else if (!strcmp(a, "--size") && i + 1 < argc) size = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--dev-a") && i + 1 < argc) dev_a_name = argv[++i];
        else if (!strcmp(a, "--dev-b") && i + 1 < argc) dev_b_name = argv[++i];
        else if (!strcmp(a, "--port-a") && i + 1 < argc) port_a = atoi(argv[++i]);
        else if (!strcmp(a, "--port-b") && i + 1 < argc) port_b = atoi(argv[++i]);
        else if (!strcmp(a, "--gid-index") && i + 1 < argc) gid_index = atoi(argv[++i]);
        else usage();
    }
    if (!file && !pattern) usage();
    if (!strcmp(dev_a_name, dev_b_name))
        t1_die("--dev-a and --dev-b are the same device -- two QPs on one device\n"
               "     do not aggregate. See the T0 trap in the plan.");

    /* Check the pinning budget BEFORE spending minutes loading. ibv_reg_mr pins
     * the whole region, so RLIMIT_MEMLOCK below the region size fails the
     * registration outright -- free RAM is irrelevant. A process may raise its
     * own soft limit up to the hard limit without root, so try that first;
     * that alone fixes the common case where hard is unlimited and soft is not. */
    {
        uint64_t need = want ? want : (uint64_t)0;
        if (file && need == 0) {
            struct stat pst;
            if (stat(file, &pst) == 0) need = (uint64_t)pst.st_size;
        } else if (!file) {
            need = size;
        }
        struct rlimit rl;
        if (need && getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
            if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < need) {
                struct rlimit up = rl;
                up.rlim_cur = rl.rlim_max;          /* no root needed for this */
                if (setrlimit(RLIMIT_MEMLOCK, &up) == 0) {
                    (void)getrlimit(RLIMIT_MEMLOCK, &rl);
                    if (rl.rlim_cur == RLIM_INFINITY || rl.rlim_cur >= need)
                        fprintf(stderr, "region-server: raised memlock soft limit "
                                        "to the hard limit\n");
                }
            }
            if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < need) {
                fprintf(stderr,
                  "region-server: memlock limit is %.2f GiB but the region needs\n"
                  "  %.2f GiB. Falling back to an ON-DEMAND PAGING registration, which\n"
                  "  does not pin and so is not bound by that limit. The pages are\n"
                  "  resident (this process just read them) but the kernel MAY evict\n"
                  "  them under pressure, in which case timings understate the fabric.\n"
                  "  For pinned registration instead, as root and then re-login:\n"
                  "    <user> hard memlock unlimited\n"
                  "    <user> soft memlock unlimited\n",
                  (double)rl.rlim_cur / 1073741824.0, (double)need / 1073741824.0);
                g_want_odp = 1;
            }
        }
    }

    if (file) {
        int fd = open(file, O_RDONLY);
        if (fd < 0) t1_die("open(%s): %s", file, strerror(errno));
        struct stat st;
        if (fstat(fd, &st)) t1_die("fstat(%s): %s", file, strerror(errno));
        region_bytes = (uint64_t)st.st_size;
        if (want && want < region_bytes) region_bytes = want;
        if (region_bytes == 0) t1_die("%s is empty", file);
        if (posix_memalign(&region, 4096, region_bytes))
            t1_die("cannot allocate %.2f GiB", (double)region_bytes / 1073741824.0);
        fprintf(stderr, "region-server: reading %.2f GiB from %s ...\n",
                (double)region_bytes / 1073741824.0, file);
        uint64_t done = 0;
        while (done < region_bytes) {
            ssize_t k = pread(fd, (char *)region + done,
                              (size_t)(region_bytes - done > (64u << 20)
                                       ? (64u << 20) : region_bytes - done),
                              (off_t)done);
            if (k < 0) { if (errno == EINTR) continue; t1_die("pread: %s", strerror(errno)); }
            if (k == 0) { region_bytes = done; break; }   /* short file */
            done += (uint64_t)k;
        }
        close(fd);
        fprintf(stderr, "region-server: loaded %.2f GiB\n",
                (double)region_bytes / 1073741824.0);
    } else {
        region_bytes = size & ~7ull;
        if (posix_memalign(&region, 4096, region_bytes)) t1_die("alloc failed");
        t1_fill(region, region_bytes, 1);
    }
    if (!g_want_odp && mlock(region, region_bytes))
        fprintf(stderr, "region-server: WARNING mlock failed (%s) -- the region can be\n"
                        "   paged out, which makes every number here a lie\n", strerror(errno));

    t1_open(&dev_a, dev_a_name, gid_index);
    t1_open(&dev_b, dev_b_name, gid_index);
    if (dev_a.node_guid == dev_b.node_guid)
        t1_die("%s and %s are one device (guid 0x%llx) -- cannot stripe",
               dev_a_name, dev_b_name, (unsigned long long)dev_a.node_guid);
    int odp_a = 0, odp_b = 0;
    dev_a.mr = reg_region(dev_a.pd, region, region_bytes, &odp_a);
    dev_b.mr = reg_region(dev_b.pd, region, region_bytes, &odp_b);
    if (!dev_a.mr || !dev_b.mr)
        t1_die("reg_mr failed: %s\n"
               "     Even on-demand paging did not register the region. Serve less\n"
               "     with --bytes, or raise memlock as root.", strerror(errno));
    if (odp_a || odp_b)
        fprintf(stderr, "region-server: registered WITHOUT pinning (on-demand paging).\n"
                        "   Expect the first pass over the region to be slower, and keep\n"
                        "   an eye on free RAM -- these pages are evictable.\n");

    printf("region-server: %.2f GiB @ %p from %s\n",
           (double)region_bytes / 1073741824.0, region, file ? file : "(pattern)");
    printf("region-server: leg A %-14s guid 0x%016llx rkey 0x%08x :%d\n",
           dev_a_name, (unsigned long long)dev_a.node_guid, dev_a.mr->rkey, port_a);
    printf("region-server: leg B %-14s guid 0x%016llx rkey 0x%08x :%d\n",
           dev_b_name, (unsigned long long)dev_b.node_guid, dev_b.mr->rkey, port_b);
    printf("region-server: distinct devices confirmed; serving many pairs.\n");
    fflush(stdout);

    int la = listen_on(port_a), lb = listen_on(port_b);
    for (;;) {
        /* One pair at a time through accept, so leg A and leg B of the same
         * client cannot interleave with another client's. The pair is then
         * handed to its own thread, so N clients are served concurrently. */
        pthread_mutex_lock(&accept_lock);
        int ca = accept(la, NULL, NULL);
        int cb = ca >= 0 ? accept(lb, NULL, NULL) : -1;
        pthread_mutex_unlock(&accept_lock);
        if (ca < 0 || cb < 0) { if (ca >= 0) close(ca); if (cb >= 0) close(cb); continue; }
        int one = 1;
        setsockopt(ca, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        setsockopt(cb, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        pair_arg *pa = (pair_arg *)malloc(sizeof *pa);
        if (!pa) { close(ca); close(cb); continue; }
        pa->fd_a = ca; pa->fd_b = cb;
        pthread_t th;
        if (pthread_create(&th, NULL, pair_thread, pa) != 0) {
            fprintf(stderr, "region-server: thread create failed\n");
            close(ca); close(cb); free(pa);
        } else {
            pthread_detach(th);
        }
    }
    return 0;
}
