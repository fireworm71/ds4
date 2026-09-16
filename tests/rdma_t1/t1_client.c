/* T1 client: fetch one expert-sized span by one-sided RDMA READ, striped
 * across two QPs on two devices, and verify every byte.
 *
 * Targets from T0 (misc/T0_RESULT.md): Q2 9.49 MiB -> ~413 us striped
 * (739 us on one NIC); Q4 18.98 MiB -> ~819 us striped (1479 us on one NIC).
 *
 * THE GUARD THAT MATTERS: --striped refuses to print a number unless the two
 * legs sit on different REMOTE devices. Two QPs on one remote device split
 * that device's ~13 GB/s exactly 50/50 and striping measures as worth nothing.
 * That false negative is what T0 hit first; see plan S6.
 */
#include "t1_common.h"

static int connect_to(const char *host, int port) {
    char svc[16]; snprintf(svc, sizeof svc, "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host, svc, &hints, &res);
    if (rc) t1_die("getaddrinfo(%s:%d): %s", host, port, gai_strerror(rc));
    int fd = socket(res->ai_family, res->ai_socktype, 0);
    if (fd < 0) t1_die("socket: %s", strerror(errno));
    if (connect(fd, res->ai_addr, res->ai_addrlen))
        t1_die("connect(%s:%d): %s -- is t1_server running there?",
               host, port, strerror(errno));
    freeaddrinfo(res);
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

static uint64_t counter_rcv(const char *dev) {
    char p[256]; FILE *f; unsigned long long v = 0;
    snprintf(p, sizeof p, "/sys/class/infiniband/%s/ports/1/counters/port_rcv_data", dev);
    if ((f = fopen(p, "r"))) { if (fscanf(f, "%llu", &v) != 1) v = 0; fclose(f); }
    return (uint64_t)v * 4ull;   /* counter is in 4-byte words */
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Post one RDMA READ and return; completion is polled separately so the two
 * legs overlap instead of serialising. */
static void post_read(t1_ep *e, void *laddr, uint32_t lkey, uint64_t raddr,
                      uint32_t rkey, uint64_t bytes, uint64_t wr_id) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof sge);
    sge.addr = (uint64_t)(uintptr_t)laddr; sge.length = (uint32_t)bytes; sge.lkey = lkey;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = wr_id; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
    if (ibv_post_send(e->qp, &wr, &bad))
        t1_die("post_send(%s): %s", e->dev, strerror(errno));
}

static void poll_one(t1_ep *e) {
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(e->cq, 1, &wc);
        if (n < 0) t1_die("poll_cq(%s) failed", e->dev);
        if (n == 0) continue;
        if (wc.status != IBV_WC_SUCCESS)
            t1_die("READ on %s failed: %s (%d)", e->dev,
                   ibv_wc_status_str(wc.status), (int)wc.status);
        return;
    }
}

static void usage(void) {
    fprintf(stderr,
      "usage: t1_client --host-a IP --host-b IP [options]\n"
      "  --host-a IP        server address reachable over local dev A (e.g. 10.99.0.2)\n"
      "  --host-b IP        server address reachable over local dev B (e.g. 10.99.2.2)\n"
      "  --port-a/-b N      server TCP ports (default 19515 / 19516)\n"
      "  --dev-a/-b NAME    local verbs devices (default rocep1s0f1 / roceP2p1s0f1)\n"
      "  --span BYTES       span to fetch (default 9950986 = 9.49 MiB, a Q2 expert)\n"
      "  --iters N          spans to fetch (default 200)\n"
      "  --single           one leg only, whole span on dev A (the baseline arm)\n"
      "  --gid-index N      default 3 (RoCEv2 here)\n"
      "  --no-verify        skip byte checking (verification is ON by default)\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *host_a = NULL, *host_b = NULL;
    const char *dev_a = "rocep1s0f1", *dev_b = "roceP2p1s0f1";
    int port_a = 19515, port_b = 19516, gid_index = T1_DEFAULT_GID_INDEX;
    int single = 0, verify = 1, iters = 200;
    uint64_t span = 9950986;   /* 9.49 MiB, the Q2 expert of the plan */

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--host-a") && i + 1 < argc) host_a = argv[++i];
        else if (!strcmp(a, "--host-b") && i + 1 < argc) host_b = argv[++i];
        else if (!strcmp(a, "--port-a") && i + 1 < argc) port_a = atoi(argv[++i]);
        else if (!strcmp(a, "--port-b") && i + 1 < argc) port_b = atoi(argv[++i]);
        else if (!strcmp(a, "--dev-a") && i + 1 < argc) dev_a = argv[++i];
        else if (!strcmp(a, "--dev-b") && i + 1 < argc) dev_b = argv[++i];
        else if (!strcmp(a, "--span") && i + 1 < argc) span = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(a, "--gid-index") && i + 1 < argc) gid_index = atoi(argv[++i]);
        else if (!strcmp(a, "--single")) single = 1;
        else if (!strcmp(a, "--no-verify")) verify = 0;
        else usage();
    }
    if (!host_a || !host_b) usage();
    span &= ~7ull;
    if (span < 4096 || iters < 1) t1_die("bad --span/--iters");

    t1_ep ea, eb;
    t1_open(&ea, dev_a, gid_index);
    t1_open(&eb, dev_b, gid_index);
    t1_make_qp(&ea); t1_make_qp(&eb);

    int fa = connect_to(host_a, port_a), fb = connect_to(host_b, port_b);
    t1_hs pa, pb, ma, mb;
    if (t1_recv_all(fa, &pa, sizeof pa)) t1_die("handshake A failed");
    t1_fill_hs(&ma, &ea, 0xabcdefu, 0, 0, 0, 0);
    if (t1_send_all(fa, &ma, sizeof ma)) t1_die("handshake A send failed");
    if (t1_recv_all(fb, &pb, sizeof pb)) t1_die("handshake B failed");
    t1_fill_hs(&mb, &eb, 0xfedcbau, 0, 0, 0, 0);
    if (t1_send_all(fb, &mb, sizeof mb)) t1_die("handshake B send failed");
    if (pa.magic != T1_HS_MAGIC || pa.version != T1_HS_VERSION ||
        pb.magic != T1_HS_MAGIC || pb.version != T1_HS_VERSION)
        t1_die("server handshake magic/version mismatch -- rebuild both sides");

    printf("leg A: local %-14s -> remote %-14s guid 0x%016llx\n",
           dev_a, pa.dev, (unsigned long long)pa.node_guid);
    printf("leg B: local %-14s -> remote %-14s guid 0x%016llx\n",
           dev_b, pb.dev, (unsigned long long)pb.node_guid);

    /* ---- the T0 trap guard ---- */
    if (!single && pa.node_guid == pb.node_guid)
        t1_die("BOTH LEGS LAND ON THE SAME REMOTE DEVICE (guid 0x%016llx).\n"
               "     Striping cannot aggregate: one remote device is one ~13 GB/s\n"
               "     sender however many local NICs pull. Any number measured in\n"
               "     this configuration is a false negative (plan S6, T0 trap).\n"
               "     Fix the server's --dev-a/--dev-b, or the addresses you passed.",
               (unsigned long long)pa.node_guid);
    if (!single && ea.node_guid == eb.node_guid)
        t1_die("both legs use the same LOCAL device (%s)", dev_a);
    if (span > pa.len) t1_die("--span %llu exceeds the server region (%llu bytes)",
                              (unsigned long long)span, (unsigned long long)pa.len);

    t1_connect(&ea, &pa, 0xabcdefu, IBV_ACCESS_LOCAL_WRITE);
    t1_connect(&eb, &pb, 0xfedcbau, IBV_ACCESS_LOCAL_WRITE);
    uint8_t go;
    if (t1_recv_all(fa, &go, 1) || t1_recv_all(fb, &go, 1))
        t1_die("server did not signal ready");

    void *buf = NULL;
    if (posix_memalign(&buf, 4096, span)) t1_die("posix_memalign");
    memset(buf, 0, span);
    ea.mr = ibv_reg_mr(ea.pd, buf, span, IBV_ACCESS_LOCAL_WRITE);
    if (!ea.mr) t1_die("reg_mr local A: %s", strerror(errno));
    eb.mr = ibv_reg_mr(eb.pd, buf, span, IBV_ACCESS_LOCAL_WRITE);
    if (!eb.mr) t1_die("reg_mr local B: %s", strerror(errno));

    uint64_t half = single ? span : ((span / 2) & ~7ull);
    uint64_t rest = span - half;
    /* Walk the region so no two iterations read the same bytes (rule 2). */
    uint64_t nslots = pa.len / span; if (nslots == 0) nslots = 1;

    double *t = (double *)malloc((size_t)iters * sizeof *t);
    uint64_t c_a0 = counter_rcv(dev_a), c_b0 = counter_rcv(dev_b);
    double wall0 = t1_now();
    long long bad_off = -1; uint64_t bad_iter = 0;

    for (int i = 0; i < iters; i++) {
        uint64_t off = (uint64_t)((uint64_t)i % nslots) * span;
        double s = t1_now();
        if (single) {
            post_read(&ea, buf, ea.mr->lkey, pa.addr + off, pa.rkey, span, 1);
            poll_one(&ea);
        } else {
            /* both legs in flight before either is waited on */
            post_read(&ea, buf, ea.mr->lkey, pa.addr + off, pa.rkey, half, 1);
            post_read(&eb, (char *)buf + half, eb.mr->lkey,
                      pb.addr + off + half, pb.rkey, rest, 2);
            poll_one(&ea); poll_one(&eb);
        }
        t[i] = t1_now() - s;
        if (verify && bad_off < 0) {
            long long e = t1_verify(buf, span, off, pa.seed);
            if (e >= 0) { bad_off = e; bad_iter = (uint64_t)i; }
        }
    }
    double wall = t1_now() - wall0;
    uint64_t c_a1 = counter_rcv(dev_a), c_b1 = counter_rcv(dev_b);

    qsort(t, (size_t)iters, sizeof *t, cmp_d);
    double sum = 0; for (int i = 0; i < iters; i++) sum += t[i];
    double mean = sum / iters, med = t[iters / 2], lo = t[0], hi = t[iters - 1];

    printf("\n== %s ==\n", single ? "SINGLE LEG (baseline)" : "STRIPED across 2 NICs");
    printf("span        %llu B (%.2f MiB)%s\n", (unsigned long long)span,
           (double)span / 1048576.0,
           single ? "" : "  split into two halves");
    printf("iters       %d over %.2f s\n", iters, wall);
    printf("per-span    min %.1f us  median %.1f us  mean %.1f us  max %.1f us\n",
           lo * 1e6, med * 1e6, mean * 1e6, hi * 1e6);
    /* Aggregate is computed from the summed per-span times, not wall: when
     * --no-verify is off, wall also contains the CPU cost of checking every
     * byte of every span, which has nothing to do with the transport. */
    printf("throughput  %.2f GB/s (median)   %.2f GB/s (sum of per-span times)\n",
           (double)span / med / 1e9, (double)span * iters / sum / 1e9);
    printf("NIC bytes   %s %.2f GB   %s %.2f GB\n",
           dev_a, (double)(c_a1 - c_a0) / 1e9, dev_b, (double)(c_b1 - c_b0) / 1e9);
    if (!single) {
        double fa_ = (double)(c_a1 - c_a0), fb_ = (double)(c_b1 - c_b0);
        double tot = fa_ + fb_;
        printf("split       %.1f%% / %.1f%%%s\n", tot ? 100.0 * fa_ / tot : 0.0,
               tot ? 100.0 * fb_ / tot : 0.0,
               (tot && (fa_ / tot < 0.4 || fa_ / tot > 0.6))
                   ? "   <-- IMBALANCED, one leg is not pulling its half" : "");
    }
    if (verify)
        printf("verify      %s\n", bad_off < 0 ? "ALL BYTES MATCH"
              : "*** MISMATCH ***");
    if (verify && bad_off >= 0) {
        printf("            first bad byte at +%lld of the span, iteration %llu\n",
               bad_off, (unsigned long long)bad_iter);
        printf("            (a mismatch at exactly the half boundary %llu means the\n"
               "             two legs' address split is wrong, not the fabric)\n",
               (unsigned long long)half);
    }
    fflush(stdout);
    close(fa); close(fb);
    return (verify && bad_off >= 0) ? 2 : 0;
}
