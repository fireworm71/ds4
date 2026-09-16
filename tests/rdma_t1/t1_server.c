/* T1 server: register one region, serve it by one-sided RDMA READ on TWO
 * devices, then get out of the way.
 *
 * The server never touches the data path after registration -- that is the
 * whole structural point of one-sided READ over NVMe-oF (plan S3). It accepts
 * one client on each of two TCP ports, one per verbs device, hands over
 * {addr, rkey, qpn, gid, node_guid}, and then blocks until the client leaves,
 * whereupon it re-arms for the next one.
 *
 * Run it ONCE with both devices; do not run two copies.
 */
#include "t1_common.h"

static int listen_on(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) t1_die("socket: %s", strerror(errno));
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    sa.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&sa, sizeof sa))
        t1_die("bind(%d): %s", port, strerror(errno));
    if (listen(fd, 8)) t1_die("listen(%d): %s", port, strerror(errno));
    return fd;
}

static void usage(void) {
    fprintf(stderr,
      "usage: t1_server [--dev-a NAME] [--dev-b NAME] [--port-a N] [--port-b N]\n"
      "                 [--size BYTES] [--seed N] [--gid-index N] [--once]\n"
      "  Defaults: --dev-a rocep1s0f1 --dev-b roceP2p1s0f1\n"
      "            --port-a 19515 --port-b 19516 --size 268435456 --seed 1\n"
      "  Device names are THIS box's; on promax they may differ -- check\n"
      "  `ibv_devices`. --dev-a and --dev-b MUST be different physical devices\n"
      "  or striping silently measures as worthless (plan S6, T0 trap).\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *dev_a = "rocep1s0f1", *dev_b = "roceP2p1s0f1";
    int port_a = 19515, port_b = 19516, gid_index = T1_DEFAULT_GID_INDEX, once = 0;
    uint64_t size = 256ull << 20, seed = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--dev-a") && i + 1 < argc) dev_a = argv[++i];
        else if (!strcmp(a, "--dev-b") && i + 1 < argc) dev_b = argv[++i];
        else if (!strcmp(a, "--port-a") && i + 1 < argc) port_a = atoi(argv[++i]);
        else if (!strcmp(a, "--port-b") && i + 1 < argc) port_b = atoi(argv[++i]);
        else if (!strcmp(a, "--size") && i + 1 < argc) size = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--gid-index") && i + 1 < argc) gid_index = atoi(argv[++i]);
        else if (!strcmp(a, "--once")) once = 1;
        else usage();
    }
    if (!strcmp(dev_a, dev_b))
        t1_die("--dev-a and --dev-b are the same device (%s). Two QPs on one\n"
               "     remote device do NOT aggregate -- that is the T0 trap.", dev_a);
    size &= ~7ull;
    if (size < (1u << 20)) t1_die("--size too small");

    /* One region, served by both devices. */
    void *region = NULL;
    if (posix_memalign(&region, 4096, size)) t1_die("posix_memalign(%llu)",
                                                    (unsigned long long)size);
    t1_fill(region, size, seed);
    if (mlock(region, size))
        fprintf(stderr, "t1-server: WARNING mlock failed (%s) -- the region can "
                        "be paged out and the numbers will lie\n", strerror(errno));

    t1_ep ea, eb;
    t1_open(&ea, dev_a, gid_index);
    t1_open(&eb, dev_b, gid_index);
    if (ea.node_guid == eb.node_guid)
        t1_die("%s and %s report the same node_guid 0x%llx -- they are one\n"
               "     device, so striping cannot aggregate. Pick two.",
               dev_a, dev_b, (unsigned long long)ea.node_guid);

    ea.mr = ibv_reg_mr(ea.pd, region, size, IBV_ACCESS_REMOTE_READ);
    if (!ea.mr) t1_die("reg_mr(%s): %s", dev_a, strerror(errno));
    eb.mr = ibv_reg_mr(eb.pd, region, size, IBV_ACCESS_REMOTE_READ);
    if (!eb.mr) t1_die("reg_mr(%s): %s", dev_b, strerror(errno));

    printf("t1-server: region %.1f MiB @ %p  seed %llu\n",
           (double)size / 1048576.0, region, (unsigned long long)seed);
    printf("t1-server: leg A  dev %-14s guid 0x%016llx  rkey 0x%08x  tcp :%d\n",
           dev_a, (unsigned long long)ea.node_guid, ea.mr->rkey, port_a);
    printf("t1-server: leg B  dev %-14s guid 0x%016llx  rkey 0x%08x  tcp :%d\n",
           dev_b, (unsigned long long)eb.node_guid, eb.mr->rkey, port_b);
    printf("t1-server: distinct remote devices confirmed. waiting for clients.\n");
    fflush(stdout);

    int la = listen_on(port_a), lb = listen_on(port_b);

    for (;;) {
        int ca = accept(la, NULL, NULL);
        if (ca < 0) { if (errno == EINTR) continue; t1_die("accept A: %s", strerror(errno)); }
        int cb = accept(lb, NULL, NULL);
        if (cb < 0) { close(ca); if (errno == EINTR) continue;
                      t1_die("accept B: %s", strerror(errno)); }
        int one = 1;
        setsockopt(ca, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        setsockopt(cb, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

        t1_make_qp(&ea); t1_make_qp(&eb);
        uint32_t psn_a = 0x123456u, psn_b = 0x654321u;
        t1_hs ha, hb, pa, pb;
        t1_fill_hs(&ha, &ea, psn_a, (uint64_t)(uintptr_t)region, size, seed, ea.mr->rkey);
        t1_fill_hs(&hb, &eb, psn_b, (uint64_t)(uintptr_t)region, size, seed, eb.mr->rkey);

        if (t1_send_all(ca, &ha, sizeof ha) || t1_recv_all(ca, &pa, sizeof pa) ||
            t1_send_all(cb, &hb, sizeof hb) || t1_recv_all(cb, &pb, sizeof pb)) {
            fprintf(stderr, "t1-server: handshake dropped, re-arming\n");
            goto cycle_end;
        }
        if (pa.magic != T1_HS_MAGIC || pa.version != T1_HS_VERSION ||
            pb.magic != T1_HS_MAGIC || pb.version != T1_HS_VERSION) {
            fprintf(stderr, "t1-server: client handshake magic/version mismatch "
                            "(rebuild both sides), re-arming\n");
            goto cycle_end;
        }
        t1_connect(&ea, &pa, psn_a, IBV_ACCESS_REMOTE_READ);
        t1_connect(&eb, &pb, psn_b, IBV_ACCESS_REMOTE_READ);

        /* Tell the client both legs are RTS; then do nothing at all. */
        uint8_t go = 1;
        if (t1_send_all(ca, &go, 1) || t1_send_all(cb, &go, 1)) goto cycle_end;
        printf("t1-server: client connected (A qpn %u <- %u, B qpn %u <- %u); "
               "passive until it leaves\n", ea.qp->qp_num, pa.qpn,
               eb.qp->qp_num, pb.qpn);
        fflush(stdout);

        uint8_t bye;
        (void)t1_recv_all(ca, &bye, 1);   /* returns when the client closes */
        (void)t1_recv_all(cb, &bye, 1);
        printf("t1-server: client done\n"); fflush(stdout);

cycle_end:
        close(ca); close(cb);
        t1_drop_qp(&ea); t1_drop_qp(&eb);
        if (once) break;
    }
    return 0;
}
