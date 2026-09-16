/* Exercise ds4_rdma_tier.c against a ds4_region_server, without ds4.
 * Checks: open + self-verification, single-threaded byte-exactness at many
 * offsets, and concurrent reads from many threads (ds4 fetches from up to 32).
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "../../ds4_rdma_tier.h"

static int g_fd;
static uint64_t g_span = 9950986, g_cov;
static volatile int g_fail;

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int check_span(void *buf, void *ref, uint64_t off, uint64_t bytes) {
    if (!ds4_rdma_tier_read(buf, off, bytes)) {
        fprintf(stderr, "FAIL: tier read refused at off=%llu\n", (unsigned long long)off);
        return 0;
    }
    uint64_t done = 0;
    while (done < bytes) {
        ssize_t k = pread(g_fd, (char *)ref + done, (size_t)(bytes - done), (off_t)(off + done));
        if (k <= 0) { fprintf(stderr, "FAIL: pread\n"); return 0; }
        done += (uint64_t)k;
    }
    if (memcmp(buf, ref, (size_t)bytes) != 0) {
        for (uint64_t i = 0; i < bytes; i++)
            if (((char *)buf)[i] != ((char *)ref)[i]) {
                fprintf(stderr, "FAIL: mismatch at off=%llu +%llu (half=%llu)\n",
                        (unsigned long long)off, (unsigned long long)i,
                        (unsigned long long)(bytes / 2));
                break;
            }
        return 0;
    }
    return 1;
}

static void *worker(void *vp) {
    const int id = (int)(long)vp;
    void *buf = NULL, *ref = NULL;
    if (posix_memalign(&buf, 4096, (size_t)g_span) ||
        posix_memalign(&ref, 4096, (size_t)g_span)) { g_fail = 1; return NULL; }
    const uint64_t slots = g_cov / g_span;
    for (int i = 0; i < 12; i++) {
        uint64_t off = ((uint64_t)(id * 7 + i * 3) % (slots ? slots : 1)) * g_span;
        if (off + g_span > g_cov) off = 0;
        if (!check_span(buf, ref, off, g_span)) { g_fail = 1; break; }
    }
    free(buf); free(ref);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_rdma_tier <file> [threads]\n"); return 1; }
    const int nthreads = argc > 2 ? atoi(argv[2]) : 8;
    g_fd = open(argv[1], O_RDONLY);
    if (g_fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(g_fd, &st);

    if (!ds4_rdma_tier_open(g_fd, (uint64_t)st.st_size)) {
        fprintf(stderr, "tier did not open (DS4_EXPERT_TIER_RDMA set? server up?)\n");
        return 1;
    }
    g_cov = ds4_rdma_tier_bytes();
    printf("tier open, coverage %.2f MiB, span %llu\n",
           (double)g_cov / 1048576.0, (unsigned long long)g_span);
    if (g_span > g_cov) g_span = g_cov & ~4095ull;

    void *buf = NULL, *ref = NULL;
    if (posix_memalign(&buf, 4096, (size_t)g_span) ||
        posix_memalign(&ref, 4096, (size_t)g_span)) { perror("posix_memalign"); return 1; }

    printf("-- single-threaded, 16 offsets --\n");
    const uint64_t slots = g_cov / g_span;
    for (uint64_t i = 0; i < 16; i++) {
        uint64_t off = (i % (slots ? slots : 1)) * g_span;
        if (off + g_span > g_cov) break;
        if (!check_span(buf, ref, off, g_span)) { g_fail = 1; break; }
    }
    /* unaligned offset and an odd length: the half split must still be exact */
    if (!g_fail && g_cov > g_span + 4097) {
        if (!check_span(buf, ref, 4097, g_span - 1234)) g_fail = 1;
        else printf("   unaligned offset + odd length: OK\n");
    }
    if (!g_fail) printf("   byte-exact\n");

    if (!g_fail) {
        printf("-- %d threads x 12 spans --\n", nthreads);
        pthread_t t[64];
        const double t0 = now();
        for (int i = 0; i < nthreads; i++) pthread_create(&t[i], NULL, worker, (void *)(long)i);
        for (int i = 0; i < nthreads; i++) pthread_join(t[i], NULL);
        const double dt = now() - t0;
        uint64_t reads = 0, bytes = 0, ns = 0;
        ds4_rdma_tier_stats(&reads, &bytes, &ns);
        printf("   %llu reads, %.2f GiB, %.2f GB/s aggregate, %.1f us/span mean\n",
               (unsigned long long)reads, (double)bytes / 1073741824.0,
               (double)bytes / dt / 1e9, reads ? (double)ns / 1e3 / (double)reads : 0.0);
    }
    /* ---- T4: promotion of spans past the mirrored prefix ---- */
    if (!g_fail && getenv("T1_TEST_PROMOTE")) {
        printf("-- promotion (spans past the %.0f MiB prefix) --\n",
               (double)g_cov / 1048576.0);
        const uint64_t span = 4u << 20;
        int promoted_ok = 0, tried = 0;
        const int nprom = getenv("T1_TEST_PROMOTE_N") ? atoi(getenv("T1_TEST_PROMOTE_N")) : 6;
        for (int k = 0; k < nprom && !g_fail; k++) {
            const uint64_t off = g_cov + (uint64_t)k * span;
            struct stat st2; fstat(g_fd, &st2);
            if (off + span > (uint64_t)st2.st_size) break;
            tried++;
            /* must not be served before it is promoted */
            if (ds4_rdma_tier_read(buf, off, span)) {
                fprintf(stderr, "FAIL: uncovered offset %llu served before promotion\n",
                        (unsigned long long)off);
                g_fail = 1; break;
            }
            uint64_t done = 0;
            while (done < span) {
                ssize_t r = pread(g_fd, (char *)ref + done, (size_t)(span - done),
                                  (off_t)(off + done));
                if (r <= 0) break;
                done += (uint64_t)r;
            }
            /* offer it until admission lets it through, then wait for the worker */
            for (int a = 0; a < 8; a++) ds4_rdma_tier_promote(ref, off, span);
            int landed = 0;
            for (int w = 0; w < 400 && !landed; w++) {
                if (ds4_rdma_tier_read(buf, off, span)) landed = 1;
                else { struct timespec ts = {0, 10 * 1000 * 1000}; nanosleep(&ts, NULL); }
            }
            if (!landed) { printf("   offset %llu: not promoted within 4 s\n",
                                  (unsigned long long)off); continue; }
            if (memcmp(buf, ref, (size_t)span) != 0) {
                fprintf(stderr, "FAIL: promoted span at %llu has WRONG BYTES\n",
                        (unsigned long long)off);
                uint64_t bad = 0;
                while (bad < span && ((char*)buf)[bad] == ((char*)ref)[bad]) bad++;
                fprintf(stderr, "  first differing byte at +%llu of %llu\n",
                        (unsigned long long)bad, (unsigned long long)span);
                fprintf(stderr, "  local (spark) :");
                for (int q = 0; q < 16; q++) fprintf(stderr, " %02x",
                        (unsigned char)((char*)ref)[bad + q]);
                fprintf(stderr, "\n  remote(promax):");
                for (int q = 0; q < 16; q++) fprintf(stderr, " %02x",
                        (unsigned char)((char*)buf)[bad + q]);
                fprintf(stderr, "\n  is remote all-zero here? ");
                { int nz = 0; for (uint64_t q = 0; q < span; q++) if (((char*)buf)[q]) { nz = 1; break; }
                  fprintf(stderr, "%s\n", nz ? "no, it has data" : "YES - slot never filled"); }
                if (getenv("T1_TEST_PROMOTE_N")) { continue; }
                g_fail = 1; break;
            }
            promoted_ok++;
        }
        uint64_t pr = 0, ph = 0, pk = 0; uint32_t ps = 0;
        ds4_rdma_tier_promote_stats(&pr, &ph, &pk, &ps);
        printf("   %d/%d promoted and byte-exact; stats: %llu promoted, %llu hits, "
               "%llu skipped, %u slots\n", promoted_ok, tried,
               (unsigned long long)pr, (unsigned long long)ph,
               (unsigned long long)pk, ps);
        if (promoted_ok == 0 && tried) { fprintf(stderr, "FAIL: nothing promoted\n"); g_fail = 1; }
    }

    ds4_rdma_tier_close();
    printf("%s\n", g_fail ? "*** FAILED ***" : "ALL CHECKS PASSED");
    return g_fail ? 2 : 0;
}
