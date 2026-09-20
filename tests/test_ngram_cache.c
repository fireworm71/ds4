#include "ds4_ngram.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

int main(void) {
    printf("Running test_ngram_cache...\n");

    ds4_ngram_table context_tbl, dynamic_tbl;
    assert(ds4_ngram_table_init(&context_tbl, 1024, 0));
    assert(ds4_ngram_table_init(&dynamic_tbl, 1024, 1));

    /* Sequence: [10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60] */
    int32_t seq1[] = {10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60};
    ds4_ngram_table_update(&context_tbl, seq1, 12, 12);

    /* Test 2-gram lookup: (50) -> next is 60? 2-gram prefix is (40, 50) -> 60 */
    int32_t prefix2[] = {40, 50};
    int32_t out_tok = -1;
    float conf = 0.0f;
    assert(ds4_ngram_table_lookup(&context_tbl, prefix2, 2, &out_tok, &conf, 1));
    assert(out_tok == 60);
    printf("  [PASS] 2-gram lookup (40, 50) -> %d (conf=%.2f)\n", out_tok, conf);

    /* Test 3-gram lookup: (30, 40, 50) -> 60 */
    int32_t prefix3[] = {30, 40, 50};
    assert(ds4_ngram_table_lookup(&context_tbl, prefix3, 3, &out_tok, &conf, 1));
    assert(out_tok == 60);
    printf("  [PASS] 3-gram lookup (30, 40, 50) -> %d (conf=%.2f)\n", out_tok, conf);

    /* Test 4-gram lookup: (20, 30, 40, 50) -> 60 */
    int32_t prefix4[] = {20, 30, 40, 50};
    assert(ds4_ngram_table_lookup(&context_tbl, prefix4, 4, &out_tok, &conf, 1));
    assert(out_tok == 60);
    printf("  [PASS] 4-gram lookup (20, 30, 40, 50) -> %d (conf=%.2f)\n", out_tok, conf);

    /* Test 1-gram lookup: (50) -> next is 60 */
    int32_t prefix1[] = {50};
    assert(ds4_ngram_table_lookup(&context_tbl, prefix1, 1, &out_tok, &conf, 1));
    assert(out_tok == 60);
    printf("  [PASS] 1-gram lookup (50) -> %d (conf=%.2f)\n", out_tok, conf);

    /* Test chained drafting: starting from history [10, 20] and current_token 30 */
    int32_t hist[] = {10, 20};
    int32_t drafts[16] = {0};
    uint32_t k = ds4_ngram_draft(&context_tbl, &dynamic_tbl, hist, 2, 30, drafts, 6);
    printf("  Chained draft from [10, 20, 30]: count=%u tokens=[", k);
    for (uint32_t i = 0; i < k; i++) printf("%d%s", drafts[i], i + 1 < k ? ", " : "");
    printf("]\n");
    assert(k >= 3);
    assert(drafts[0] == 40);
    assert(drafts[1] == 50);
    assert(drafts[2] == 60);
    printf("  [PASS] Chained drafting matched expected continuation (40, 50, 60)\n");

    /* Test Dynamic tier persistence: save context_tbl to /tmp/test_ngram.bin and load into dynamic_tbl */
    const char *tmp_path = "/tmp/test_ngram.bin";
    assert(ds4_ngram_table_save(&context_tbl, tmp_path));
    assert(ds4_ngram_table_load(&dynamic_tbl, tmp_path));

    /* Verify dynamic_tbl has entries */
    assert(dynamic_tbl.total_entries > 0);
    assert(ds4_ngram_table_lookup(&dynamic_tbl, prefix3, 3, &out_tok, &conf, 1));
    assert(out_tok == 60);
    printf("  [PASS] Persistence verified (saved %u entries, successfully reloaded)\n", dynamic_tbl.total_entries);

    /* Test 3-Tier Fallback Precedence: Tier 2 (Context) -> Tier 1 (Static) -> Tier 3 (Dynamic) */
    printf("  Testing 3-tier fallback precedence...\n");
    ds4_ngram_table static_tbl;
    assert(ds4_ngram_table_init(&static_tbl, 1024, 2));

    /* Dynamic table has: [100, 200, 300] -> 301 (from dynamic) */
    int32_t dyn_seq[] = {100, 200, 300, 301};
    ds4_ngram_table_update(&dynamic_tbl, dyn_seq, 4, 4);

    /* Static table has: [100, 200, 300] -> 302 (static override)
     * and [400, 500, 600] -> 602 */
    int32_t sta_seq[] = {100, 200, 300, 302, 400, 500, 600, 602};
    ds4_ngram_table_update(&static_tbl, sta_seq, 8, 8);

    /* Context table has: [100, 200, 300] -> 303 (context override) */
    int32_t ctx_seq[] = {100, 200, 300, 303};
    ds4_ngram_table_update(&context_tbl, ctx_seq, 4, 4);

    /* Case A: [100, 200, 300] is in all 3 tiers -> Tier 2 (Context) must win (yielding 303) */
    int32_t test_histA[] = {100, 200};
    int32_t draft_res[4] = {0};
    uint32_t d_count = ds4_ngram_draft_3tier(&context_tbl, &static_tbl, &dynamic_tbl,
                                             test_histA, 2, 300, draft_res, 1);
    assert(d_count == 1);
    assert(draft_res[0] == 303);
    printf("    [PASS] Tier 2 (Context) overrode Static & Dynamic: got %d (expected 303)\n", draft_res[0]);

    /* Case B: If not in Context, Tier 1 (Static) must win over Tier 3 (Dynamic)
     * For [400, 500, 600]: not in Context, present in Static (yielding 602) */
    int32_t test_histB[] = {400, 500};
    d_count = ds4_ngram_draft_3tier(&context_tbl, &static_tbl, &dynamic_tbl,
                                    test_histB, 2, 600, draft_res, 1);
    assert(d_count == 1);
    assert(draft_res[0] == 602);
    printf("    [PASS] Tier 1 (Static) overrode Dynamic: got %d (expected 602)\n", draft_res[0]);

    /* Case C: If not in Context and not in Static, fallback to Tier 3 (Dynamic) */
    int32_t dyn_only[] = {700, 800, 900, 903};
    ds4_ngram_table_update(&dynamic_tbl, dyn_only, 4, 4);
    int32_t test_histC[] = {700, 800};
    d_count = ds4_ngram_draft_3tier(&context_tbl, &static_tbl, &dynamic_tbl,
                                    test_histC, 2, 900, draft_res, 1);
    assert(d_count == 1);
    assert(draft_res[0] == 903);
    printf("    [PASS] Tier 3 (Dynamic) fallback succeeded: got %d (expected 903)\n", draft_res[0]);

    ds4_ngram_table_free(&static_tbl);
    ds4_ngram_table_free(&context_tbl);
    ds4_ngram_table_free(&dynamic_tbl);
    remove(tmp_path);

    printf("ALL N-GRAM CACHE TESTS PASSED!\n");
    return 0;
}
