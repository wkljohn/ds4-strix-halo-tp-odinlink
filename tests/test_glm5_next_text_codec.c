#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tokens_equal(const ds4_tokens *tokens, const int *expected,
                        size_t n_expected) {
    return tokens && tokens->len == (int)n_expected &&
           memcmp(tokens->v, expected, n_expected * sizeof(expected[0])) == 0;
}

int main(void) {
    const char *model = getenv("DS4_GLM5_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL DS4_GLM5_MODEL is required\n");
        return 1;
    }

    ds4_glm5_next_text_codec *codec = NULL;
    ds4_tokens tokens = {0};
    if (ds4_glm5_next_text_codec_open(&codec, model) != 0 || !codec) {
        fprintf(stderr, "FAIL open glm5-next text codec\n");
        return 1;
    }
    ds4_glm5_next_text_codec_encode_chat(
        codec, NULL, "hi", DS4_THINK_MAX, &tokens);
    static const int hi_ids[] = {
        154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487,
        154827, 6023, 154828, 154841,
    };

    const char *expected =
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>hi<|assistant|><think>";
    size_t cap = strlen(expected) + 1u;
    char *rendered = calloc(cap, 1u);
    size_t used = 0u;
    if (!rendered) return 1;
    for (int i = 0; i < tokens.len; ++i) {
        size_t n = 0u;
        char *piece = ds4_glm5_next_text_codec_token_text(
            codec, tokens.v[i], &n);
        if (!piece || n > cap - used - 1u) {
            free(piece);
            free(rendered);
            ds4_tokens_free(&tokens);
            ds4_glm5_next_text_codec_close(codec);
            fprintf(stderr, "FAIL invalid rendered token piece\n");
            return 1;
        }
        memcpy(rendered + used, piece, n);
        used += n;
        free(piece);
    }

    const int stop_ok =
        ds4_glm5_next_text_codec_token_is_stop(codec, 154820) &&
        ds4_glm5_next_text_codec_token_is_stop(codec, 154827) &&
        ds4_glm5_next_text_codec_token_is_stop(codec, 154828) &&
        ds4_glm5_next_text_codec_token_is_stop(codec, 154829) &&
        !ds4_glm5_next_text_codec_token_is_stop(codec, 154841) &&
        !ds4_glm5_next_text_codec_token_is_stop(codec, 6023);
    const int hi_ok = strcmp(rendered, expected) == 0 &&
                      tokens_equal(&tokens, hi_ids,
                                   sizeof(hi_ids) / sizeof(hi_ids[0]));
    ds4_tokens_free(&tokens);

    static const int contraction_ids[] = {
        154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487,
        154827, 2132, 594, 6915, 13, 154828, 154841,
    };
    ds4_glm5_next_text_codec_encode_chat(
        codec, NULL, "It's fine.", DS4_THINK_MAX, &tokens);
    const int contraction_ok = tokens_equal(
        &tokens, contraction_ids,
        sizeof(contraction_ids) / sizeof(contraction_ids[0]));
    ds4_tokens_free(&tokens);

    static const int cjk_ids[] = {
        154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487,
        154827, 109377, 3837, 99011, 154828, 154841,
    };
    ds4_glm5_next_text_codec_encode_chat(
        codec, NULL, "你好，世界", DS4_THINK_MAX, &tokens);
    const int cjk_ok = tokens_equal(
        &tokens, cjk_ids, sizeof(cjk_ids) / sizeof(cjk_ids[0]));
    ds4_tokens_free(&tokens);

    static const char diverse_prompt[] =
        "Name the chemical symbol for gold and the author of Pride and "
        "Prejudice. Answer in one sentence.";
    static const int diverse_ids[] = {
        154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487,
        154827, 675, 279, 11478, 7735, 369, 6623, 323, 279, 3150, 315,
        41907, 323, 4968, 18110, 558, 13, 21754, 304, 825, 11646, 13,
        154828, 154841,
    };
    ds4_glm5_next_text_codec_encode_chat(
        codec, NULL, diverse_prompt, DS4_THINK_MAX, &tokens);
    const int diverse_ok = tokens_equal(
        &tokens, diverse_ids, sizeof(diverse_ids) / sizeof(diverse_ids[0]));
    const int ok = hi_ok && contraction_ok && cjk_ok && diverse_ok && stop_ok;
    if (!ok) {
        fprintf(stderr,
                "FAIL rendered=%s hi=%d contraction=%d cjk=%d diverse=%d "
                "stop=%d final_tokens=%d\n",
                rendered, hi_ok, contraction_ok, cjk_ok, diverse_ok,
                stop_ok, tokens.len);
    } else {
        fprintf(stderr,
                "PASS GLM5 text codec official_ids hi=%zu contraction=%zu "
                "cjk=%zu diverse=%zu stop_markers=1\n",
                sizeof(hi_ids) / sizeof(hi_ids[0]),
                sizeof(contraction_ids) / sizeof(contraction_ids[0]),
                sizeof(cjk_ids) / sizeof(cjk_ids[0]),
                sizeof(diverse_ids) / sizeof(diverse_ids[0]));
    }

    free(rendered);
    ds4_tokens_free(&tokens);
    ds4_glm5_next_text_codec_close(codec);
    return ok ? 0 : 1;
}
