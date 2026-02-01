#ifndef ENGINEGEN_PLUGIN_ABI_H
#define ENGINEGEN_PLUGIN_ABI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * EngineGen native plugin ABI (v1)
 *
 * All inputs/outputs are UTF-8 JSON.
 * - enginegen_plugin_descriptor returns a JSON string describing the plugin:
 *   {
 *     "name": "plugin_name",
 *     "kind": "synthesizer|geometry_backend|adapter|analysis|optimization",
 *     "api_version": "1.0.0",
 *     "plugin_version": "0.1.0",
 *     "capabilities": { ... }
 *   }
 * - enginegen_plugin_call dispatches by method name with a JSON payload.
 *   It returns 0 on success and non-zero on error. On error, the output buffer
 *   should contain a UTF-8 JSON error message or a string message.
 * - enginegen_plugin_free must release buffers returned by call/descriptor.
 */

const char *enginegen_plugin_descriptor(void);

int enginegen_plugin_call(
    const char *method,
    const void *input,
    size_t input_len,
    void **output,
    size_t *output_len
);

void enginegen_plugin_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* ENGINEGEN_PLUGIN_ABI_H */
