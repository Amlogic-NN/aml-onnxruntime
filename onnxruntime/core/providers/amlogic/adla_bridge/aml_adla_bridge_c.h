#ifndef AML_ADLA_BRIDGE_C_H_
#define AML_ADLA_BRIDGE_C_H_

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compile an already-built Amlogic reference-style ops JSON document.
 *
 * The JSON is supplied in memory. The bridge writes the ADLA artifact below
 * output_dir and returns its path in adla_path. The API is stable for C,
 * C++, ctypes, and other FFI clients; no C++ allocation crosses the ABI.
 *
 * Return value: 0 on success, non-zero on failure.
 */
int aml_compile_ops_json_to_adla(const char* ops_json,
                                 const char* output_dir,
                                 const char* model_name,
                                 const char* target,
                                 char* adla_path,
                                 unsigned long adla_path_capacity,
                                 char* error_message,
                                 unsigned long error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif  /* AML_ADLA_BRIDGE_C_H_ */
