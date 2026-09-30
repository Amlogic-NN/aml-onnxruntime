# nlohmann JSON

This directory vendors the single-header nlohmann JSON 3.12.0 dependency used
by the standalone ADLA bridge tools. The embedded bridge uses ONNX Runtime's
existing `nlohmann_json::nlohmann_json` dependency instead.

- Upstream: https://github.com/nlohmann/json
- Header: `single_include/nlohmann/json.hpp`
- License: MIT (`LICENSE.MIT`)
- SHA-256: `AAF127C04CB31C406E5B04A63F1AE89369FCCDE6D8FA7CDDA1ED4F32DFC5DE63`

The bridge public API intentionally uses `std::string`, not nlohmann JSON
types, so this third-party dependency does not cross the library ABI boundary.
