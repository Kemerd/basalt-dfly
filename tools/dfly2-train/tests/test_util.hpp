// tools/dfly2-train/tests/test_util.hpp - what every trainer test shares: the check line and a scratch directory.
#pragma once

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

namespace dfly2_test {

inline int g_fail = 0;

/// Prints one result line and counts a failure.
inline void check(bool ok, const std::string& what) {
    std::printf("  %-86s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// Runs a test's body: its lines appear as they are printed, and an exception is a failed test with its message.
template <class Body> int run(const char* name, Body&& body) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return body();
    } catch (const std::exception& e) {
        std::printf("%s: exception: %s\n", name, e.what());
        return 1;
    }
}

/// The test's exit code, after a summary line.
inline int finish(const char* name) {
    std::printf("%s: %s\n", name, g_fail == 0 ? "all checks passed" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}

/// A directory under the system's temp directory, removed with the object.
struct TempDir {
    std::filesystem::path path;
    explicit TempDir(const std::string& stem) {
        path = std::filesystem::temp_directory_path() /
               (stem + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    /// A file inside it, as the UTF-8 string the readers take.
    std::string file(const std::string& name) const {
        const std::u8string u = (path / name).u8string();
        return std::string(u.begin(), u.end());
    }
};

}  // namespace dfly2_test
