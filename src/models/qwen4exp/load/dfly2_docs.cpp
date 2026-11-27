// src/models/qwen4exp/load/dfly2_docs.cpp - see dfly2_docs.hpp.
#include "dfly2_docs.hpp"

#include "../../../program/cli/tokens.hpp"
#include "basalt/platform/file.hpp"

#include <cstring>

namespace basalt::models::qwen4exp::generate {

namespace {
constexpr size_t kReadBytes = 1u << 20;   // the file is read this much at a time
}  // namespace

DocsFile::~DocsFile() {
    if (f_ != nullptr) std::fclose(f_);
}

bool DocsFile::open(const std::string& path) {
    if (f_ != nullptr) std::fclose(f_);
    f_ = basalt::platform::open_file_utf8(path, "rb");
    buf_.resize(kReadBytes);
    at_ = got_ = 0;
    done_ = failed_ = false;
    return f_ != nullptr;
}

bool DocsFile::next(std::string& line) {
    line.clear();
    for (;;) {
        if (at_ == got_) {
            if (f_ == nullptr || done_) break;
            got_ = std::fread(buf_.data(), 1, buf_.size(), f_);
            at_ = 0;
            if (got_ < buf_.size()) {
                done_ = true;
                failed_ = std::ferror(f_) != 0;
            }
            if (got_ == 0) break;
        }
        const char* from = buf_.data() + at_;
        const char* nl = (const char*) std::memchr(from, '\n', got_ - at_);
        if (nl == nullptr) {   // the line goes on in the next read
            line.append(from, got_ - at_);
            at_ = got_;
            continue;
        }
        line.append(from, (size_t) (nl - from));
        at_ += (size_t) (nl - from) + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    }
    // the last line may end without a newline
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return !failed_ && !line.empty();
}

bool document_parse(const std::string& line, int64_t n_vocab, std::vector<int64_t>& doc, std::string& err) {
    doc.clear();
    if (n_vocab < 1) {
        err = "the vocabulary is empty";
        return false;
    }
    // the list's parser reads a C string: a NUL would end the line early
    if (line.find('\0') != std::string::npos) {
        err = "not a list of token ids";
        return false;
    }
    if (!basalt::program::cli::parse_i64_list(line.c_str(), doc, err)) {
        if (err.empty()) err = "not a list of token ids";
        return false;
    }
    for (const int64_t id : doc)
        if (id < 0 || id >= n_vocab) {
            err = "token " + std::to_string(id) + " is not in the vocabulary of " + std::to_string(n_vocab);
            doc.clear();
            return false;
        }
    return true;
}

bool documents_check(const std::string& path, int64_t n_vocab, uint64_t& lines, std::string& err) {
    lines = 0;
    DocsFile docs;
    if (!docs.open(path)) {
        err = "cannot read " + path;
        return false;
    }
    std::string line, why;
    std::vector<int64_t> doc;
    while (docs.next(line)) {
        ++lines;
        if (!document_parse(line, n_vocab, doc, why)) {
            err = path + " line " + std::to_string(lines) + ": " + why;
            return false;
        }
    }
    if (docs.failed()) {
        err = "reading " + path + " failed";
        return false;
    }
    return true;
}

}  // namespace basalt::models::qwen4exp::generate
