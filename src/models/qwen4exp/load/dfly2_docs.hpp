// src/models/qwen4exp/load/dfly2_docs.hpp - the teacher export's documents file: one document a line, as token ids
// separated by spaces or commas. Line k is sequence k of the export. Host only.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::models::qwen4exp::generate {

/// The documents file, a line at a time: a document may be longer than any fixed buffer.
class DocsFile {
public:
    DocsFile() = default;
    DocsFile(const DocsFile&) = delete;
    DocsFile& operator=(const DocsFile&) = delete;
    ~DocsFile();

    /// Opens `path` (UTF-8) at its first line; false when it cannot be read.
    bool open(const std::string& path);
    /// The next line without its end; false at the end of the file, or when a read failed (`failed`).
    bool next(std::string& line);
    bool failed() const { return failed_; }

private:
    std::FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t at_ = 0, got_ = 0;
    bool done_ = false, failed_ = false;
};

/// A line of the documents file as a document: at least one token id, each below `n_vocab`. False with `err`.
bool document_parse(const std::string& line, int64_t n_vocab, std::vector<int64_t>& doc, std::string& err);

/// Reads the whole of `path` once, before anything is computed from it: every line must be a document of a
/// vocabulary of `n_vocab` tokens. `lines` gets their count. False with `err`, which names the file and the line.
bool documents_check(const std::string& path, int64_t n_vocab, uint64_t& lines, std::string& err);

}  // namespace basalt::models::qwen4exp::generate
