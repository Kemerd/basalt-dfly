// src/spec/scope_drafter.cpp - see include/basalt/spec/scope_drafter.hpp.
#include "basalt/spec/scope_drafter.hpp"

#include <algorithm>
#include <cstring>

namespace basalt::spec {

namespace {

constexpr uint32_t kNone = 0xFFFFFFFFu;        // no entry, scope or function
constexpr uint32_t kAnyScope = 0xFFFFFFFEu;    // the scope of the entries that tell which scopes have a name
constexpr uint32_t kAmbiguous = 0xFFFFFFFDu;   // such an entry's child when two scopes have the name
constexpr uint32_t kMaxColumn = 512;           // a list that starts further right is not realigned

// Tokens: a punctuation mark is its own character.
constexpr uint8_t kWord = 1, kLiteral = 2, kScopeOp = 3, kArrow = 4, kOther = 5;

// What a word means at declaration level.
enum : int {
    kw_none, kw_ns, kw_cls, kw_enm, kw_alias, kw_skip, kw_tmpl, kw_static, kw_inline, kw_extern, kw_oper, kw_access,
    kw_drop,    // said in a declaration only: never a name, never repeated
    kw_trail,   // a qualifier a definition repeats after its parameter list
    kw_type     // any other word that is no name
};
struct Keyword {
    std::string_view text;
    int kw;
};
constexpr Keyword kKeywords[] = {
    {"namespace", kw_ns}, {"class", kw_cls}, {"struct", kw_cls}, {"union", kw_cls}, {"enum", kw_enm},
    {"using", kw_alias}, {"typedef", kw_alias}, {"friend", kw_skip}, {"static_assert", kw_skip},
    {"template", kw_tmpl}, {"static", kw_static}, {"inline", kw_inline}, {"extern", kw_extern},
    {"operator", kw_oper}, {"public", kw_access}, {"private", kw_access}, {"protected", kw_access},
    {"override", kw_drop}, {"final", kw_drop},
    {"const", kw_trail}, {"volatile", kw_trail}, {"noexcept", kw_trail},
    {"void", kw_type}, {"bool", kw_type}, {"char", kw_type}, {"short", kw_type}, {"int", kw_type}, {"long", kw_type},
    {"float", kw_type}, {"double", kw_type}, {"signed", kw_type}, {"unsigned", kw_type}, {"auto", kw_type},
    {"wchar_t", kw_type}, {"char8_t", kw_type}, {"char16_t", kw_type}, {"char32_t", kw_type},
    {"constexpr", kw_type}, {"consteval", kw_type}, {"constinit", kw_type}, {"virtual", kw_type},
    {"explicit", kw_type}, {"mutable", kw_type}, {"thread_local", kw_type}, {"typename", kw_type},
    {"decltype", kw_type}, {"alignas", kw_type}, {"alignof", kw_type}, {"sizeof", kw_type},
    {"__declspec", kw_type}, {"__attribute__", kw_type}, {"requires", kw_type}, {"throw", kw_type},
    {"try", kw_type}, {"if", kw_type}, {"else", kw_type}, {"for", kw_type}, {"while", kw_type}, {"do", kw_type},
    {"switch", kw_type}, {"case", kw_type}, {"default", kw_type}, {"return", kw_type}, {"new", kw_type},
    {"delete", kw_type}, {"goto", kw_type}, {"break", kw_type}, {"continue", kw_type}, {"this", kw_type},
    {"nullptr", kw_type}, {"true", kw_type}, {"false", kw_type},
};

int keyword(std::string_view w) {
    if (w.size() < 2 || w.size() > 13) return kw_none;
    for (const Keyword& k : kKeywords)
        if (k.text == w) return k.kw;
    return kw_none;
}

struct WordChars {
    bool is[256] = {};
    constexpr WordChars() {
        for (int c = '0'; c <= '9'; ++c) is[c] = true;
        for (int c = 'a'; c <= 'z'; ++c) is[c] = true;
        for (int c = 'A'; c <= 'Z'; ++c) is[c] = true;
        is[(unsigned char) '_'] = is[(unsigned char) '$'] = true;
    }
};
constexpr WordChars kWordChars;
bool word_char(char c) { return kWordChars.is[(unsigned char) c]; }
bool digit(char c) { return c >= '0' && c <= '9'; }

// A word right before a quote: 1 a literal's prefix, 2 a raw string's, 0 a word of its own.
int literal_prefix(const char* w, size_t n) {
    if (n == 1) return w[0] == 'R' ? 2 : (w[0] == 'L' || w[0] == 'u' || w[0] == 'U') ? 1 : 0;
    if (n == 2 && w[1] == 'R') return (w[0] == 'L' || w[0] == 'u' || w[0] == 'U') ? 2 : 0;
    if (n == 2) return w[0] == 'u' && w[1] == '8' ? 1 : 0;
    return n == 3 && w[0] == 'u' && w[1] == '8' && w[2] == 'R' ? 2 : 0;
}

uint64_t hash_of(uint32_t scope, std::string_view name) {
    uint64_t h = 0xcbf29ce484222325ull ^ ((uint64_t) scope * 0x9E3779B97F4A7C15ull);
    for (const unsigned char c : name) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h ^ (h >> 32);
}

bool ends_with(const std::string& s, std::string_view tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

void trim_right(std::string& s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.pop_back();
}

// A parameter list as a definition repeats it: `in` without its default arguments.
void strip_defaults(const std::string& in, std::string& out) {
    out.clear();
    const size_t n = in.size();
    int depth = 0, angle = 0;
    bool skipping = false;
    char before = 0;      // the last byte that is not blank
    for (size_t i = 0; i < n; ++i) {
        const char c = in[i];
        // a literal goes whole: nothing in it is structure
        if (c == '"' || (c == '\'' && !word_char(before))) {
            size_t j = i + 1;
            while (j < n && in[j] != c) j += in[j] == '\\' ? 2 : 1;
            j = std::min(j + 1, n);
            if (!skipping) out.append(in, i, j - i);
            i = j - 1;
            before = c;
            continue;
        }
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if ((c == ')' || c == ']' || c == '}') && depth > 0) --depth;
        if (skipping) {
            // the default ends at the comma that is not inside its own brackets
            if (c == '<' && word_char(before)) ++angle;
            else if (c == '>' && angle > 0 && before != '-') --angle;
            else if (c == ',' && depth == 0 && angle == 0) {
                skipping = false;
                out += c;
            }
        } else if (c == '=' && depth == 0) {
            skipping = true;
            angle = 0;
            trim_right(out);
        } else {
            out += c;
        }
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') before = c;
    }
}

}  // namespace

ScopeDrafter::ScopeDrafter(const Limits& limits) : limits_(limits) { reset(); }

void ScopeDrafter::reset() {
    entries_.clear();
    scopes_.clear();
    fns_.clear();
    slots_.clear();
    text_.clear();
    scopes_.push_back(Scope{kNone, kNone, kNone, Kind::ns});   // the file level
    total_ = 0;
    boundary();
}

void ScopeDrafter::boundary() {
    cut();
    col_ = indent_ = 0;
    line_ = Line::blank;
    first_ = false;
}

// Ends whatever is open: the scan goes on at file level, in code, with no text to look back over.
void ScopeDrafter::cut() {
    lex_ = Lex::code;
    id_len_ = 0;
    colon_ = minus_ = pp_cont_ = false;
    raw_n_ = raw_at_ = 0;
    depth_ = 1;
    frames_[0] = Frame{0, kNone, 0, Kind::ns};
    reset_statement();
    tail_n_ = 0;
}

void ScopeDrafter::reset_statement() {
    st_ = Stmt();
    cap_ = Cap::off;
    cap_over_ = false;
}

void ScopeDrafter::append(std::string_view text) {
    const char* p = text.data();
    const size_t n = text.size();
    if (p == nullptr || n == 0) return;
    size_t tail_from = 0;   // the look-back restarts behind a line that cut the scan
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = (unsigned char) p[i];
        if (track_line(c)) tail_from = i + 1;
        else lex(c);
    }
    total_ += n;
    keep_tail(p + tail_from, n - tail_from);
}

void ScopeDrafter::keep_tail(const char* p, size_t n) {
    if (n >= (size_t) kTail) {
        std::memcpy(tail_, p + n - kTail, kTail);
        tail_n_ = kTail;
        return;
    }
    // full: the newest kTail bytes move to the front, which leaves room for less than kTail more
    if (tail_n_ + n > sizeof tail_) {
        std::memmove(tail_, tail_ + tail_n_ - kTail, kTail);
        tail_n_ = kTail;
    }
    std::memcpy(tail_ + tail_n_, p, n);
    tail_n_ += (uint32_t) n;
}

// Follows the column and the line's indent, and spots the lines that are not code. True where `c` completes the
// opening of a fence or of a tag: the scan was cut and the rest of the line is skipped.
bool ScopeDrafter::track_line(unsigned char c) {
    if (c == '\n') {
        col_ = indent_ = 0;
        line_ = Line::blank;
        first_ = false;
        return false;
    }
    if (col_ != UINT32_MAX) ++col_;
    first_ = false;
    switch (line_) {
    case Line::blank:
        if (c == ' ' || c == '\t') {
            if (indent_ != UINT32_MAX) ++indent_;
            return false;
        }
        if (c == '\r') return false;
        first_ = true;
        line_ = c == '`' ? Line::tick1 : c == '<' ? Line::lt : Line::text;
        return false;
    case Line::tick1:
        line_ = c == '`' ? Line::tick2 : Line::text;
        return false;
    case Line::tick2:
        line_ = Line::text;
        if (c != '`') return false;
        break;
    case Line::lt:
        line_ = Line::text;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '/')) return false;
        break;
    case Line::text:
        return false;
    }
    cut();
    lex_ = Lex::skip_line;
    return true;
}

void ScopeDrafter::capture(unsigned char c) {
    if (cap_ == Cap::off) return;
    std::string& dst = cap_ == Cap::params ? params_ : quals_;
    if (dst.size() < (size_t) kMaxParams) dst += (char) c;
    else cap_over_ = true;
}

void ScopeDrafter::lex(unsigned char c) {
    switch (lex_) {
    case Lex::code:
        break;
    case Lex::slash:
        if (c == '/') { lex_ = Lex::line_comment; return; }
        if (c == '*') { lex_ = Lex::block_comment; return; }
        // a division: the slash was code after all, and so is `c`
        lex_ = Lex::code;
        capture('/');
        token(kOther);
        break;
    case Lex::line_comment:
    case Lex::skip_line:
        if (c == '\n') lex_ = Lex::code;
        return;
    case Lex::block_comment:
        if (c == '*') lex_ = Lex::block_star;
        return;
    case Lex::block_star:
        lex_ = c == '/' ? Lex::code : c == '*' ? Lex::block_star : Lex::block_comment;
        return;
    case Lex::string:
    case Lex::chr:
        // a literal ends at its quote, or at the end of the line when the quote is missing
        capture(c);
        if (c == '\\') lex_ = lex_ == Lex::string ? Lex::string_esc : Lex::chr_esc;
        else if (c == '\n' || c == (lex_ == Lex::string ? '"' : '\'')) {
            lex_ = Lex::code;
            token(kLiteral);
        }
        return;
    case Lex::string_esc:
    case Lex::chr_esc:
        capture(c);
        lex_ = lex_ == Lex::string_esc ? Lex::string : Lex::chr;
        return;
    case Lex::raw_delim:
        capture(c);
        if (c == '(') { lex_ = Lex::raw_body; return; }
        if (c == '"' || c == '\n') {   // no raw string after all
            lex_ = Lex::code;
            token(kLiteral);
            return;
        }
        if (raw_n_ < sizeof raw_) raw_[raw_n_++] = (char) c;
        else lex_ = Lex::string;       // no delimiter is that long
        return;
    case Lex::raw_body:
        capture(c);
        if (c == ')') {
            lex_ = Lex::raw_close;
            raw_at_ = 0;
        }
        return;
    case Lex::raw_close:
        // behind a `)`: the delimiter again, then the quote
        capture(c);
        if (raw_at_ == raw_n_ && c == '"') {
            lex_ = Lex::code;
            token(kLiteral);
        } else if (raw_at_ < raw_n_ && c == (unsigned char) raw_[raw_at_]) {
            ++raw_at_;
        } else if (c == ')') {
            raw_at_ = 0;
        } else {
            lex_ = Lex::raw_body;
        }
        return;
    case Lex::preproc:
        // a directive runs to the end of its line, and over a line that ends in a backslash
        if (c == '\n') {
            if (!pp_cont_) lex_ = Lex::code;
            pp_cont_ = false;
        } else if (c != '\r') {
            pp_cont_ = c == '\\';
        }
        return;
    }

    const bool was_minus = minus_;
    minus_ = false;
    if (colon_ && c != ':') {
        colon_ = false;
        token(':');
    }
    if (word_char((char) c)) {
        if (id_len_ < (uint32_t) kMaxName) id_[id_len_] = (char) c;
        if (id_len_ != UINT32_MAX) ++id_len_;
        capture(c);
        return;
    }
    if (id_len_ > 0) {
        if (c == '"' || c == '\'') {
            // a quote right behind a word: a prefixed literal, or an apostrophe inside the word
            const int prefix = literal_prefix(id_, id_len_);
            if (prefix == 2 && c == '"') {
                id_len_ = 0;
                capture(c);
                raw_n_ = 0;
                lex_ = Lex::raw_delim;
                return;
            }
            if (prefix == 1) {
                id_len_ = 0;
                capture(c);
                lex_ = c == '"' ? Lex::string : Lex::chr;
                return;
            }
            if (c == '\'') {   // a digit separator, or prose
                capture(c);
                return;
            }
        }
        flush_word();
    }
    if (c == '/') {   // code only if no comment follows
        lex_ = Lex::slash;
        return;
    }
    capture(c);
    switch (c) {
    case ' ': case '\t': case '\r': case '\n': case '\f': case '\v':
        return;
    case '"':
        lex_ = Lex::string;
        return;
    case '\'':
        lex_ = Lex::chr;
        return;
    case '#':
        if (first_) {
            lex_ = Lex::preproc;
            pp_cont_ = false;
            return;
        }
        token(kOther);
        return;
    case ':':
        if (colon_) {
            colon_ = false;
            token(kScopeOp);
        } else {
            colon_ = true;
        }
        return;
    case '-':
        minus_ = true;
        token(kOther);
        return;
    case '>':
        token(was_minus ? kArrow : (uint8_t) '>');
        return;
    case '{': case '}': case '(': case ')': case '[': case ']': case '<': case ';': case ',': case '=': case '~':
    case '&':
        token(c);
        return;
    default:
        token(kOther);
        return;
    }
}

void ScopeDrafter::flush_word() {
    const size_t len = id_len_;
    id_len_ = 0;
    const Frame& f = top();
    if (f.nested > 0 || f.kind == Kind::block) return;   // inside a body nothing is declared
    if (digit(id_[0])) {
        token(kOther);
        return;
    }
    if (f.kind == Kind::enm) {
        enum_word(len);
        return;
    }
    statement_word(len);
    st_.prev2 = st_.prev;
    st_.prev = kWord;
}

void ScopeDrafter::token(uint8_t t) {
    if (t == '{') { open_brace(); return; }
    if (t == '}') { close_brace(); return; }
    const Frame& f = top();
    if (f.nested > 0 || f.kind == Kind::block) return;
    if (f.kind == Kind::enm) {
        enum_token(t);
        return;
    }
    statement_token(t);
    st_.prev2 = st_.prev;
    st_.prev = t;
}

// A brace at declaration level opens what the statement before it names: a scope, or a body.
void ScopeDrafter::open_brace() {
    Frame& f = top();
    if (f.nested > 0 || f.kind == Kind::block || f.kind == Kind::enm) {
        if (f.nested != UINT32_MAX) ++f.nested;
        return;
    }
    Stmt& s = st_;
    if (s.group > 0) {   // inside the list being captured (a default argument): part of the list
        if (s.gbrace != UINT16_MAX) ++s.gbrace;
        return;
    }
    if (s.ret) end_return();
    Frame nf;
    nf.scope = f.scope;
    nf.also = kNone;
    nf.kind = Kind::block;
    const bool plain = s.paren == 0 && s.angle == 0 && s.group == 0 && !s.init && !s.ignore;
    if (s.has_fn) {
        // a function's body: its declaration is complete, and a member's body looks names up from its class
        if (!s.in_params) {
            declarator();
            if (s.fn_qualified && fn_scope_ != kNone) nf.scope = fn_scope_;
        }
    } else if (plain && s.head == Head::ns) {
        // an inline or unnamed namespace declares into the one around it
        nf.kind = Kind::ns;
        if (!s.is_inline) {
            uint32_t sc = s.chain_over ? kNone : f.scope;
            size_t at = 0;
            for (int i = 0; i < s.chain_n; ++i) {
                sc = child_scope(sc, std::string_view(chain_ + at, chain_seg_[i]), Kind::ns);
                at += chain_seg_[i];
            }
            if (s.name_len > 0) sc = child_scope(sc, name(), Kind::ns);
            nf.scope = sc;
        }
    } else if (plain && s.head == Head::cls) {
        nf.kind = Kind::cls;
        if (s.name_len > 0) nf.scope = child_scope(s.qualified ? resolve_chain(f.scope) : f.scope, name(), Kind::cls);
    } else if (plain && s.head == Head::enm) {
        nf.kind = Kind::enm;
        nf.scope = s.name_len > 0 ? child_scope(f.scope, name(), Kind::enm) : kNone;
        if (!s.enum_scoped) nf.also = f.scope;
    } else if (plain && s.is_extern && s.literal && s.head == Head::none && s.name_len == 0) {
        nf.kind = Kind::ns;   // a linkage block: the scope around it goes on
    } else if (s.head == Head::none && !s.ignore && !s.base) {
        declarator();         // a variable with a braced initializer
    }
    if (s.over) nf.scope = nf.also = kNone;   // a scope whose name is not kept keeps no names
    const bool pushed = depth_ < kMaxDepth;
    if (pushed) frames_[depth_++] = nf;
    else if (f.nested != UINT32_MAX) ++f.nested;
    reset_statement();
    st_.expect = pushed && nf.kind == Kind::enm;
}

void ScopeDrafter::close_brace() {
    Frame& f = top();
    if (f.nested > 0) {
        if (--f.nested == 0 && (f.kind == Kind::ns || f.kind == Kind::cls)) reset_statement();
        return;
    }
    if (st_.gbrace > 0) {   // a brace of the list being captured
        --st_.gbrace;
        return;
    }
    if (depth_ > 1) --depth_;   // a brace too many closes nothing
    reset_statement();
}

void ScopeDrafter::statement_word(size_t len) {
    if (st_.group > 0 || st_.paren > 0) return;   // inside a list: captured as bytes, or not read
    const bool over = len > (size_t) kMaxName;
    const std::string_view w(id_, over ? (size_t) kMaxName : len);
    const int kw = over ? (int) kw_type : keyword(w);
    // a word that can only start a declaration: what was open before it was not one
    const bool starts = kw == kw_ns ||
                        ((kw == kw_cls || kw == kw_enm || kw == kw_alias || kw == kw_tmpl) && st_.angle == 0);
    if (starts && (st_.init || st_.ignore || st_.angle > 0 || (st_.base && kw == kw_ns))) reset_statement();
    Stmt& s = st_;
    if (s.angle > 0 || s.init || s.ignore || s.base || s.head == Head::skip) return;
    if (s.op) {   // the symbol of an operator function
        s.op_first = false;
        return;
    }
    if (s.after_params && trailing_word(w, kw)) return;
    switch (kw) {
    case kw_ns:
        s.head = s.head == Head::alias ? Head::skip : Head::ns;   // a using-directive declares nothing
        s.name_len = 0;
        break;
    case kw_cls:
        if (s.head == Head::enm) s.enum_scoped = true;
        else if (s.head == Head::none || s.head == Head::alias) s.head = Head::cls;
        s.name_len = 0;
        break;
    case kw_enm:
        if (s.head == Head::none || s.head == Head::alias) s.head = Head::enm;
        s.name_len = 0;
        break;
    case kw_alias:
        if (s.head == Head::none) s.head = Head::alias;
        break;
    case kw_skip:
        s.head = Head::skip;
        break;
    case kw_tmpl:
        s.tmpl = true;
        break;
    case kw_static:
        s.is_static = true;
        break;
    case kw_inline:
        s.is_inline = true;
        break;
    case kw_extern:
        s.is_extern = true;
        break;
    case kw_oper:
        set_name(w);
        s.op = s.op_first = true;
        break;
    case kw_access:
        s.access = true;
        break;
    case kw_drop:
        break;
    case kw_trail:
    case kw_type:
        s.over = over;
        s.name_len = 0;
        s.chain_n = 0;
        s.chain_len = 0;
        s.qualified = s.global = s.chain_over = false;
        break;
    default:
        set_name(w);
        break;
    }
}

// A word behind a parameter list. False when it is neither a qualifier nor a word that ends the declarator: the
// caller then reads it as any other word of the statement.
bool ScopeDrafter::trailing_word(std::string_view w, int kw) {
    Stmt& s = st_;
    if (s.ret) return true;   // inside a trailing return type: captured as bytes
    s.noexc = false;
    if (kw == kw_trail) {
        if (quals_.size() + w.size() + 1 > (size_t) kMaxParams) {
            cap_over_ = true;
            return true;
        }
        quals_ += ' ';
        quals_.append(w);
        s.noexc = w == "noexcept";
        return true;
    }
    if (kw == kw_drop || w == "try") return true;
    if (w == "requires" || w == "throw") {
        s.ignore = true;
        return true;
    }
    // any other word after the qualifiers is not read; with none yet, the list may have been a macro's ahead of
    // the declaration, and a later list replaces it
    if (quals_.empty()) return false;
    s.ignore = true;
    return true;
}

void ScopeDrafter::set_name(std::string_view w) {
    Stmt& s = st_;
    const bool tilde = s.prev == '~';
    if ((tilde ? s.prev2 : s.prev) == kScopeOp) {
        // the name before the `::` is a qualifier of this one
        if (s.name_len > 0) {
            if (s.chain_n < kMaxQualifiers && s.chain_len + s.name_len <= sizeof chain_) {
                std::memcpy(chain_ + s.chain_len, name_, s.name_len);
                chain_seg_[s.chain_n++] = s.name_len;
                s.chain_len = (uint16_t) (s.chain_len + s.name_len);
            } else {
                s.chain_over = true;
            }
        } else if (s.chain_n == 0) {
            s.global = true;
        }
        s.qualified = true;
    } else {
        s.chain_n = 0;
        s.chain_len = 0;
        s.qualified = s.global = s.chain_over = false;
    }
    s.over = false;
    size_t n = 0;
    if (tilde) name_[n++] = '~';
    const size_t len = std::min(w.size(), (size_t) kMaxName);
    std::memcpy(name_ + n, w.data(), len);
    s.name_len = (uint8_t) (n + len);
}

void ScopeDrafter::statement_token(uint8_t t) {
    Stmt& s = st_;
    // inside the list being captured only its parentheses count
    if (s.group > 0) {
        if (t == '(') {
            if (s.group != UINT16_MAX) ++s.group;
        } else if (t == ')') {
            if (--s.group == 0) end_group();
        } else if (t == ';') {
            // a list that never closed: the braces open in it are a body's
            const uint16_t braces = s.gbrace;
            reset_statement();
            top().nested = braces;
        }
        return;
    }
    if (t == ';') {
        if (s.ret) end_return();
        declarator();
        reset_statement();
        return;
    }
    if (s.paren > 0) {
        if (t == '(' || t == '[') {
            if (s.paren != UINT16_MAX) ++s.paren;
        } else if (t == ')' || t == ']') {
            --s.paren;
        }
        return;
    }
    if (s.op) {
        // an operator's symbol runs to its parameter list; `operator()` has a pair of its own ahead of it
        if (t != '(') {
            s.op_first = false;
        } else if (s.op_first) {
            s.op_first = false;
            s.paren = 1;
        } else {
            s.op = false;
            start_params();
        }
        return;
    }
    if (s.angle > 0 && t != '<' && t != '>' && t != '(' && t != '[') return;
    switch (t) {
    case '(':
        if (s.noexc && s.after_params && s.prev == kWord) {   // noexcept(...): kept as written
            s.noexc = false;
            if (quals_.size() < (size_t) kMaxParams) quals_ += '(';
            else cap_over_ = true;
            cap_ = Cap::quals;
            s.group = 1;
            return;
        }
        // a name and a parenthesis: a function, unless a list was read already and qualifiers followed it
        if (s.angle == 0 && !s.init && !s.base && !s.ignore && !s.ret && s.prev == kWord && s.name_len > 0 &&
            (s.head == Head::none || s.head == Head::cls || s.head == Head::enm) &&
            (!s.has_fn || (s.after_params && quals_.empty()))) {
            start_params();
            return;
        }
        s.paren = 1;
        return;
    case '[':
        s.paren = 1;
        return;
    case '<':
        if (s.tmpl) {
            s.tmpl = false;
            s.angle = 1;
        } else if (s.angle > 0) {
            if (s.angle != UINT16_MAX) ++s.angle;
        } else if (s.prev == kWord && !s.ignore && (!s.after_params || s.ret)) {
            s.angle = 1;   // behind a name: its template arguments
        }
        return;
    case '>':
        if (s.angle > 0) --s.angle;
        return;
    case ',':
        if (s.base || s.ignore || s.head == Head::skip) return;
        // the next declarator of the same declaration
        if (s.ret) end_return();
        declarator();
        s.name_len = 0;
        s.chain_n = 0;
        s.chain_len = 0;
        s.qualified = s.global = s.chain_over = false;
        s.init = s.has_fn = s.in_params = s.after_params = s.fn_qualified = s.noexc = false;
        s.fn_entry = kNone;
        cap_ = Cap::off;
        cap_over_ = false;
        return;
    case '=':
        if (s.ret) end_return();
        s.init = true;
        s.noexc = false;
        return;
    case ':':
        if (s.init || s.base || s.ignore) return;
        if (s.access) {
            reset_statement();   // an access specifier
            return;
        }
        // a base clause, an underlying type or a constructor's initializers; else a bit field or a label
        if (s.head == Head::cls || s.head == Head::enm || s.after_params) s.base = true;
        else s.ignore = true;
        return;
    case kScopeOp:
        if (s.prev != kWord && s.prev != '>') {   // no name ahead of it: the file level's
            s.name_len = 0;
            s.chain_n = 0;
            s.chain_len = 0;
            s.qualified = s.global = s.chain_over = false;
        }
        return;
    case kArrow:
        if (s.after_params && !s.ret && !s.ignore && !s.init && !s.base && quals_.size() + 3 <= (size_t) kMaxParams) {
            s.noexc = false;
            quals_ += " ->";
            cap_ = Cap::quals;
            s.ret = true;
        }
        return;
    case '&':
        if (s.after_params && !s.ret && !s.ignore && !s.init && !s.base && quals_.size() + 2 <= (size_t) kMaxParams)
            quals_ += s.prev == '&' ? "&" : " &";
        return;
    case kLiteral:
        if (s.is_extern && s.head == Head::none && s.name_len == 0 && !s.has_fn) s.literal = true;
        return;
    default:
        return;
    }
}

void ScopeDrafter::enum_word(size_t len) {
    Stmt& s = st_;
    if (!s.expect || s.init || s.paren > 0 || len > (size_t) kMaxName) return;
    const Frame f = top();
    const std::string_view w(id_, len);
    add_name(f.scope, w);
    add_name(f.also, w);
    s.expect = false;
}

void ScopeDrafter::enum_token(uint8_t t) {
    Stmt& s = st_;
    if (t == '(' || t == '[') {
        if (s.paren != UINT16_MAX) ++s.paren;
    } else if (t == ')' || t == ']') {
        if (s.paren > 0) --s.paren;
    } else if (s.paren == 0 && t == ',') {
        s.init = false;
        s.expect = true;
    } else if (s.paren == 0 && t == '=') {
        s.init = true;
    }
}

// The name before this `(` is a function's: its list is captured from here, and the declarations it may repeat
// are looked up while the name's scope is at hand.
void ScopeDrafter::start_params() {
    Stmt& s = st_;
    s.has_fn = s.in_params = true;
    s.after_params = s.noexc = false;
    s.group = 1;
    s.fn_qualified = s.qualified;
    fn_name_len_ = s.name_len;
    std::memcpy(fn_name_, name_, s.name_len);
    params_.clear();
    quals_.clear();
    cap_ = Cap::params;
    cap_over_ = false;
    fn_col_ = std::min(col_, (uint32_t) UINT16_MAX);
    fn_indent_ = std::min(indent_, (uint32_t) UINT16_MAX);
    const Frame& f = top();
    fn_scope_ = s.qualified ? resolve_chain(f.scope) : f.kind == Kind::ns ? f.scope : kNone;
    s.fn_entry = fn_scope_ != kNone ? find(fn_scope_, fn_name()) : kNone;
}

void ScopeDrafter::end_group() {
    if (cap_ == Cap::params) {
        if (!cap_over_ && !params_.empty()) params_.pop_back();   // the `)` that closed it
        st_.in_params = false;
        st_.after_params = true;
    }
    cap_ = Cap::off;
}

void ScopeDrafter::end_return() {
    st_.ret = false;
    cap_ = Cap::off;
    if (!cap_over_ && !quals_.empty()) quals_.pop_back();   // the byte that ended it
    // a definition does not repeat these
    for (;;) {
        trim_right(quals_);
        if (ends_with(quals_, " override")) quals_.resize(quals_.size() - 9);
        else if (ends_with(quals_, " final")) quals_.resize(quals_.size() - 6);
        else break;
    }
}

// What the statement so far declares goes into the scope it is in.
void ScopeDrafter::declarator() {
    const Stmt& s = st_;
    const Frame f = top();
    if (f.scope == kNone || s.head == Head::skip) return;
    if (s.has_fn) {
        // a member defined outside its class is declared in the class, not here
        if (s.in_params || cap_over_ || s.fn_qualified || s.head == Head::alias || fn_name_len_ == 0) return;
        if (fn_name() == "operator") add_name(f.scope, fn_name());
        else add_fn(f.scope, fn_name());
        return;
    }
    if (s.name_len == 0) return;
    // a variable: a namespace's, or a class's static one
    if (s.head == Head::none && (s.qualified || (f.kind == Kind::cls && !s.is_static))) return;
    add_name(f.scope, name());
}

uint32_t ScopeDrafter::find(uint32_t scope, std::string_view name) const {
    if (slots_.empty() || scope == kNone) return kNone;
    const size_t mask = slots_.size() - 1;
    for (size_t i = (size_t) hash_of(scope, name) & mask;; i = (i + 1) & mask) {
        const uint32_t v = slots_[i];
        if (v == 0) return kNone;   // the table is never more than half full
        const Entry& e = entries_[v - 1];
        if (e.scope == scope && e.len == name.size() && std::memcmp(text_.data() + e.text, name.data(), name.size()) == 0)
            return v - 1;
    }
}

void ScopeDrafter::grow_slots() {
    const size_t size = std::max<size_t>(64, slots_.size() * 2);
    slots_.assign(size, 0);
    const size_t mask = size - 1;
    for (size_t e = 0; e < entries_.size(); ++e) {
        const std::string_view name(text_.data() + entries_[e].text, entries_[e].len);
        size_t i = (size_t) hash_of(entries_[e].scope, name) & mask;
        while (slots_[i] != 0) i = (i + 1) & mask;
        slots_[i] = (uint32_t) e + 1;
    }
}

uint32_t ScopeDrafter::add_name(uint32_t scope, std::string_view name) {
    if (scope == kNone || name.empty() || name.size() > (size_t) kMaxName + 1) return kNone;
    if (const uint32_t e = find(scope, name); e != kNone) return e;
    if (entries_.size() >= limits_.names || text_.size() + name.size() > limits_.text) return kNone;
    if ((entries_.size() + 1) * 2 > slots_.size()) grow_slots();
    Entry e;
    e.text = (uint32_t) text_.size();
    e.len = (uint16_t) name.size();
    e.scope = scope;
    e.next = e.child = e.fn = kNone;
    const uint32_t idx = (uint32_t) entries_.size();
    const size_t mask = slots_.size() - 1;
    size_t i = (size_t) hash_of(scope, name) & mask;
    while (slots_[i] != 0) i = (i + 1) & mask;
    slots_[i] = idx + 1;
    text_.append(name);
    if (scope < scopes_.size()) {   // the newest name of its scope
        e.next = scopes_[scope].first;
        scopes_[scope].first = idx;
    }
    entries_.push_back(e);
    return idx;
}

uint32_t ScopeDrafter::child_scope(uint32_t parent, std::string_view name, Kind kind) {
    const uint32_t e = add_name(parent, name);
    if (e == kNone) return kNone;
    if (entries_[e].child != kNone) return entries_[e].child;
    if (scopes_.size() >= limits_.names) return kNone;
    const uint32_t id = (uint32_t) scopes_.size();
    scopes_.push_back(Scope{parent, e, kNone, kind});
    entries_[e].child = id;
    // which scope has this name, for a lookup from outside its parent; a second one makes the name ambiguous
    const uint32_t any = add_name(kAnyScope, name);
    if (any != kNone) entries_[any].child = entries_[any].child == kNone ? id : kAmbiguous;
    return id;
}

// The function read last (fn_name_, params_, quals_) as a declaration of `name` in `scope`.
void ScopeDrafter::add_fn(uint32_t scope, std::string_view name) {
    const uint32_t e = add_name(scope, name);
    if (e == kNone) return;
    strip_defaults(params_, scratch_);
    // a declaration read before (the header twice, or a definition behind its declaration) is not kept again
    uint32_t last = kNone;
    int n = 0;
    for (uint32_t i = entries_[e].fn; i != kNone; i = fns_[i].next, ++n) {
        const Fn& f = fns_[i];
        if (f.params_len == scratch_.size() && f.quals_len == quals_.size() &&
            std::memcmp(text_.data() + f.text, scratch_.data(), scratch_.size()) == 0 &&
            std::memcmp(text_.data() + f.text + f.params_len, quals_.data(), quals_.size()) == 0)
            return;
        last = i;
    }
    if (n >= kMaxOverloads || fns_.size() >= limits_.functions ||
        text_.size() + scratch_.size() + quals_.size() > limits_.text)
        return;
    Fn f;
    f.text = (uint32_t) text_.size();
    f.next = kNone;
    f.params_len = (uint16_t) scratch_.size();
    f.quals_len = (uint16_t) quals_.size();
    f.indent = (uint16_t) fn_indent_;
    f.col = (uint16_t) fn_col_;
    text_.append(scratch_);
    text_.append(quals_);
    const uint32_t idx = (uint32_t) fns_.size();
    fns_.push_back(f);
    if (last == kNone) entries_[e].fn = idx;
    else fns_[last].next = idx;
}

// The scope `name` names as seen from `from`: the nearest one outwards, else the only one of that name anywhere.
uint32_t ScopeDrafter::lookup_scope(uint32_t from, std::string_view name) const {
    for (uint32_t s = from < scopes_.size() ? from : 0; s < scopes_.size(); s = scopes_[s].parent) {
        const uint32_t e = find(s, name);
        if (e != kNone && entries_[e].child < scopes_.size()) return entries_[e].child;
    }
    const uint32_t any = find(kAnyScope, name);
    return any != kNone && entries_[any].child < scopes_.size() ? entries_[any].child : kNone;
}

// The scope `quals[0]::...::quals[n - 1]` names (outermost first), or kNone.
uint32_t ScopeDrafter::resolve(const std::string_view* quals, int n, bool global, uint32_t from) const {
    if (n <= 0) return kNone;
    uint32_t s = kNone;
    if (global) {
        const uint32_t e = find(0, quals[0]);
        if (e != kNone) s = entries_[e].child;
    } else {
        s = lookup_scope(from, quals[0]);
    }
    for (int i = 1; i < n && s < scopes_.size(); ++i) {
        const uint32_t e = find(s, quals[i]);
        s = e != kNone ? entries_[e].child : kNone;
    }
    return s < scopes_.size() ? s : kNone;
}

// The scope the qualifiers of the statement's name stand for.
uint32_t ScopeDrafter::resolve_chain(uint32_t from) const {
    const Stmt& s = st_;
    if (s.chain_over) return kNone;
    if (s.chain_n == 0) return s.global ? 0 : kNone;
    std::string_view quals[kMaxQualifiers];
    size_t at = 0;
    for (int i = 0; i < s.chain_n; ++i) {
        quals[i] = std::string_view(chain_ + at, chain_seg_[i]);
        at += chain_seg_[i];
    }
    return resolve(quals, s.chain_n, s.global, from);
}

// The end of the text as `A::B::typed`: the start of a name and the scopes it is qualified with (outermost first).
// False when the text does not end so.
bool ScopeDrafter::tail_name(std::string_view& typed, std::string_view* quals, int& n, bool& global) const {
    const char* t = tail_;
    const size_t end = tail_n_;
    const size_t begin = end > (size_t) kTail ? end - kTail : 0;   // the same reach however the text was split
    size_t i = end;
    while (i > begin && word_char(t[i - 1])) --i;
    if (end - i > (size_t) kMaxName || (i < end && digit(t[i]))) return false;
    if (i > begin && t[i - 1] == '~') --i;
    typed = std::string_view(t + i, end - i);
    if (i < begin + 2 || t[i - 1] != ':' || t[i - 2] != ':') return false;
    i -= 2;
    std::string_view inner[kMaxQualifiers];   // innermost first
    int k = 0;
    global = false;
    for (;;) {
        // the arguments of a class template's name are stepped over
        if (i > begin && t[i - 1] == '>') {
            int open = 0;
            bool matched = false;
            while (i > begin && !matched) {
                const char c = t[--i];
                if (c == '>') ++open;
                else if (c == '<') matched = --open == 0;
                else if (c == ';' || c == '{' || c == '}') return false;
            }
            if (!matched) return false;
        }
        size_t j = i;
        while (j > begin && word_char(t[j - 1])) --j;
        if (j == i) {   // `::` with no name ahead of it: the file level
            if (k == 0) return false;
            global = true;
            break;
        }
        if (i - j > (size_t) kMaxName || digit(t[j]) || k == kMaxQualifiers) return false;
        inner[k++] = std::string_view(t + j, i - j);
        i = j;
        if (i < begin + 2 || t[i - 1] != ':' || t[i - 2] != ':') break;
        i -= 2;
    }
    n = k;
    for (int q = 0; q < k; ++q) quals[q] = inner[k - 1 - q];
    return true;
}

// `f` as a definition writes it from the column `col` on a line indented by `indent`: its list, `)` and its
// qualifiers, at most about `limit` bytes. A continuation line keeps its place relative to the parenthesis when it
// was aligned to it, else relative to the line the list starts on.
void ScopeDrafter::signature(const Fn& f, uint32_t col, uint32_t indent, size_t limit, std::string& out) const {
    const char* p = text_.data() + f.text;
    const size_t n = f.params_len;
    const bool realign = col <= kMaxColumn && f.col <= kMaxColumn;
    for (size_t i = 0; i < n && out.size() < limit; ++i) {
        out += p[i];
        if (p[i] != '\n' || !realign) continue;
        size_t k = i + 1;
        while (k < n && p[k] == ' ') ++k;
        if (k < n && (p[k] == '\t' || p[k] == '\n' || p[k] == '\r')) continue;   // tabs or an empty line: as written
        const size_t blanks = k - (i + 1);
        size_t moved = 0;
        if (blanks >= f.col) moved = blanks - f.col + col;
        else if (blanks + indent >= f.indent) moved = blanks + indent - f.indent;
        out.append(std::min<size_t>(moved, 2 * kMaxColumn), ' ');
        i = k - 1;
    }
    if (out.size() >= limit) return;
    out += ')';
    out.append(p + n, f.quals_len);
}

// What the declarations from `fn` on have in common behind `typed`, the start of the list as written so far.
void ScopeDrafter::common_signature(uint32_t fn, uint32_t col, uint32_t indent, std::string_view typed, size_t limit,
                                    std::string& out) const {
    out.clear();
    std::string sig;
    bool have = false;
    for (uint32_t i = fn; i < fns_.size(); i = fns_[i].next) {
        sig.clear();
        signature(fns_[i], col, indent, typed.size() + limit, sig);
        if (sig.size() < typed.size() || std::memcmp(sig.data(), typed.data(), typed.size()) != 0) continue;
        const std::string_view rest = std::string_view(sig).substr(typed.size());
        if (!have) {
            out.assign(rest);
            have = true;
            continue;
        }
        size_t k = 0;
        while (k < out.size() && k < rest.size() && out[k] == rest[k]) ++k;
        out.resize(k);
    }
}

ScopeDrafter::Proposal ScopeDrafter::propose(size_t max_bytes, std::string& out) const {
    out.clear();
    Proposal p;
    if (max_bytes == 0 || lex_ != Lex::code) return p;
    max_bytes = std::min<size_t>(max_bytes, UINT16_MAX);
    const Frame& f = top();
    const Stmt& s = st_;
    const bool level = f.nested == 0 && (f.kind == Kind::ns || f.kind == Kind::cls);
    if (level && s.in_params) {
        // inside the parameter list of a function declared before: the rest of it
        if (cap_over_ || s.fn_entry >= entries_.size()) return p;
        common_signature(entries_[s.fn_entry].fn, fn_col_, fn_indent_, params_, max_bytes, out);
        if (out.size() > max_bytes) out.resize(max_bytes);
        p.signature = (uint32_t) out.size();
        return p;
    }
    std::string_view typed, quals[kMaxQualifiers];
    int n = 0;
    bool global = false;
    if (!tail_name(typed, quals, n, global)) return p;
    const uint32_t scope = resolve(quals, n, global, f.scope);
    if (scope == kNone) return p;
    // the names of the scope that start as typed: what they all go on with
    const char* first = nullptr;
    size_t common = 0;
    uint32_t only = kNone;
    int count = 0;
    for (uint32_t e = scopes_[scope].first; e < entries_.size(); e = entries_[e].next) {
        const Entry& en = entries_[e];
        if (en.len < typed.size() || std::memcmp(text_.data() + en.text, typed.data(), typed.size()) != 0) continue;
        const char* rest = text_.data() + en.text + typed.size();
        const size_t rest_n = en.len - typed.size();
        if (count++ == 0) {
            first = rest;
            common = rest_n;
            only = e;
            continue;
        }
        only = kNone;
        size_t k = 0;
        while (k < common && k < rest_n && first[k] == rest[k]) ++k;
        common = k;
        if (common == 0) return p;
    }
    if (count == 0) return p;
    out.assign(first, common);
    size_t name_bytes = out.size();
    if (only != kNone) {
        // one name is left: what follows a function's or a namespace's is known too
        const Entry& en = entries_[only];
        if (en.fn != kNone) {
            out += '(';
            name_bytes = out.size();
            // at declaration level this is the function's definition
            if (level && s.head == Head::none && !s.has_fn && !s.init && !s.base && !s.ignore && !s.op && s.paren == 0 &&
                s.angle == 0 && out.size() < max_bytes) {
                std::string sig;
                const uint32_t col = (uint32_t) std::min<size_t>((size_t) col_ + out.size(), UINT16_MAX);
                common_signature(en.fn, col, std::min(indent_, (uint32_t) UINT16_MAX), std::string_view(),
                                 max_bytes - out.size(), sig);
                out += sig;
            }
        } else if (en.child < scopes_.size() && scopes_[en.child].kind == Kind::ns) {
            out += "::";
            name_bytes = out.size();
        }
    }
    if (out.size() > max_bytes) out.resize(max_bytes);
    p.name = (uint32_t) std::min(name_bytes, out.size());
    p.signature = (uint32_t) (out.size() - p.name);
    return p;
}

std::string ScopeDrafter::dump() const {
    std::string out;
    std::vector<uint32_t> order;
    for (size_t sc = 0; sc < scopes_.size(); ++sc) {
        // the qualified name, outermost scope first
        order.clear();
        for (uint32_t s = (uint32_t) sc; s < scopes_.size() && scopes_[s].entry != kNone; s = scopes_[s].parent)
            order.push_back(scopes_[s].entry);
        if (order.empty()) out += "::";
        for (size_t i = order.size(); i-- > 0;) {
            out.append(text_, entries_[order[i]].text, entries_[order[i]].len);
            if (i > 0) out += "::";
        }
        const Kind kind = scopes_[sc].kind;
        out += kind == Kind::ns ? " namespace\n" : kind == Kind::cls ? " class\n" : " enum\n";
        // its names, oldest first
        order.clear();
        for (uint32_t e = scopes_[sc].first; e < entries_.size(); e = entries_[e].next) order.push_back(e);
        for (size_t i = order.size(); i-- > 0;) {
            const Entry& en = entries_[order[i]];
            if (en.fn == kNone) {
                out += "  ";
                out.append(text_, en.text, en.len);
                out += '\n';
            }
            for (uint32_t fi = en.fn; fi < fns_.size(); fi = fns_[fi].next) {
                const Fn& fn = fns_[fi];
                out += "  ";
                out.append(text_, en.text, en.len);
                out += '(';
                for (size_t b = 0; b < fn.params_len; ++b) {
                    const char c = text_[fn.text + b];
                    if (c == '\n') out += "\\n";
                    else out += c;
                }
                out += ')';
                out.append(text_, fn.text + fn.params_len, fn.quals_len);
                out += '\n';
            }
        }
    }
    return out;
}

}  // namespace basalt::spec
