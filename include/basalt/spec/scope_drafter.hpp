// include/basalt/spec/scope_drafter.hpp - drafts from the C and C++ declarations the context holds (no weights, no GPU).
//
// Code written against declarations that are already in the context (a header in the prompt, or one written earlier
// in the reply) is known ahead in two places: after `Scope::` only a name declared in that scope can follow, and the
// parameter list of a function defined out of line repeats its declaration. A ScopeDrafter reads the text as it
// arrives and keeps the scopes (namespace, class, struct, union, enum), the names declared in each and the parameter
// list of every function declaration; propose() gives the bytes that follow the text so far where they are known.
// The scanner is tolerant, not a compiler, so a proposal is a draft to verify and never text to emit.
//
// What is scanned: all of the text, as C or C++; nothing tells code from prose. Text that is not code must not
// swallow the code after it, so three things end whatever is open (a comment, a literal, a statement, the open
// scopes) and the scan goes on at file level: a line that starts with a code fence (```), a line that starts with
// a markup tag (`<` and a letter or `/`), and boundary(), which the caller reports where a token that is not text
// arrives. Prose between them is read as code and leaves at most stray names, which are only ever offered after a
// `Scope::` that names a known scope.
//
// What is kept: member and namespace-level functions, static members, namespace-level variables, types, aliases,
// enumerators (those of an unscoped enum in the enclosing scope too) and nested scopes. Not kept: non-static data
// members, the contents of function bodies, the parameters of operator functions (their name is `operator`).
// Base classes, using-directives and aliases are not followed, and both branches of a preprocessor conditional are
// read.
//
// Appends cost O(1) per byte with no recursion; memory is bounded by Limits and the constants below, and what is
// past a bound is not indexed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace basalt::spec {

class ScopeDrafter {
public:
    /// The bounds of the index.
    struct Limits {
        uint32_t names = 1u << 16;       ///< names kept, over all scopes
        uint32_t functions = 1u << 16;   ///< function declarations kept
        uint32_t text = 4u << 20;        ///< bytes of names, parameter lists and qualifiers kept
    };
    static constexpr int kMaxDepth = 64;      ///< scopes open at once; a brace past it is only counted
    static constexpr int kMaxName = 96;       ///< bytes of a name; a longer word is no name
    static constexpr int kMaxParams = 1024;   ///< bytes of a parameter list; a longer one is not kept
    static constexpr int kMaxOverloads = 16;  ///< declarations kept per function name
    static constexpr int kMaxQualifiers = 6;  ///< `A::B::` scopes read ahead of a name
    static constexpr int kTail = 512;         ///< bytes of the text's end that propose() looks back over

    /// What propose() wrote: `name` bytes that finish a name of the scope ahead of the cursor (with the `(` of a
    /// function and the `::` of a namespace), then `signature` bytes of a parameter list and what follows it.
    struct Proposal {
        uint32_t name = 0;
        uint32_t signature = 0;
        size_t size() const { return (size_t) name + signature; }
    };

    ScopeDrafter();
    explicit ScopeDrafter(const Limits& limits);

    /// Forgets the text and the index.
    void reset();
    /// Reads more of the text: the prompt, then every accepted token's bytes. Any split of the same bytes gives the
    /// same index.
    void append(std::string_view text);
    /// The text before and the text after are unrelated (a token that is not text came between): whatever was open
    /// ends, the scan goes on at file level, the index stays.
    void boundary();

    /// The bytes that come after the text so far, at most `max_bytes`, in `out` (replaced; empty when none are known):
    ///   - after `Scope::` and the start of a name, what the names of that scope starting so have in common; one
    ///     name left is finished, a function's with `(` and a namespace's with `::`. `Scope` is looked up from the
    ///     scope the text is in outwards, then as the one scope of that name anywhere; two of them give nothing.
    ///     A class or an enum lists all its names, so they are offered right after its `::`; a namespace is open
    ///     to other files, so its names are offered only once one is started.
    ///   - inside the parameter list of a function being declared again (`Scope::name(` at declaration level), the
    ///     rest of the declared list without default arguments, `)` and the qualifiers a definition repeats (const,
    ///     volatile, a reference qualifier, noexcept, a trailing return type), with continuation lines moved to
    ///     the new column. Overloads give what they have in common.
    ///   - at declaration level the two join: the rest of the name, `(`, the list.
    /// Nothing inside a comment, a literal or a preprocessor line. The index is not changed.
    Proposal propose(size_t max_bytes, std::string& out) const;

    size_t names() const { return entries_.size(); }
    size_t scopes() const { return scopes_.size(); }
    size_t functions() const { return fns_.size(); }
    /// The bytes read since the last reset.
    uint64_t size() const { return total_; }

    /// The index as text (a test view): each scope's qualified name and kind, then its names in the order declared,
    /// a function once per declaration as `name(parameters)qualifiers`.
    std::string dump() const;

private:
    enum class Kind : uint8_t { ns, cls, enm, block };
    enum class Lex : uint8_t {
        code, slash, line_comment, block_comment, block_star, string, string_esc, chr, chr_esc,
        raw_delim, raw_body, raw_close, preproc, skip_line
    };
    enum class Line : uint8_t { blank, tick1, tick2, lt, text };
    enum class Head : uint8_t { none, ns, cls, enm, alias, skip };
    enum class Cap : uint8_t { off, params, quals };

    /// A name declared in a scope.
    struct Entry {
        uint32_t text = 0;      ///< where its bytes start in text_
        uint32_t scope = 0;     ///< the scope it is declared in
        uint32_t next = 0;      ///< the name declared before it in the same scope
        uint32_t child = 0;     ///< the scope it names
        uint32_t fn = 0;        ///< its first function declaration
        uint16_t len = 0;
    };
    struct Scope {
        uint32_t parent = 0;
        uint32_t entry = 0;     ///< the name it has in its parent
        uint32_t first = 0;     ///< the name declared last in it
        Kind kind = Kind::ns;
    };
    /// One declaration of a function: its parameter list without default arguments, then its qualifiers, in text_.
    struct Fn {
        uint32_t text = 0;
        uint32_t next = 0;      ///< the next declaration of the same name
        uint16_t params_len = 0, quals_len = 0;
        uint16_t indent = 0;    ///< the indent of the line its `(` is on
        uint16_t col = 0;       ///< the column after its `(`
    };
    /// An open brace the scan is inside: a scope, or a body whose contents declare nothing.
    struct Frame {
        uint32_t scope = 0;     ///< where names are looked up from here (and declared, unless a body)
        uint32_t also = 0;      ///< an unscoped enum: the enclosing scope its enumerators are declared in too
        uint32_t nested = 0;    ///< braces open inside it that are not frames
        Kind kind = Kind::ns;
    };
    /// The declaration being read at declaration level, from the last `;`, `{` or `}`.
    struct Stmt {
        Head head = Head::none;
        uint8_t prev = 0, prev2 = 0;   ///< the last two tokens
        bool is_static = false, is_inline = false, is_extern = false;
        bool literal = false;          ///< a string literal came first (`extern "C"`)
        bool tmpl = false;             ///< `template`: the next `<` opens its parameters
        bool init = false;             ///< after `=`: an initializer, words in it are no names
        bool base = false;             ///< after the `:` of a class head or a constructor: the same
        bool ignore = false;           ///< nothing more of this statement is read
        bool access = false;           ///< an access specifier was read
        bool op = false, op_first = false;   ///< after `operator`, up to its parameter list
        bool has_fn = false;           ///< a parameter list was opened: fn_name_ is the function
        bool in_params = false, after_params = false;
        bool fn_qualified = false;     ///< the function's name was `Scope::name`
        bool noexc = false;            ///< the last qualifier was noexcept: a `(` after it belongs to it
        bool ret = false;              ///< a trailing return type is being captured
        bool qualified = false;        ///< the name follows `::`
        bool global = false;           ///< the qualified name starts with `::`
        bool chain_over = false;       ///< more qualifiers than are kept
        bool over = false;             ///< the last name was longer than is kept
        bool enum_scoped = false;
        bool expect = false;           ///< an enum body: the next word is an enumerator
        uint8_t name_len = 0;
        uint8_t chain_n = 0;
        uint16_t chain_len = 0;
        uint16_t paren = 0;            ///< open ( and [ that are no parameter list
        uint16_t angle = 0;            ///< open template brackets
        uint16_t group = 0;            ///< open parentheses of the list being captured
        uint16_t gbrace = 0;           ///< open braces inside that list
        uint32_t fn_entry = UINT32_MAX;   ///< the declared name the open parameter list repeats, if any
    };

    // the lexer
    bool track_line(unsigned char c);
    void lex(unsigned char c);
    void flush_word();
    void capture(unsigned char c);
    void cut();
    void keep_tail(const char* p, size_t n);

    // the declarations
    void token(uint8_t t);
    void open_brace();
    void close_brace();
    void statement_word(size_t len);
    bool trailing_word(std::string_view w, int kw);
    void statement_token(uint8_t t);
    void enum_word(size_t len);
    void enum_token(uint8_t t);
    void set_name(std::string_view w);
    void start_params();
    void end_group();
    void end_return();
    void declarator();
    void reset_statement();
    Frame& top() { return frames_[depth_ - 1]; }
    const Frame& top() const { return frames_[depth_ - 1]; }

    // the index
    uint32_t find(uint32_t scope, std::string_view name) const;
    uint32_t add_name(uint32_t scope, std::string_view name);
    uint32_t child_scope(uint32_t parent, std::string_view name, Kind kind);
    void add_fn(uint32_t scope, std::string_view name);
    void grow_slots();
    uint32_t lookup_scope(uint32_t from, std::string_view name) const;
    uint32_t resolve(const std::string_view* quals, int n, bool global, uint32_t from) const;
    uint32_t resolve_chain(uint32_t from) const;
    std::string_view name() const { return std::string_view(name_, st_.name_len); }
    std::string_view fn_name() const { return std::string_view(fn_name_, fn_name_len_); }

    // the proposals
    bool tail_name(std::string_view& typed, std::string_view* quals, int& n, bool& global) const;
    void signature(const Fn& f, uint32_t col, uint32_t indent, size_t limit, std::string& out) const;
    void common_signature(uint32_t fn, uint32_t col, uint32_t indent, std::string_view typed, size_t limit,
                          std::string& out) const;

    Limits limits_;

    // the lexer's state: read on every byte
    Lex lex_ = Lex::code;
    Line line_ = Line::blank;
    Cap cap_ = Cap::off;
    bool first_ = false;        ///< this byte is the first of its line that is not blank
    bool colon_ = false;        ///< a `:` waits for the byte that tells `::` from `:`
    bool minus_ = false;        ///< the byte before was `-`
    bool pp_cont_ = false;      ///< a preprocessor line: the last byte was `\`
    bool cap_over_ = false;     ///< the captured list outgrew kMaxParams
    uint8_t raw_n_ = 0, raw_at_ = 0;
    uint32_t id_len_ = 0;       ///< bytes of the word being read (counted past kMaxName)
    uint32_t col_ = 0;          ///< bytes of this line so far
    uint32_t indent_ = 0;       ///< blanks this line starts with
    uint64_t total_ = 0;
    char id_[kMaxName] = {};
    char raw_[16] = {};         ///< a raw string's delimiter

    // the declaration being read
    Stmt st_;
    int depth_ = 1;
    uint32_t fn_scope_ = 0;     ///< the scope a qualified function name resolved to
    uint32_t fn_col_ = 0, fn_indent_ = 0;
    uint8_t fn_name_len_ = 0;
    char name_[kMaxName + 1] = {};      ///< the declared name so far (a destructor's with its `~`)
    char fn_name_[kMaxName + 1] = {};
    char chain_[kMaxQualifiers * kMaxName] = {};   ///< the qualifiers ahead of name_, end to end
    uint8_t chain_seg_[kMaxQualifiers] = {};       ///< their lengths
    Frame frames_[kMaxDepth];
    std::string params_, quals_;        ///< the open or last parameter list; the qualifiers after it

    // the end of the text
    uint32_t tail_n_ = 0;
    char tail_[2 * kTail] = {};

    // the index
    std::vector<Entry> entries_;
    std::vector<Scope> scopes_;
    std::vector<Fn> fns_;
    std::vector<uint32_t> slots_;       ///< open addressing over (scope, name): an entry's index + 1, 0 empty
    std::string text_;
    std::string scratch_;               ///< a parameter list on its way into text_
};

}  // namespace basalt::spec
