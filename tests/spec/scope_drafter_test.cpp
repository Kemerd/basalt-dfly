// tests/spec/scope_drafter_test.cpp - the C and C++ scope index: what the scanner keeps of declarations, what it
// proposes after `Scope::` and inside a parameter list declared before, the proposals as draft tokens, input that is
// not code or not sane, and that the index does not depend on how the text was split. Ends with a timing line.
#include "basalt/spec/scope_drafter.hpp"

#include "basalt/program/grammar/grammar.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using basalt::program::grammar::Vocab;
using basalt::spec::ScopeDrafter;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// The index of `text`, read in one piece.
static std::string index_of(std::string_view text) {
    ScopeDrafter d;
    d.append(text);
    return d.dump();
}

// `got` against `want`, with both printed when they differ.
static bool same(const std::string& got, const std::string& want) {
    if (got == want) return true;
    std::printf("--- got\n%s--- want\n%s---\n", got.c_str(), want.c_str());
    return false;
}

// What is proposed after `text`.
static std::string next(std::string_view text, size_t max_bytes = 256) {
    ScopeDrafter d;
    d.append(text);
    std::string out = "replaced";
    d.propose(max_bytes, out);
    return out;
}

static const char* kHeader = R"cpp(// a header
#pragma once
#include <string>

namespace app::net {

enum class Mode : uint8_t { idle, reading = 2, writing };
enum Flags { kNone = 0, kFast = 1 << 0, kSafe };

struct Options;   // defined elsewhere

/// A connection.
class Connection final : public Base<Connection> {
public:
    enum class State { closed, open, failed };
    struct Stats { int sent = 0; int received = 0; };
    using Callback = std::function<void(int)>;

    explicit Connection(const Options& options, int retries = 3);
    ~Connection();
    static std::shared_ptr<Connection> connect(std::string_view host, uint16_t port = 80,
                                               Mode mode = Mode::idle);
    virtual bool send(const char* data, size_t size) noexcept = 0;
    bool send(std::string_view text) noexcept;
    [[nodiscard]] size_t pending() const { return queue_.size(); }
    State state() const noexcept override;
    template <class T> void set(const std::map<std::string, T>& values, T fallback = T());
    auto stats() const -> Stats;
    bool operator==(const Connection& other) const;
    static const int kMaxRetries = 5;
    static constexpr char kName[] = "conn";

private:
    void reset_();
    int fd_ = -1;
    std::vector<char> queue_;
};

int checksum(const uint8_t* data, size_t size);
extern int g_verbose;

}  // namespace app::net
)cpp";

static const char* kHeaderIndex =
    ":: namespace\n"
    "  app\n"
    "app namespace\n"
    "  net\n"
    "app::net namespace\n"
    "  Mode\n"
    "  Flags\n"
    "  kNone\n"
    "  kFast\n"
    "  kSafe\n"
    "  Options\n"
    "  Connection\n"
    "  checksum(const uint8_t* data, size_t size)\n"
    "  g_verbose\n"
    "app::net::Mode enum\n"
    "  idle\n"
    "  reading\n"
    "  writing\n"
    "app::net::Flags enum\n"
    "  kNone\n"
    "  kFast\n"
    "  kSafe\n"
    "app::net::Connection class\n"
    "  State\n"
    "  Stats\n"
    "  Callback\n"
    "  Connection(const Options& options, int retries)\n"
    "  ~Connection()\n"
    "  connect(std::string_view host, uint16_t port,\\n                                               Mode mode)\n"
    "  send(const char* data, size_t size) noexcept\n"
    "  send(std::string_view text) noexcept\n"
    "  pending() const\n"
    "  state() const noexcept\n"
    "  set(const std::map<std::string, T>& values, T fallback)\n"
    "  stats() const -> Stats\n"
    "  operator\n"
    "  kMaxRetries\n"
    "  kName\n"
    "  reset_()\n"
    "app::net::Connection::State enum\n"
    "  closed\n"
    "  open\n"
    "  failed\n"
    "app::net::Connection::Stats class\n";

// What the scanner keeps of declarations.
static void test_scan() {
    CHECK(same(index_of(kHeader), kHeaderIndex));
    CHECK(same(index_of(""), ":: namespace\n"));
    // namespaces: nested, reopened, unnamed and inline ones (which declare into the one around them), an alias
    CHECK(same(index_of("namespace a { namespace b { int f(); } }\n"
                        "namespace a::b { int g(int x); }\n"
                        "namespace { int hidden(); }\n"
                        "namespace a { inline namespace v1 { int h(); } }\n"
                        "namespace short_name = a::b;\n"),
               ":: namespace\n  a\n  hidden()\n  short_name\n"
               "a namespace\n  b\n  h()\n"
               "a::b namespace\n  f()\n  g(int x)\n"));
    // classes, structs and unions; forward declarations; nested types; what a class keeps and what it does not
    CHECK(same(index_of("class A;\nstruct B { union U { int i; float f; }; class Inner; static int count; int size; };\n"
                        "class A { public: int get() const; protected: static void make(); private: int x_; struct Late; };\n"
                        "struct A::Late { static int late(); };\nstruct Missing::Lost { static int lost(); };\n"),
               ":: namespace\n  A\n  B\n"
               "B class\n  U\n  Inner\n  count\n"
               "B::U class\n"
               "A class\n  get() const\n  make()\n  Late\n"
               "A::Late class\n  late()\n"));
    // enums: scoped, unscoped (declared around them too), unnamed, with values, attributes and a trailing comma
    CHECK(same(index_of("enum class E : int { a = 1, b = (2, 3), c [[deprecated]] = f(4, 5), };\n"
                        "enum { loose, ends = sizeof(int[2]) };\nenum Plain : char;\n"),
               ":: namespace\n  E\n  loose\n  ends\n  Plain\n"
               "E enum\n  a\n  b\n  c\n"));
    // templates: parameter lists are stepped over, arguments in declarators too
    CHECK(same(index_of("template <class T, int N = (3 > 2)> struct Box { T get(std::array<T, N>& from) const; };\n"
                        "template <typename K, typename V> std::map<K, std::vector<V>> group(const std::vector<V>& in);\n"
                        "template <> struct Box<int, 0> { void only(); };\n"),
               ":: namespace\n  Box\n  group(const std::vector<V>& in)\n"
               "Box class\n  get(std::array<T, N>& from) const\n  only()\n"));
    // what a definition does not repeat is dropped; what it does is kept in order
    CHECK(same(index_of("struct Q {\n"
                        "  [[nodiscard]] virtual int a(int x = 1, const char* s = \"a,b\", std::pair<int, int> p = {}) const & = 0;\n"
                        "  static constexpr inline int b(char c = ',', int d = g(1, 2)) noexcept(true) { return 0; }\n"
                        "  explicit Q(std::function<void(int, int)> cb = nullptr) noexcept;\n"
                        "  auto c() && -> decltype(x) override final;\n"
                        "  void d() volatile throw();\n"
                        "  friend bool operator<(const Q&, const Q&);\n"
                        "  friend class R;\n"
                        "  operator bool() const;\n"
                        "  void operator()(int x);\n"
                        "};\n"),
               ":: namespace\n  Q\n"
               "Q class\n"
               "  a(int x, const char* s, std::pair<int, int> p) const &\n"
               "  b(char c, int d) noexcept(true)\n"
               "  Q(std::function<void(int, int)> cb) noexcept\n"
               "  c() && -> decltype(x)\n"
               "  d() volatile\n"
               "  operator\n"));
    // a member defined outside its class declares nothing where it is defined; a body declares nothing at all
    CHECK(same(index_of("struct S { void f(int a); };\n"
                        "void S::f(int a) { struct Local { int g(); }; int local = 0; }\n"
                        "static int helper(int v) { return v; }\n"
                        "int table[] = {1, 2, 3}, *cursor = table;\n"
                        "auto lambda = [](int x) { return x; };\n"),
               ":: namespace\n  S\n  helper(int v)\n  table\n  cursor\n  lambda\n"
               "S class\n  f(int a)\n"));
    // linkage blocks, aliases, macros ahead of and behind a declaration, and the same declaration read twice
    CHECK(same(index_of("extern \"C\" {\nint c_open(const char* path);\n}\nextern \"C\" int c_close(int fd);\n"
                        "typedef unsigned long word;\nusing text = std::string;\nusing std::swap;\nusing namespace std;\n"
                        "EXPORT(\"x\") void exported(int a);\n__declspec(dllexport) int declared(void);\n"
                        "void trailing(int a) ATTRIBUTE;\n"
                        "int c_open(const char* path);\nint c_open(const char* path, int flags);\n"),
               ":: namespace\n  c_open(const char* path)\n  c_open(const char* path, int flags)\n  c_close(int fd)\n"
               "  word\n  text\n  swap\n  exported(int a)\n  declared(void)\n  trailing(int a)\n"));
}

// Comments, literals and preprocessor lines hide what is in them.
static void test_lex() {
    CHECK(same(index_of("// class Hidden { void f(); };\n"
                        "/* struct Also { void g(); }; */ /** and { this */\n"
                        "#define OPEN namespace m { \\\n   class Macro {\n"
                        "#if 0\n"
                        "const char* a = \"class S { void h(); }; \\\" } {\";\n"
                        "const char b = '{', c = '\\'', d = '}';\n"
                        "const char* e = R\"x(class R { void i(); }; )\" still )x\";\n"
                        "const wchar_t* w = L\"{\"; auto u = u8R\"(})\";\n"
                        "int big = 1'000'000; int div = 6 / 2; /* / */ int shown(int x /* count */, int y); // tail\n"),
               ":: namespace\n  a\n  b\n  c\n  d\n  e\n  w\n  u\n  big\n  div\n  shown(int x , int y)\n"));
    // a literal that lost its quote ends with its line; a comment that never ends hides the rest
    // (the statement the literal was in runs to the next `;`)
    CHECK(same(index_of("const char* s = \"open;\nint eaten(int a);\nint after(int a);\n/* never closed\nint lost(int b);\n"),
               ":: namespace\n  s\n  after(int a)\n"));
    CHECK(same(index_of("int r = 1; auto q = R\"tag(never closed\nint lost(int b);\n"), ":: namespace\n  r\n"));
}

// Text that is not code: prose is read as code but cannot swallow what follows a fence, a tag line or a boundary.
static void test_regions() {
    const char* chat = "Here's the glob src/*.cpp and it's \"unterminated { prose\n"
                       "```cpp\nnamespace ui {\nclass Button {\npublic:\n    void click(int times);\n};\n}\n```\n"
                       "Now don't forget /* this\n"
                       "<tool_response>\nstruct Raw { static int count(const char* s); };\n</tool_response>\n"
                       "trailing words = unfinished {\n"
                       "<parameter=content>\nnamespace late { int last(); }\n";
    // (`words` is a stray name: the prose around it reads as a variable with a braced initializer)
    CHECK(same(index_of(chat), ":: namespace\n  ui\n  Raw\n  words\n  late\n"
                               "ui namespace\n  Button\n"
                               "ui::Button class\n  click(int times)\n"
                               "Raw class\n  count(const char* s)\n"
                               "late namespace\n  last()\n"));
    // a fence may be indented; the scopes open at a cut are closed by it
    CHECK(same(index_of("namespace open {\n   ```\nint top();\n"), ":: namespace\n  open\n  top()\nopen namespace\n"));
    // not tags: a shift at the start of a line, a comparison
    CHECK(same(index_of("namespace n {\nint a = 1\n<< 2;\nint b = a\n< 3;\nint c();\n}\n"),
               ":: namespace\n  n\nn namespace\n  a\n  b\n  c()\n"));
    // the caller's boundary does the same and keeps the index
    ScopeDrafter d;
    d.append("namespace n { int f();\n/* open");
    d.boundary();
    d.append("int g();");
    CHECK(same(d.dump(), ":: namespace\n  n\n  g()\nn namespace\n  f()\n"));
    std::string out;
    d.append(" int x = n::f");
    CHECK(d.propose(64, out).name == 1 && out == "(");
    d.boundary();
    CHECK(d.propose(64, out).size() == 0 && out.empty());   // the text before a boundary is not looked back over
    d.reset();
    CHECK(same(d.dump(), ":: namespace\n") && d.size() == 0);
}

static const std::string kSource = std::string(kHeader) + "#include \"connection.hpp\"\n\nnamespace app::net {\n\n";

// After `Scope::`: the common continuation of the scope's names.
static void test_names() {
    const std::string src = kSource;
    CHECK(next(src + "Connection::").empty());                        // many names, nothing in common
    CHECK(next(src + "bool Connection::se").empty());                 // send and set part here
    CHECK(next(src + "void Connection::rese") == "t_()");
    CHECK(next(src + "auto x = Connection::k").empty());              // kMaxRetries, kName
    CHECK(next(src + "auto x = Connection::kM") == "axRetries");
    CHECK(next(src + "Connection::Sta") == "t");                      // State, Stats
    CHECK(next(src + "Connection::Stat").empty());
    CHECK(next(src + "Connection::zzz").empty() && next(src + "Connection::sendx").empty());
    CHECK(next(src + "Unknown::a").empty() && next(src + "std::").empty() && next(src + "::app").empty());
    CHECK(next(src + "Connection:").empty() && next(src + "Connection").empty() && next(src + "Connection::1").empty());
    // a namespace is open: nothing until a name is started; its name ends in `::`, but not where it is declared
    CHECK(next("namespace app::net { int f(); }\nint x = ap").empty());
    CHECK(next(src + "} int x = app::").empty() && next(src + "} int x = app::net::").empty());
    CHECK(next(src + "} int x = app::n") == "et::" && next(src + "} int x = ::app::n") == "et::");
    CHECK(next(src + "} namespace app::n") == "et" && next(src + "} using namespace app::n") == "et");
    // qualifiers are followed inwards
    CHECK(next(src + "} int x = app::net::Mode::r") == "eading" && next(src + "} int x = app::net::ch") == "ecksum(");
    CHECK(next(src + "} int x = app::Mode::r").empty());              // no such scope in app
    // enumerators, of a scoped enum and of an unscoped one in both places
    CHECK(next(src + "Mode m = Mode::").empty() && next(src + "Mode m = Mode::w") == "riting");
    CHECK(next(src + "int f = Flags::kS") == "afe" && next(src + "} int f = app::net::kF") == "ast");
    // inside a body: a function's name ends at its parenthesis, and nothing is known of a call's arguments
    const std::string body = src + "bool Connection::send(std::string_view text) noexcept {\n    ";
    CHECK(next(body + "auto c = Connection::conn") == "ect(" && next(body + "auto c = Connection::connect(").empty());
    CHECK(next(body + "if (state() == State::f") == "ailed");         // looked up from the member's class
    CHECK(next(body + "Stats s; s.sent = Connection::kN") == "ame");
    CHECK(next(body + "delete this; Connection::~") == "Connection(");
    // a class template's arguments are stepped over
    CHECK(next("template <class T> struct Box { static int make(T v); };\nint x = Box<std::pair<int, int>>::ma") == "ke(");
    // not inside a comment, a literal or a preprocessor line
    CHECK(next(src + "// see Connection::conn").empty() && next(src + "/* Connection::conn").empty());
    CHECK(next(src + "const char* s = \"Connection::conn").empty() && next(src + "#define X Connection::conn").empty());
    CHECK(next(src + "int x = 6 /").empty());
    // a scope name that two scopes have is looked up from where the text is, and is nothing from elsewhere
    const std::string two = "struct A { enum class Kind { left, right }; void f(); };\n"
                            "struct B { enum class Kind { up, down }; void g(); };\n";
    CHECK(next(two + "int x = (int) Kind::l").empty());
    CHECK(next(two + "void A::f() { auto k = Kind::l") == "eft" && next(two + "void B::g() { auto k = Kind::d") == "own");
    CHECK(next(two + "int x = (int) A::Kind::r") == "ight");
    CHECK(next(two + "struct A { void h() { auto k = Kind::r") == "ight");
    // the bound, and the two parts of a proposal
    CHECK(next(src + "auto x = Connection::kM", 4) == "axRe" && next(src + "auto x = Connection::kM", 0).empty());
    ScopeDrafter d;
    d.append(src + "int x = Connection::pend");
    std::string out;
    const ScopeDrafter::Proposal p = d.propose(64, out);
    CHECK(out == "ing(" && p.name == 4 && p.signature == 0 && p.size() == 4);
    CHECK(d.propose(64, out).size() == 4 && out == "ing(");           // asking changes nothing
}

// Inside the parameter list of a function declared before, and from its name at declaration level.
static void test_signatures() {
    const std::string src = kSource;
    CHECK(next(src + "Connection::Connection(") == "const Options& options, int retries)");
    CHECK(next(src + "Connection::Conn") == "ection(const Options& options, int retries)");
    CHECK(next(src + "Connection::~Connection(") == ")" && next(src + "Connection::~") == "Connection()");
    CHECK(next(src + "Connection::State Connection::state") == "() const noexcept");
    CHECK(next(src + "auto Connection::stats(") == ") const -> Stats");
    CHECK(next(src + "template <class T>\nvoid Connection::set(") == "const std::map<std::string, T>& values, T fallback)");
    CHECK(next(src + "void Connection::reset_(") == ")" && next(src + "void Connection::reset_()").empty());
    // overloads: what they have in common, then the one the text has chosen
    CHECK(next(src + "bool Connection::sen") == "d(" && next(src + "bool Connection::send(").empty());
    CHECK(next(src + "bool Connection::send(c") == "onst char* data, size_t size) noexcept");
    CHECK(next(src + "bool Connection::send(std::string_view t") == "ext) noexcept");
    CHECK(next(src + "bool Connection::send(int").empty());           // not a declared list
    CHECK(next(src + "bool Connection::send(const char* data, size_t size") == ") noexcept");
    // a continuation line follows the parenthesis to its new column
    CHECK(next(src + "std::shared_ptr<Connection> Connection::con") ==
          "nect(std::string_view host, uint16_t port,\n" + std::string(48, ' ') + "Mode mode)");
    CHECK(next(src + "std::shared_ptr<Connection>\nConnection::connect(std::string_view host, uint16_t port,") ==
          "\n" + std::string(20, ' ') + "Mode mode)");
    // one indented from its line follows the line
    const std::string wrapped = "struct W {\n    void long_name(\n        int first,\n        int second = 2) const;\n};\n";
    CHECK(next(wrapped + "void W::long_name(") == "\n    int first,\n    int second) const");
    CHECK(next(wrapped + "  void W::long_n") == "ame(\n      int first,\n      int second) const");
    // an unqualified name is not taken for a declaration made again: it may as well be a new overload
    CHECK(next(src + "int checksum(").empty() && next(src + "int checksum(const uint8_t* data").empty());
    CHECK(next(src + "int app::net::checksum(const uint8_t* data") == ", size_t size)");
    CHECK(next("struct T { void f(int a); void f(").empty());
    // only at declaration level: not in an initializer, a body or a call's arguments
    CHECK(next(src + "auto p = Connection::conn") == "ect(" && next(src + "auto p = Connection::connect(").empty());
    CHECK(next(src + "void g() { checksum(").empty() && next(src + "void g(int a = checksum(").empty());
    // a member of a class template, and a qualified name down from a namespace
    CHECK(next("template <class T> struct Box { T get(int at) const; };\ntemplate <class T> T Box<T>::get(") == "int at) const");
    CHECK(next(std::string(kHeader) + "bool app::net::Connection::send(c") == "onst char* data, size_t size) noexcept");
    CHECK(next(std::string(kHeader) + "int app::net::chec") == "ksum(const uint8_t* data, size_t size)");
    // the bound cuts the list, and the parts are counted apart
    ScopeDrafter d;
    d.append(src + "Connection::Conn");
    std::string out;
    ScopeDrafter::Proposal p = d.propose(64, out);
    CHECK(p.name == 7 && p.signature == 36 && out.size() == 43);
    p = d.propose(10, out);
    CHECK(out == "ection(con" && p.name == 7 && p.signature == 3);
    p = d.propose(3, out);
    CHECK(out == "ect" && p.name == 3 && p.signature == 0);
}

static Vocab draft_vocab() {
    return Vocab({"ection", "(", "const", " Options", "&", " options", ",", " int", " ret", "ries", ")", "", "ect", " O",
                  "ection("},
                 {11});
}

// The proposal's bytes as draft tokens, through the vocabulary trie.
static void test_tokens() {
    const Vocab voc = draft_vocab();
    ScopeDrafter d;
    d.append(kSource + "Connection::Conn");
    std::string bytes;
    d.propose(64, bytes);
    std::vector<int32_t> ids = {-7};
    voc.spell(bytes, 16, ids);
    // the longest entry at each step: `ection(` over `ection`, ` Options` over ` O`
    CHECK(ids == (std::vector<int32_t>{14, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    std::string spelled;
    for (int32_t id : ids) spelled += voc.bytes(id);
    CHECK(spelled == bytes);
    voc.spell(bytes, 3, ids);
    CHECK(ids == (std::vector<int32_t>{14, 2, 3}));
    // a proposal the vocabulary cannot spell from its first byte gives no draft
    d.reset();
    d.append(kSource + "auto x = Connection::kM");
    d.propose(64, bytes);
    voc.spell(bytes, 16, ids);
    CHECK(bytes == "axRetries" && ids.empty());
}

static uint32_t next_random(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
}

// Input that is not sane leaves the index within its bounds and the calls returning.
static void test_robust() {
    std::string out;
    // random bytes, then bytes drawn from what C++ is made of, a proposal asked at every step
    for (uint32_t seed = 1; seed <= 6; ++seed) {
        ScopeDrafter d;
        uint32_t state = seed;
        const std::string_view pieces[] = {"{", "}", "(", ")", "<", ">", "::", ":", ";", ",", "=", "\"", "'", "/*", "*/", "//",
                                           "\n", " ", "#", "\\", "R\"x(", ")x\"", "```", "class ", "struct ", "enum ",
                                           "namespace ", "template ", "operator", "static ", "using ", "A", "b", "~", "&", "->"};
        for (int i = 0; i < 20000; ++i) {
            if (seed <= 2) {
                const char c = (char) next_random(state);
                d.append(std::string_view(&c, 1));
            } else {
                d.append(pieces[next_random(state) % (sizeof pieces / sizeof pieces[0])]);
            }
            const ScopeDrafter::Proposal p = d.propose(64, out);
            CHECK(p.size() == out.size() && out.size() <= 64);
        }
        CHECK(!d.dump().empty());
    }
    // nesting far past what is kept: counted, and unwound to the same place
    {
        ScopeDrafter d;
        d.append("namespace deep {\n");
        for (int kind = 0; kind < 3; ++kind) {
            const char* open = kind == 0 ? "{" : kind == 1 ? "namespace n {" : "struct S {";
            for (int i = 0; i < 100000; ++i) d.append(open);
            for (int i = 0; i < 100000; ++i) d.append("}");
        }
        d.append(std::string(100000, '(') + std::string(100000, ')') + ";");
        d.append("template " + std::string(100000, '<') + std::string(100000, '>') + ";");
        d.append("int after(int a);\n}\n}}}}\nint top();\n");
        const std::string index = d.dump();
        CHECK(index.find("deep namespace\n  n\n  S\n  after(int a)\n") != std::string::npos);
        CHECK(index.find(":: namespace\n  deep\n  top()\n") == 0);
        CHECK(d.scopes() <= 3 + 2 * (size_t) ScopeDrafter::kMaxDepth);
    }
    // words and lists longer than what is kept are not indexed and break nothing after them
    {
        const std::string long_name(5000, 'x'), long_list(5000, 'y');
        ScopeDrafter d;
        d.append("struct " + long_name + " { int a(); };\nint " + long_name + "(int b);\nint f(int " + long_list + ");\n"
                 "int g(int c) noexcept(" + long_list + ");\nint ok(int d);\nint x = " + long_name + "::");
        CHECK(same(d.dump(), ":: namespace\n  ok(int d)\n"));
        CHECK(d.propose(64, out).size() == 0);
        d.append(std::string(600, 'a') + "::" + std::string(600, 'b') + "::");
        CHECK(d.propose(64, out).size() == 0);
    }
    // the bounds of the index: full, it stays as it is
    {
        ScopeDrafter::Limits tight;
        tight.names = 8;
        tight.functions = 3;
        ScopeDrafter d(tight);
        for (int i = 0; i < 100; ++i) d.append("int f" + std::to_string(i) + "(int a); struct s" + std::to_string(i) + " {};\n");
        CHECK(d.names() == 8 && d.functions() == 3 && d.scopes() <= 9);
        tight.names = 1000;
        tight.functions = 1000;
        tight.text = 64;
        ScopeDrafter e(tight);
        for (int i = 0; i < 100; ++i) e.append("int function_number_" + std::to_string(i) + "(int a);\n");
        CHECK(e.names() < 100 && e.dump().size() < 400);
        // more overloads than are kept per name
        ScopeDrafter o;
        for (int i = 0; i < 100; ++i) o.append("void many(int a" + std::to_string(i) + ");\n");
        CHECK(o.functions() == (size_t) ScopeDrafter::kMaxOverloads && o.names() == 1);
    }
    // a megabyte of declarations in a bounded time
    {
        std::string big;
        for (int i = 0; big.size() < (1u << 20); ++i) {
            big += "namespace part" + std::to_string(i) + " {\n";
            big += kHeader;
            big += "}\n";
        }
        ScopeDrafter d;
        const auto t0 = std::chrono::steady_clock::now();
        d.append(big);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ms < 2000.0);
        CHECK(d.names() > 10000 && d.functions() > 5000);
        std::printf("scope_drafter_test: %.2f MB in %.1f ms (%.0f MB/s), %zu names, %zu scopes, %zu functions\n",
                    (double) big.size() / (1 << 20), ms, (double) big.size() / (1 << 20) / (ms / 1000.0), d.names(), d.scopes(),
                    d.functions());
        // a proposal where there is none to make, the common case
        d.append("int x = y + ");
        const auto p0 = std::chrono::steady_clock::now();
        size_t found = 0;
        for (int i = 0; i < 1000000; ++i) found += d.propose(64, out).size();
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - p0).count() / 1e6;
        CHECK(found == 0);
        std::printf("scope_drafter_test: %.1f ns per proposal with nothing to propose\n", ns);
    }
}

// The index and the proposals do not depend on how the text was split.
static void test_chunks() {
    const std::string text = std::string(kHeader) +
                             "Prose, it's here: src/*.h\n```cpp\nnamespace ui { class Button { void click(int n); }; }\n```\n" +
                             kSource +
                             "Connection::Connection(const Options& options, int retries) : fd_(-1) {}\n"
                             "std::shared_ptr<Connection> Connection::connect(std::string_view host, uint16_t port,\n"
                             "                                                Mode mode) {\n"
                             "    return mode == Mode::idle ? nullptr : std::make_shared<Connection>(Options{}, 1);\n}\n"
                             "const char* raw = R\"json({\"a\": \"}\"})json\"; /* done */ bool Connection::send(std::string_view text";
    // byte by byte, with the proposal after every byte
    ScopeDrafter ref;
    std::vector<std::string> proposals(text.size() + 1);
    size_t proposed = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        ref.append(std::string_view(&text[i], 1));
        ref.propose(64, proposals[i + 1]);
        proposed += !proposals[i + 1].empty();
    }
    CHECK(proposed > 100);
    CHECK(same(ref.dump(), index_of(text)));
    CHECK(proposals[text.size()] == ") noexcept");
    uint32_t state = 7;
    for (int round = 0; round < 12; ++round) {
        const size_t fixed[] = {2, 3, 5, 7, 16, 61, 256, 1000};
        ScopeDrafter d;
        std::string out;
        size_t at = 0;
        while (at < text.size()) {
            const size_t want = round < 8 ? fixed[round] : 1 + next_random(state) % (round < 10 ? 9 : 700);
            const size_t n = std::min(want, text.size() - at);
            d.append(std::string_view(text.data() + at, n));
            at += n;
            d.propose(64, out);
            CHECK(out == proposals[at]);
        }
        CHECK(d.dump() == ref.dump() && d.size() == text.size());
    }
}

int main() {
    test_scan();
    test_lex();
    test_regions();
    test_names();
    test_signatures();
    test_tokens();
    test_robust();
    test_chunks();
    if (g_fail) { std::printf("scope_drafter_test: %d failures\n", g_fail); return 1; }
    std::printf("scope_drafter_test: ok\n");
    return 0;
}
