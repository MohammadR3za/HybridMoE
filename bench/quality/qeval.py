# Head-to-head quality check for a llama-server: 10 numeric reasoning questions (answers computed here), 8 coding tasks
# (the model's code runs against hidden tests inside bwrap: read-only filesystem, no network, time and memory limits)
# and 3 instruction-following checks. The model thinks (its default). One sample per task per run; results go to a
# jsonl file, and tasks already in it are skipped, so an interrupted run resumes.
#
# usage: qeval.py <port> <results.jsonl> <sample index> [max_tokens]   (against a running llama-server; needs bwrap, and
# `python3 qselfcheck.py` first shows the tests accept right code and reject wrong code)
import itertools, json, math, os, random, re, subprocess, sys, tempfile, time, urllib.request
from fractions import Fraction

WORK = os.environ.get("QEVAL_WORK")  # where the model's code is written for the tests; default: the system temp dir

# ---------------------------------------------------------------- numeric questions
def knights():
    sols = []
    for A, B, C, D in itertools.product([True, False], repeat=4):  # True = knight, who tells the truth
        said = [([B, C, D].count(True) == 1), (A == D), (not B), (C or not A)]
        if all(kind == truth for kind, truth in zip([A, B, C, D], said)):
            sols.append((A, B, C, D))
    assert len(sols) == 1, sols
    return sum(sols[0])

def derangements(n):
    return sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n)))

def coin_ways(total, coins):
    ways = [1] + [0] * total
    for c in coins:
        for v in range(c, total + 1):
            ways[v] += ways[v - c]
    return ways[total]

TAIL = "\n\nThink it through carefully. End your reply with a final line of the form `ANSWER: <integer>`."
NUMERIC = [
    ("n_div", "How many positive integers less than 1000 are divisible by 7 but not divisible by 11?" + TAIL,
     sum(1 for n in range(1, 1000) if n % 7 == 0 and n % 11 != 0)),
    ("n_mod", "What is the remainder when 7^2026 is divided by 13?" + TAIL, pow(7, 2026, 13)),
    ("n_coins", "In how many different ways can you make exactly 100 cents using any number of pennies (1 cent), nickels (5), dimes (10) "
                "and quarters (25)? The order of the coins does not matter." + TAIL, coin_ways(100, [1, 5, 10, 25])),
    ("n_zeros", "How many trailing zeros does 125! (125 factorial) have when written in base 10?" + TAIL,
     len(str(math.factorial(125))) - len(str(math.factorial(125)).rstrip("0"))),
    ("n_digits", "What is the sum of the decimal digits of 2^100?" + TAIL, sum(map(int, str(2 ** 100)))),
    ("n_incr", "How many 4-digit positive integers (from 1000 to 9999) have strictly increasing digits from left to right?" + TAIL,
     sum(1 for n in range(1000, 10000) if all(a < b for a, b in zip(str(n), str(n)[1:])))),
    ("n_knights", "Four inhabitants of an island, A, B, C and D, are each either a knight (always tells the truth) or a knave (always lies). "
                  "A says: \"Exactly one of B, C and D is a knight.\" B says: \"A and D are the same type.\" C says: \"B is a knave.\" "
                  "D says: \"C is a knight or A is a knave.\" How many of the four are knights?" + TAIL, knights()),
    ("n_derange", "In how many ways can 8 non-attacking rooks be placed on an 8x8 chessboard so that no rook stands on the main diagonal "
                  "(the squares (1,1), (2,2), ..., (8,8))?" + TAIL, derangements(8)),
    ("n_tank", "A tank is filled by pipe A alone in 6 hours and by pipe B alone in 9 hours. A drain C alone would empty the full tank in 18 hours. "
               "If A, B and C are all open at once, how many minutes does it take to fill the empty tank?" + TAIL,
     int(1 / (Fraction(1, 6) + Fraction(1, 9) - Fraction(1, 18)) * 60)),
    ("n_seven", "How many integers from 1 to 1000 inclusive contain at least one digit 7?" + TAIL, sum(1 for n in range(1, 1001) if "7" in str(n))),
]

def last_int(text):
    m = re.findall(r"ANSWER:\s*[*`]*\s*(-?[\d,]+)", text)
    if not m:
        m = re.findall(r"-?\d[\d,]*", text)
    try:
        return int(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None

# ---------------------------------------------------------------- coding tasks
CODE_TAIL = "\n\nReply with the complete Python 3 code in a single ```python block, using only the standard library."
CODE = [
 ("c_merge", "Write a function `merge_intervals(intervals)` that takes a list of [start, end] pairs of integers (start <= end, in any order, "
  "possibly empty) and returns the merged list sorted by start: intervals that overlap or touch (next start <= current end) are merged." + CODE_TAIL,
 """
from solution import merge_intervals
import random
norm = lambda r: [list(x) for x in r]
def ref(iv):
    out = []
    for s, e in sorted([list(x) for x in iv]):
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out
assert norm(merge_intervals([[1,3],[2,6],[8,10],[15,18]])) == [[1,6],[8,10],[15,18]]
assert norm(merge_intervals([[1,4],[4,5]])) == [[1,5]]
assert norm(merge_intervals([])) == []
assert norm(merge_intervals([[5,7],[1,2]])) == [[1,2],[5,7]]
assert norm(merge_intervals([[1,10],[2,3],[4,5]])) == [[1,10]]
random.seed(1)
for _ in range(300):
    iv = []
    for _ in range(random.randint(0, 8)):
        a = random.randint(0, 20); iv.append([a, a + random.randint(0, 6)])
    assert norm(merge_intervals([list(x) for x in iv])) == ref(iv), iv
print("PASS")
"""),
 ("c_expr", "Write a function `eval_expr(s)` that evaluates an arithmetic expression given as a string and returns a float. It supports non-negative "
  "decimal numbers, the binary operators + - * / with the usual precedence and left associativity, parentheses, unary minus (also repeated, as in `--3`, "
  "and after an operator, as in `2*-3`), and arbitrary whitespace. Dividing by zero raises ZeroDivisionError. Do not use eval, exec or ast." + CODE_TAIL,
 """
from solution import eval_expr
import random
cases = {"2+3*4":14, "(2+3)*4":20, "-(3+4)*2":-14, "2*-3":-6, "10/4":2.5, " 1 + 2 * (3 - 4) / 5 ":0.6, "--3":3, "2*(3+(4-1))":12,
         "8/2/2":2, "2-3-4":-5, "1.5*4":6, "((7))":7, "-2*-3":6, "3 - -2":5, "2 * 3 + 4 * 5":26, "100/10/5*2":4}
for s, v in cases.items():
    assert abs(eval_expr(s) - v) < 1e-9, (s, eval_expr(s), v)
try:
    eval_expr("1/0"); raise SystemExit("no ZeroDivisionError")
except ZeroDivisionError: pass
random.seed(2)
def gen(d=0):
    if d > 3 or random.random() < 0.3: return str(random.randint(0, 9))
    r = random.random()
    if r < 0.2: return "-" + gen(d + 1)
    if r < 0.4: return "(" + gen(d + 1) + ")"
    return gen(d + 1) + random.choice("+-*") + gen(d + 1)
for _ in range(500):
    e = gen()
    assert abs(eval_expr(e) - eval(e)) < 1e-9, e
print("PASS")
"""),
 ("c_osa", "Write a function `osa_distance(a, b)` that returns the optimal string alignment distance between two strings: the minimum number of "
  "single-character insertions, deletions, substitutions and transpositions of two adjacent characters needed to turn a into b, where no substring may be "
  "edited more than once (so osa_distance('ca', 'abc') is 3)." + CODE_TAIL,
 """
from solution import osa_distance
import random
def ref(a, b):
    n, m = len(a), len(b)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2]+1)
    return d[n][m]
for a, b, v in [("ca","abc",3), ("ab","ba",1), ("kitten","sitting",3), ("","abc",3), ("abc","",3), ("a","a",0), ("abcdef","abcfad",3)]:
    assert osa_distance(a, b) == v, (a, b, osa_distance(a, b), v)
random.seed(3)
for _ in range(1500):
    a = "".join(random.choice("abc") for _ in range(random.randint(0, 8)))
    b = "".join(random.choice("abc") for _ in range(random.randint(0, 8)))
    assert osa_distance(a, b) == ref(a, b), (a, b)
print("PASS")
"""),
 ("c_topo", "Write a function `smallest_topo_order(n, edges)` for a directed graph with nodes 0..n-1 and a list of edges (u, v) meaning u must come before v "
  "(duplicate edges are possible). Return the lexicographically smallest topological order as a list, and raise ValueError if the graph has a cycle "
  "(a self-loop is a cycle)." + CODE_TAIL,
 """
from solution import smallest_topo_order
import heapq, random
def ref(n, edges):
    adj = [[] for _ in range(n)]; indeg = [0]*n
    for u, v in edges: adj[u].append(v); indeg[v] += 1
    h = [i for i in range(n) if indeg[i] == 0]; heapq.heapify(h); out = []
    while h:
        u = heapq.heappop(h); out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(h, v)
    return out if len(out) == n else None
assert list(smallest_topo_order(0, [])) == []
assert list(smallest_topo_order(3, [])) == [0, 1, 2]
assert list(smallest_topo_order(4, [(3, 0), (2, 1)])) == [2, 1, 3, 0]
for bad in [(3, [(0,1),(1,2),(2,0)]), (1, [(0,0)]), (2, [(0,1),(1,0)])]:
    try:
        smallest_topo_order(*bad); raise SystemExit("no ValueError")
    except ValueError: pass
random.seed(4)
for _ in range(400):
    n = random.randint(1, 9); perm = list(range(n)); random.shuffle(perm)
    edges = []
    for _ in range(random.randint(0, 14)):
        i, j = sorted(random.sample(range(n), 2)) if n > 1 else (0, 0)
        if i != j: edges.append((perm[i], perm[j]))
    if random.random() < 0.25 and edges:
        u, v = random.choice(edges); edges.append((v, u))
    want = ref(n, edges)
    if want is None:
        try:
            smallest_topo_order(n, edges); raise SystemExit("cycle not detected")
        except ValueError: pass
    else:
        assert list(smallest_topo_order(n, edges)) == want, (n, edges)
print("PASS")
"""),
 ("c_lru", "Write a class `LRUCache` with `__init__(self, capacity)` (capacity >= 1), `get(self, key)` returning the value or -1 if absent, "
  "`put(self, key, value)` which inserts or updates a key (both count as a use) and evicts the least recently used key when the size would exceed the capacity, "
  "and `keys(self)` returning the list of keys from the most recently used to the least recently used. `get` of a present key counts as a use." + CODE_TAIL,
 """
from solution import LRUCache
from collections import OrderedDict
import random
class Ref:
    def __init__(s, c): s.c = c; s.d = OrderedDict()
    def get(s, k):
        if k not in s.d: return -1
        s.d.move_to_end(k); return s.d[k]
    def put(s, k, v):
        if k in s.d: s.d.move_to_end(k)
        s.d[k] = v
        if len(s.d) > s.c: s.d.popitem(last=False)
    def keys(s): return list(reversed(s.d.keys()))
c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); assert c.get(2) == -1; assert c.keys() == [3, 1]
random.seed(5)
for _ in range(200):
    cap = random.randint(1, 5); a, b = LRUCache(cap), Ref(cap)
    for _ in range(60):
        k = random.randint(0, 7)
        if random.random() < 0.5:
            v = random.randint(0, 99); a.put(k, v); b.put(k, v)
        else:
            assert a.get(k) == b.get(k)
        assert list(a.keys()) == b.keys()
print("PASS")
"""),
 ("c_kth", "Write a function `kth_smallest_pair_distance(nums, k)` that returns the k-th smallest value (k is 1-based) among the absolute differences "
  "|nums[i] - nums[j]| over all pairs of indices i < j. It must run fast for 30000 numbers (an O(n^2) approach is too slow)." + CODE_TAIL,
 """
from solution import kth_smallest_pair_distance as f
import random
def ref(nums, k):
    a = sorted(nums); n = len(a); lo, hi = 0, a[-1] - a[0]
    while lo < hi:
        mid = (lo + hi) // 2; cnt = 0; j = 0
        for i in range(n):
            while a[i] - a[j] > mid: j += 1
            cnt += i - j
        if cnt >= k: hi = mid
        else: lo = mid + 1
    return lo
def brute(nums, k):
    return sorted(abs(x - y) for i, x in enumerate(nums) for y in nums[i+1:])[k-1]
assert f([1, 3, 1], 1) == 0 and f([1, 1, 1], 2) == 0 and f([1, 6, 1], 3) == 5
random.seed(6)
for _ in range(400):
    n = random.randint(2, 12); nums = [random.randint(0, 30) for _ in range(n)]; k = random.randint(1, n*(n-1)//2)
    assert f(list(nums), k) == brute(nums, k), (nums, k)
nums = [random.randint(0, 10**6) for _ in range(30000)]
for k in (1, 123456, 225000000, 449985000):
    assert f(list(nums), k) == ref(nums, k), k
print("PASS")
"""),
 ("c_paren", "Write a function `longest_valid_parentheses(s)` that, for a string s made only of '(' and ')', returns the length of the longest "
  "contiguous substring that is a valid (well-formed, balanced) parentheses string. It must handle strings of 200000 characters quickly." + CODE_TAIL,
 """
from solution import longest_valid_parentheses as f
import random
def brute(s):
    best = 0
    for i in range(len(s)):
        bal = 0
        for j in range(i, len(s)):
            bal += 1 if s[j] == "(" else -1
            if bal < 0: break
            if bal == 0: best = max(best, j - i + 1)
    return best
def ref(s):
    st = [-1]; best = 0
    for i, ch in enumerate(s):
        if ch == "(": st.append(i)
        else:
            st.pop()
            if not st: st.append(i)
            else: best = max(best, i - st[-1])
    return best
assert f("") == 0 and f("(()") == 2 and f(")()())") == 4 and f("()(())") == 6 and f("))((") == 0
random.seed(7)
for _ in range(600):
    s = "".join(random.choice("()") for _ in range(random.randint(0, 14))); assert f(s) == brute(s), s
big = "".join(random.choice("(((()))") for _ in range(200000)); assert f(big) == ref(big)
print("PASS")
"""),
 ("c_window", "Write a function `min_window(s, t)` that returns the shortest substring of s that contains every character of t, counting multiplicity "
  "(if t has two 'a', the window needs two 'a'). If several windows have the shortest length return the leftmost; if there is none (or t is empty) return ''. "
  "It must handle strings of 200000 characters quickly." + CODE_TAIL,
 """
from solution import min_window as f
import random
from collections import Counter
def brute(s, t):
    if not t: return ""
    need = Counter(t); best = ""
    for i in range(len(s)):
        for j in range(i + len(t), len(s) + 1):
            if not (need - Counter(s[i:j])):
                if best == "" or j - i < len(best): best = s[i:j]
                break
    return best
def ref(s, t):
    if not t: return ""
    need = Counter(t); missing = len(t); l = 0; best = (10**9, 0)
    for r, ch in enumerate(s):
        if need[ch] > 0: missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[l]] < 0: need[s[l]] += 1; l += 1
            if r - l + 1 < best[0]: best = (r - l + 1, l)
            need[s[l]] += 1; missing += 1; l += 1
    return "" if best[0] == 10**9 else s[best[1]:best[1] + best[0]]
assert f("ADOBECODEBANC", "ABC") == "BANC" and f("a", "aa") == "" and f("aa", "aa") == "aa" and f("abc", "") == "" and f("ab", "b") == "b"
random.seed(8)
for _ in range(800):
    s = "".join(random.choice("abc") for _ in range(random.randint(0, 12))); t = "".join(random.choice("abc") for _ in range(random.randint(0, 4)))
    assert f(s, t) == brute(s, t), (s, t)
s = "".join(random.choice("abcdefgh") for _ in range(200000)); t = "abcdefgh" * 3
assert f(s, t) == ref(s, t)
print("PASS")
"""),
]

def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None

def run_tests(code, test_src, timeout=90):
    if WORK:
        os.makedirs(WORK, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK) as d:
        open(d + "/solution.py", "w").write(code)
        open(d + "/test_main.py", "w").write(test_src)
        boot = 'import sys; sys.path.insert(0, "/tmp/work"); exec(compile(open("/tmp/work/test_main.py").read(), "test_main", "exec"))'
        cmd = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc", "--ro-bind", d, "/tmp/work", "--chdir", "/tmp/work",
               "--unshare-all", "--die-with-parent", "prlimit", "--as=3221225472", "--cpu=%d" % timeout, "python3", "-I", "-c", boot]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout + 15, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        ok = r.returncode == 0 and "PASS" in r.stdout
        return ok, ("" if ok else (r.stderr.strip().splitlines() or ["failed"])[-1][:200])

# ---------------------------------------------------------------- instruction following
def check_four(text):
    t = text.strip()
    s = [x for x in re.split(r"(?<=[.!?])\s+", t) if x]
    return len(s) == 4 and "," not in t and re.search(r"horizon[.!?]?[\"')]*$", s[-1], re.I) is not None

def check_fruits(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return len(lines) == 5 and all(re.fullmatch(r"- [a-z][a-z ]*", l.strip()) for l in lines) and len({l.strip() for l in lines}) == 5

def check_json(text):
    try:
        o = json.loads(text.strip())
    except Exception:
        return False
    return isinstance(o, dict) and set(o) == {"name", "age"} and isinstance(o["name"], str) and isinstance(o["age"], int) \
        and not isinstance(o["age"], bool) and 20 <= o["age"] <= 40

IF = [
 ("i_four", "Write exactly four sentences about the sea. Do not use any commas. The last word of the fourth sentence must be \"horizon\". Output only the sentences.", check_four),
 ("i_fruits", "List exactly five different fruits, one per line, each line starting with \"- \" followed by the fruit's name in lowercase letters. Output nothing else.", check_fruits),
 ("i_json", "Reply with only a JSON object, with no markdown fences and no other text, that has exactly the keys \"name\" (a string) and \"age\" (an integer between 20 and 40) for an imaginary person.", check_json),
]

# ---------------------------------------------------------------- driver
def ask(port, prompt, seed, max_tokens):
    req = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed}
    r = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(req).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=3 * 3600))
    m = j["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", j["choices"][0]["finish_reason"], j["usage"]["completion_tokens"]

def main():
    port, out, sample = sys.argv[1], sys.argv[2], int(sys.argv[3])
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["task"], json.loads(l)["sample"]) for l in open(out)}
    tasks = [(n, p, "num", a) for n, p, a in NUMERIC] + [(n, p, "code", t) for n, p, t in CODE] + [(n, p, "if", c) for n, p, c in IF]
    for idx, (name, prompt, kind, extra) in enumerate(tasks):
        if (name, sample) in done:
            continue
        t0 = time.time()
        content, reasoning, finish, ntok = ask(port, prompt, 1000 * sample + idx, max_tokens)
        if kind == "num":
            got = last_int(content) if finish == "stop" else None
            ok, detail = got == extra, "got %s want %s" % (got, extra)
        elif kind == "code":
            code = extract_code(content) if finish == "stop" else None
            ok, detail = run_tests(code, extra) if code else (False, "no code" if finish == "stop" else "truncated")
        else:
            ok, detail = (finish == "stop" and extra(content)), ("" if finish == "stop" else "truncated")
        rec = {"task": name, "sample": sample, "ok": bool(ok), "detail": detail, "finish": finish, "tokens": ntok,
               "seconds": round(time.time() - t0, 1), "reasoning_chars": len(reasoning), "answer": content}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("%-10s %s  %5d tokens %5.0f s  %s" % (name, "PASS" if ok else "fail", ntok, rec["seconds"], detail if not ok else ""), flush=True)

if __name__ == "__main__":
    main()
