# 50 factual questions with short unambiguous answers, asked one by one (temperature 0). Reasoning models are asked not
# to think when their template allows it (THINK=off); LFM2.5 always thinks, so it gets a larger token limit.
# usage: qknow.py <port> <results.json> <think on|off> [max_tokens]   (against a running llama-server)
import json, re, sys, time, unicodedata, urllib.request

QA = [
 ("What is the capital of Burkina Faso?", ["ouagadougou"]),
 ("Who wrote the novel 'The Master and Margarita'?", ["bulgakov"]),
 ("What is the atomic number of tungsten?", ["74"]),
 ("In what year was the Battle of Lepanto fought?", ["1571"]),
 ("Which composer wrote the opera 'Wozzeck'?", ["berg"]),
 ("What is the largest moon of Neptune?", ["triton"]),
 ("Which river flows through Baghdad?", ["tigris"]),
 ("Who painted 'The Garden of Earthly Delights'?", ["bosch"]),
 ("What is the capital of Kyrgyzstan?", ["bishkek"]),
 ("Which element has the chemical symbol Sb?", ["antimony"]),
 ("Who wrote the novel 'Pedro Paramo'?", ["rulfo"]),
 ("What is the capital of the Chinese province of Gansu?", ["lanzhou"]),
 ("Which treaty of 1648 ended the Thirty Years' War?", ["westphalia"]),
 ("In what year was the Peace of Augsburg signed?", ["1555"]),
 ("At which battle of 1260 did the Mamluks defeat the Mongols?", ["ain jalut", "ayn jalut", "ain jalud", "goliath"]),
 ("Who was the first Secretary-General of the United Nations?", ["trygve"]),
 ("Who painted 'The Arnolfini Portrait'?", ["van eyck"]),
 ("Who wrote 'Godel, Escher, Bach: An Eternal Golden Braid'?", ["hofstadter"]),
 ("Which Roman emperor directly succeeded Commodus in 193 AD?", ["pertinax"]),
 ("What is the largest city of Kazakhstan by population?", ["almaty"]),
 ("Which enzyme unwinds the DNA double helix during replication?", ["helicase"]),
 ("What is the SI unit of magnetic flux density?", ["tesla"]),
 ("Who directed the 1979 film 'Stalker'?", ["tarkovsky"]),
 ("Which chess opening begins 1.e4 e5 2.Nf3 Nc6 3.Bb5?", ["ruy lopez", "spanish"]),
 ("Who proved Fermat's Last Theorem?", ["wiles"]),
 ("What is the capital of Bhutan?", ["thimphu"]),
 ("Who wrote 'Critique of Pure Reason'?", ["kant"]),
 ("What is the largest lake in Africa by surface area?", ["victoria"]),
 ("What is the exact speed of light in vacuum in metres per second?", ["299792458"]),
 ("Who discovered penicillin?", ["fleming"]),
 ("Which Mughal emperor built the Taj Mahal?", ["shah jahan"]),
 ("What is the chemical formula of table salt?", ["nacl"]),
 ("Who composed 'The Rite of Spring'?", ["stravinsky"]),
 ("Which Dutch scientist built early practical microscopes and was the first to describe bacteria?", ["leeuwenhoek"]),
 ("In which Spanish town was the Treaty of Tordesillas signed?", ["tordesillas"]),
 ("Who is the author of 'Invisible Cities'?", ["calvino"]),
 ("What is the capital of Slovenia?", ["ljubljana"]),
 ("Which physicist formulated the uncertainty principle in 1927?", ["heisenberg"]),
 ("Who wrote the epic poem 'Os Lusiadas'?", ["camoes"]),
 ("What was the name of the first artificial satellite, launched in 1957?", ["sputnik"]),
 ("Which Byzantine emperor was defeated and captured at the Battle of Manzikert in 1071?", ["romanos", "romanus"]),
 ("In which year did the Taiping Rebellion begin?", ["1850", "1851"]),
 ("What is the longest river in Poland?", ["vistula", "wisla"]),
 ("Who discovered the neutron in 1932?", ["chadwick"]),
 ("Which Spanish painter created 'Las Meninas'?", ["velazquez"]),
 ("Which language family does Hungarian belong to?", ["uralic", "finno-ugric", "finno ugric", "finnic"]),
 ("Approximately what is the Chandrasekhar limit, in solar masses?", ["1.4"]),
 ("Which 1713 treaty, named for a Dutch city, ended the War of the Spanish Succession?", ["utrecht"]),
 ("Who wrote the novel 'The Man Without Qualities'?", ["musil"]),
]

def norm(t):
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()

def main():
    port, out, think = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    results = []
    t0 = time.time()
    for q, answers in QA:
        req = {"messages": [{"role": "user", "content": q + " Answer briefly."}], "max_tokens": max_tokens, "temperature": 0}
        if think == "off":
            req["chat_template_kwargs"] = {"enable_thinking": False}
        r = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(req).encode(), {"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(r, timeout=3600))
        text = j["choices"][0]["message"].get("content") or ""
        flat = re.sub(r"[ ,_'  ]", "", norm(text)) if answers[0].isdigit() else norm(text)
        hit = lambda a: (a in flat) if any(c.isdigit() for c in a) else re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", flat) is not None
        ok = j["choices"][0]["finish_reason"] == "stop" and any(hit(a) for a in answers)
        results.append({"q": q, "ok": ok, "answer": text.strip()[:300], "finish": j["choices"][0]["finish_reason"], "tokens": j["usage"]["completion_tokens"]})
    json.dump(results, open(out, "w"), indent=1)
    n = sum(r["ok"] for r in results)
    print("   %d/%d correct in %.0f s (%d tokens); missed: %s" % (n, len(results), time.time() - t0, sum(r["tokens"] for r in results),
          "; ".join("%s -> %s" % (r["q"][:50], (r["answer"] or "<empty>").replace("\n", " ")[:50]) for r in results if not r["ok"])), flush=True)

if __name__ == "__main__":
    main()
