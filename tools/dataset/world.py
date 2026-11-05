"""Builds the synth_v1 world from a seed: documents, media specs, facts, queries and dialogues.

Pure and deterministic (no I/O): every random choice comes from `random.Random` instances seeded
with "synth_v1:<seed>:<stream>", so editing one modality does not shift the others. Rendering of
media files lives in `render.py`; `generate.py` writes everything to disk.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from tools.dataset import content

NAME = "synth_v1"
QUERY_MEDIA_DIR = "queries/media"
REVISED = "Revised 2024 edition. This edition supersedes the archived 2019 edition."
ARCHIVED = "Archived 2019 edition, kept for reference only. Superseded by the revised 2024 edition."
REVISED_MARK = "(Figure checked for the revised 2024 edition.)"
ARCHIVED_MARK = "(Figure as printed in the archived 2019 edition.)"

# Which entity of each type hosts a dialogue, which carry near-duplicates, which facts other items rely on.
DIALOGUE_HOSTS = {"lighthouse": 0, "observatory": 0, "company": 1, "festival": 0, "ferry": 0, "museum": 2, "crop": 0, "species": 0, "railway": 0}
NEAR_DUPLICATES = (("lighthouse", 1, "steps"), ("ferry", 1, "fare"), ("museum", 1, "fee"), ("railway", 2, "stations"))
EXTRA_FORCED = {("company", 0): ("flagship",), ("festival", 1): ("month",), ("company", 1): ("employees",)}

MONTH_ABBR = tuple(m[:3] for m in content.MONTHS)
FALLBACK = {
    "overview": ("notes", "place"),
    "history": ("notes", "people"),
    "flavor": ("place", "notes"),
    "people": ("notes", "visiting"),
    "place": ("notes", "overview"),
    "visiting": ("notes", "place"),
    "notes": ("overview", "history"),
}


# --------------------------------------------------------------------------- values


@dataclass(frozen=True)
class Value:
    text: str
    answers: tuple[str, ...]
    keywords: tuple[str, ...]
    regex: str
    spoken: str = ""
    number: float | None = None


def _dedupe(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for it in items:
        if it and it not in out:
            out.append(it)
    return tuple(out)


def word_rx(text: str) -> str:
    """Case-insensitive-friendly pattern for a word/phrase, tolerant to spacing, hyphens and curly apostrophes."""
    tokens = [re.escape(t).replace("'", "['’]?") for t in re.split(r"[\s-]+", text.strip()) if t]
    return r"(?<![A-Za-z])" + r"[\s-]*".join(tokens) + r"(?![A-Za-z])"


def int_rx(n: int) -> str:
    d = str(n)
    core = f"{d[:-3]}[,.\\s]?{d[-3:]}" if n >= 1000 else d
    return rf"(?<!\d){core}(?!\d)"


def dec_rx(x: float, places: int) -> str:
    a, b = f"{x:.{places}f}".split(".")
    if places == 2 and b.endswith("0"):
        b = b[0] + "0?"
    return rf"(?<!\d){a}[.,]{b}(?!\d)"


def v_int(n: int, unit: str = "") -> Value:
    disp = f"{n:,}" if n >= 10000 else str(n)
    text = f"{disp}%" if unit == "%" else (f"{disp} {unit}" if unit else disp)
    return Value(text, _dedupe((text, disp, str(n))), (disp,), int_rx(n), spoken=text.replace("%", " percent"), number=n)


def v_dec(x: float, unit: str = "", places: int = 1) -> Value:
    num = f"{x:.{places}f}"
    text = f"{num} {unit}" if unit else num
    return Value(text, _dedupe((text, num)), (num,), dec_rx(x, places), spoken=text, number=x)


def v_money(x: float) -> Value:
    num = f"{x:.2f}"
    text = f"{num} Saltmere crowns"
    return Value(text, (text, num), (num,), dec_rx(x, 2), spoken=text, number=x)


def v_year(y: int) -> Value:
    return Value(str(y), (str(y),), (str(y),), rf"(?<!\d){y}(?!\d)", spoken=year_words(y), number=y)


def v_person(full: str) -> Value:
    last = full.split()[-1]
    return Value(full, (full, last), (last,), word_rx(last), spoken=full)


def v_word(text: str, keyword: str = "") -> Value:
    kw = keyword or text
    return Value(text, _dedupe((text, kw)), (kw,), word_rx(kw), spoken=text)


def v_month(month: str) -> Value:
    abbr = month[:3]
    rest = f"(?:{month[3:]})?" if month[3:] else ""
    return Value(month, (month,), (abbr,), rf"(?<![A-Za-z]){abbr}{rest}(?![A-Za-z])", spoken=month)


def v_weekday(day: str) -> Value:
    return Value(day, (day, day[:3]), (day[:3],), rf"(?<![A-Za-z]){day[:3]}(?:{day[3:]})?s?(?![A-Za-z])", spoken=day)


_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def year_words(y: int) -> str:
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{num_words(hi)} hundred"
    return f"{num_words(hi)} {'oh ' + num_words(lo) if lo < 10 else num_words(lo)}"


def lcfirst(s: str) -> str:
    return s[:1].lower() + s[1:]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# --------------------------------------------------------------------------- records


@dataclass
class Fact:
    id: str
    modality: str
    source: str
    statement: str
    value: Value
    question: str
    location: dict[str, Any] = field(default_factory=dict)
    distractor_ids: list[str] = field(default_factory=list)
    key: str = ""
    entity: str = ""
    followup: str = ""
    es: str = ""
    ja: str = ""
    sentence: str = ""
    phrasing: int = 0
    template: content.Template | None = None

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modality": self.modality,
            "source": self.source,
            "location": self.location,
            "statement": self.statement,
            "answers": list(self.value.answers),
            "distractor_ids": sorted(self.distractor_ids),
        }


@dataclass
class Para:
    sentences: list[str]
    fact: Fact | None = None

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class Section:
    heading: str
    paras: list[Para]


@dataclass
class Doc:
    path: str
    kind: str
    entity: content.Entity
    title: str
    subtitle: str
    sections: list[Section]
    edition: str = ""
    see_also: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)

    def markdown(self) -> str:
        parts = [f"# {self.title}", f"_{self.subtitle}_"]
        if self.edition:
            parts.append(f"_{self.edition}_")
        for sec in self.sections:
            parts.append(f"## {sec.heading}")
            parts.extend(p.text for p in sec.paras)
        if self.see_also:
            parts.append("## See also")
            parts.append("\n".join(f"- {t}" for t in self.see_also))
        return "\n\n".join(parts) + "\n"


@dataclass
class TextLine:
    text: str
    size: int
    color: tuple[int, int, int]
    bold: bool = True


@dataclass
class Chart:
    path: str
    title: str
    kind: str  # bar | hbar | line | pie
    labels: list[str]
    values: list[float]
    ylabel: str
    decimals: int = 0
    facts: list[Fact] = field(default_factory=list)

    @property
    def winner(self) -> int:
        return max(range(len(self.values)), key=self.values.__getitem__)


@dataclass
class Slide:
    path: str
    kind: str  # colour | spec | sign | query
    bg: tuple[int, int, int]
    lines: list[TextLine]
    pictogram: str
    pict_color: tuple[int, int, int]
    accent: tuple[int, int, int]
    size: tuple[int, int] = (640, 480)
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Photo:
    path: str
    shape: str
    color: str
    background: str
    seed: int
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Audio:
    path: str
    text: str
    spoken: str
    voice: str
    speed: int
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Video:
    path: str
    kind: str  # bulletin | motion
    header: str = ""
    lines: list[str] = field(default_factory=list)
    shape: str = ""
    color: str = ""
    motion: str = ""  # left_to_right | right_to_left | bottom_to_top | diagonal
    duration: float = 7.0
    fps: int = 4
    seed: int = 0
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Query:
    id: str
    type: str
    lang: str
    text: str
    attachment: str | None
    expected_sources: list[str]
    answer_keywords: list[str]
    answer_regex: str
    must_refuse: bool
    fact_ids: list[str]

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "lang": self.lang,
            "text": self.text,
            "attachment": self.attachment,
            "expected_sources": self.expected_sources,
            "answer_keywords": self.answer_keywords,
            "answer_regex": self.answer_regex,
            "must_refuse": self.must_refuse,
            "fact_ids": self.fact_ids,
        }


@dataclass
class World:
    seed: int
    docs: list[Doc] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    slides: list[Slide] = field(default_factory=list)
    photos: list[Photo] = field(default_factory=list)
    audios: list[Audio] = field(default_factory=list)
    videos: list[Video] = field(default_factory=list)
    query_slides: list[Slide] = field(default_factory=list)
    query_audios: list[Audio] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    dialogues: list[dict[str, Any]] = field(default_factory=list)
    probe: dict[str, Any] = field(default_factory=dict)
    sample_subset: list[str] = field(default_factory=list)

    def fact(self, fid: str) -> Fact:
        return next(f for f in self.facts if f.id == fid)


# --------------------------------------------------------------------------- builder


class Names:
    """Unique invented people. Surnames never repeat, so a surname alone identifies a person."""

    def __init__(self, rng: random.Random) -> None:
        sur = [h + t for h in content.SURNAME_HEADS for t in content.SURNAME_TAILS]
        sur = [s for s in sur if self._ok(s)]
        rng.shuffle(sur)
        self._rng = rng
        self._sur = sur
        self._i = 0

    @staticmethod
    def _ok(s: str) -> bool:
        low = s.lower()
        if re.search(r"(.)\1\1", low) or "owow" in low or "rrm" in low or not 5 <= len(s) <= 11:
            return False
        return not any(w in low for w in ("red", "blue", "gold", "grey", "pink"))

    def person(self) -> str:
        s = self._sur[self._i]
        self._i += 1
        return f"{self._rng.choice(content.FIRST_NAMES)} {s}"


def entity_slots(dt: content.DocType, e: content.Entity) -> dict[str, str]:
    the = f"the {e.name}" if e.article else e.name
    return {
        "name": e.name,
        "the_name": the,
        "The_name": the[:1].upper() + the[1:],
        "place": e.island,
        "Place": e.island,
        "town": content.ISLANDS[e.island][0],
        "the_kind": dt.kind_noun,
    }


class Builder:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.world = World(seed=seed)
        self.names = Names(self._rng("names"))
        self._fid = 0
        self._unique: dict[tuple[str, str], list[str]] = {}
        self.forced: dict[tuple[str, int], tuple[str, ...]] = {}
        for dt in content.DOC_TYPES:
            idx = DIALOGUE_HOSTS[dt.kind]
            self.forced[(dt.kind, idx)] = dt.dialogue
        for kind, idx, key in NEAR_DUPLICATES:
            self.forced[(kind, idx)] = tuple(dict.fromkeys((*self.forced.get((kind, idx), ()), key)))
        for k, keys in EXTRA_FORCED.items():
            self.forced[k] = tuple(dict.fromkeys((*self.forced.get(k, ()), *keys)))

    def _rng(self, stream: str) -> random.Random:
        return random.Random(f"{NAME}:{self.seed}:{stream}")

    def _new_fact(self, **kw: Any) -> Fact:
        self._fid += 1
        f = Fact(id=f"F{self._fid:03d}", **kw)
        self.world.facts.append(f)
        return f

    # ----------------------------------------------------------------- values

    def _value(self, rng: random.Random, dt: content.DocType, e: content.Entity, t: content.Template, differ_from: Value | None = None) -> Value:
        used = self._unique.setdefault((dt.kind, t.key), [])
        for _ in range(500):
            v = self._draw_value(rng, e, t.value)
            if v.text in used:
                continue
            close = differ_from is not None and differ_from.number is not None and v.number is not None
            if close and abs(v.number - differ_from.number) < max(0.1 * abs(differ_from.number), 1):  # type: ignore[operator,union-attr,arg-type]
                continue
            used.append(v.text)
            return v
        raise RuntimeError(f"could not draw a unique value for {dt.kind}.{t.key}")

    def _draw_value(self, rng: random.Random, e: content.Entity, spec: content.V) -> Value:
        k = spec.kind
        if k == "int":
            return v_int(rng.randint(int(spec.lo), int(spec.hi)), spec.unit)
        if k == "dec":
            while True:
                x = round(rng.uniform(spec.lo, spec.hi), 1)
                if round(x * 10) % 10:
                    return v_dec(x, spec.unit)
        if k == "money":
            return v_money(rng.randint(int(spec.lo), int(spec.hi) - 1) + rng.choice((0.25, 0.5, 0.75)))
        if k == "year":
            return v_year(rng.randint(int(spec.lo), int(spec.hi)))
        if k == "person":
            return v_person(self.names.person())
        if k == "pick":
            text, kw = rng.choice(content.POOLS[spec.pool])
            return v_word(text, kw)
        if k == "fixed":
            _, text, kw = next(x for x in e.fixed if x[0] == spec.pool)
            return v_word(text, kw)
        if k == "month":
            return v_month(rng.choice(content.MONTHS))
        if k == "town":
            return v_word(rng.choice(content.ISLANDS[e.island]))
        if k == "place":
            return v_word(e.island)
        raise ValueError(f"unknown value kind {k}")

    # ----------------------------------------------------------------- text

    def _fill(self, rng: random.Random, tpl: str, slots: dict[str, str], island: str) -> str:
        extra = {
            "adj": rng.choice(content.ADJECTIVES),
            "season": rng.choice(content.SEASONS),
            "Weather": rng.choice(content.WEATHER).capitalize(),
            "craft": rng.choice(content.CRAFTS),
            "food": rng.choice(content.FOODS),
            "town": rng.choice(content.ISLANDS[island]),
        }
        if "{person}" in tpl:
            extra["person"] = self.names.person()
        return tpl.format(**{**slots, **extra})

    def _paragraph(self, rng: random.Random, dt: content.DocType, pool: str, used: set[str], n: int, slots: dict[str, str], island: str) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            order = [pool, *FALLBACK[pool]]
            if pool != "flavor" and rng.random() < 0.2:
                order.insert(0, "flavor")
            tpl = None
            for name in order:
                src = dt.flavor if name == "flavor" else content.FILLER[name]
                avail = [s for s in src if s not in used]
                if avail:
                    tpl = rng.choice(avail)
                    used.add(tpl)
                    break
            if tpl is None:
                break
            out.append(self._fill(rng, tpl, slots, island))
        return out

    def _build_doc(self, rng: random.Random, dt: content.DocType, idx: int, e: content.Entity) -> Doc:
        slots = entity_slots(dt, e)
        path = f"corpus/text/{slugify(e.name)}.md"
        forced = self.forced.get((dt.kind, idx), ())
        k = max(len(forced), rng.randint(4, 6))
        rest = [t.key for t in dt.templates if t.key not in forced]
        chosen = set(forced) | set(rng.sample(rest, k - len(forced)))
        doc = Doc(path=path, kind=dt.kind, entity=e, title=e.name, subtitle=dt.subtitle.format(**slots), sections=[])
        for t in dt.templates:
            if t.key not in chosen:
                continue
            val = self._value(rng, dt, e, t)
            ph = rng.randrange(2)
            sentence = t.phrasings[ph].format(value=val.text, **slots)
            doc.facts.append(
                self._new_fact(
                    modality="text",
                    source=path,
                    statement=sentence,
                    value=val,
                    question=t.question.format(**slots),
                    key=t.key,
                    entity=e.name,
                    followup=t.followup,
                    es=t.es.format(**slots) if t.es else "",
                    ja=t.ja.format(**slots) if t.ja else "",
                    sentence=sentence,
                    phrasing=ph,
                    template=t,
                )
            )
        used: set[str] = set()
        for heading, pool in dt.headings:
            sfacts: list[Fact | None] = [f for f in doc.facts if f.template is not None and f.template.section == pool]
            base = 2 if pool in ("overview", "flavor") else rng.choice((1, 2, 2))
            slots_list: list[Fact | None] = sfacts + [None] * max(0, base - len(sfacts))
            rng.shuffle(slots_list)
            paras = []
            for f in slots_list:
                sents = self._paragraph(rng, dt, pool, used, rng.randint(5, 7), slots, e.island)
                if f is not None:
                    sents.insert(rng.randint(0, min(2, len(sents))), f.sentence)
                paras.append(Para(sents, f))
            doc.sections.append(Section(heading.format(**slots), paras))
        return doc

    def _near_duplicate(self, rng: random.Random, dt: content.DocType, base: Doc, key: str) -> Doc:
        bf = next(f for f in base.facts if f.key == key)
        assert bf.template is not None
        e = base.entity
        slots = entity_slots(dt, e)
        new_val = self._value(rng, dt, e, bf.template, differ_from=bf.value)
        new_sentence = bf.template.phrasings[1 - bf.phrasing].format(value=new_val.text, **slots)
        path = base.path.replace(".md", "_2019_edition.md")
        variant = Doc(path=path, kind=base.kind, entity=e, title=f"{e.name} (archived 2019 edition)", subtitle=base.subtitle, sections=[], edition=ARCHIVED)
        vf = self._new_fact(
            modality="text",
            source=path,
            statement=f"{new_sentence} {ARCHIVED_MARK}",
            value=new_val,
            question=f"According to the archived 2019 edition of the guide, {lcfirst(bf.question)}",
            key=key,
            entity=e.name,
            sentence=new_sentence,
            phrasing=1 - bf.phrasing,
            template=bf.template,
            distractor_ids=[bf.id],
        )
        variant.facts.append(vf)
        bf.distractor_ids.append(vf.id)
        bf.statement = f"{bf.sentence} {REVISED_MARK}"
        bf.question = f"According to the revised 2024 edition of the guide, {lcfirst(bf.question)}"
        base.title = f"{e.name} (revised 2024 edition)"
        base.edition = REVISED
        for sec in base.sections:
            vparas = []
            for p in sec.paras:
                if p.fact is bf:
                    i = p.sentences.index(bf.sentence)
                    p.sentences.insert(i + 1, REVISED_MARK)
                    others = [s for s in p.sentences if s not in (bf.sentence, REVISED_MARK)]
                    rng.shuffle(others)
                    pos = rng.randint(0, min(2, len(others)))
                    vparas.append(Para([*others[:pos], new_sentence, ARCHIVED_MARK, *others[pos:]], vf))
                    continue
                others = [s for s in p.sentences if p.fact is None or s != p.fact.sentence]
                rng.shuffle(others)
                vparas.append(Para(others))
            variant.sections.append(Section(sec.heading, vparas))
        return variant

    def build_text(self) -> None:
        rng = self._rng("text")
        docs: list[Doc] = []
        for dt in content.DOC_TYPES:
            for idx, e in enumerate(dt.entities):
                docs.append(self._build_doc(rng, dt, idx, e))
        variants = []
        for kind, idx, key in NEAR_DUPLICATES:
            dt = next(d for d in content.DOC_TYPES if d.kind == kind)
            base = next(d for d in docs if d.entity is dt.entities[idx])
            variants.append(self._near_duplicate(rng, dt, base, key))
        titles = [d.entity.name for d in docs]
        for d in docs:
            same_island = [x.entity.name for x in docs if x.entity.island == d.entity.island and x is not d]
            others = [t for t in titles if t != d.entity.name and t not in same_island]
            picks = same_island[:1] + rng.sample(others, 3 - len(same_island[:1]))
            d.see_also = picks
        for v in variants:
            v.see_also = next(d for d in docs if d.entity is v.entity).see_also
        self.world.docs = docs + variants
        for d in self.world.docs:
            self._set_chunk_hints(d)

    @staticmethod
    def _set_chunk_hints(doc: Doc) -> None:
        texts = [p.text for s in doc.sections for p in s.paras]
        for s in doc.sections:
            for p in s.paras:
                if p.fact is None:
                    continue
                words = p.text.split()
                n = 8
                while n < len(words) and sum(t.startswith(" ".join(words[:n])) for t in texts) > 1:
                    n += 1
                p.fact.location = {"chunk_hint": " ".join(words[:n]), "section": s.heading}

    # ----------------------------------------------------------------- images

    def build_images(self) -> None:
        rng = self._rng("images")
        self._build_charts(rng)
        self._build_slides(rng)
        self._build_photos(rng)
        self._build_query_slides()

    def _chart_values(self, rng: random.Random, n: int, lo: float, hi: float, decimals: int, avoid_winner: int | None = None) -> list[float]:
        winner = rng.choice([i for i in range(n) if i != avoid_winner])
        span = hi - lo
        vals = [round(rng.uniform(lo, lo + 0.75 * span), decimals) for _ in range(n)]
        vals[winner] = round(max(vals) + span * rng.uniform(0.12, 0.25), decimals)
        return vals

    def _build_charts(self, rng: random.Random) -> None:
        rail = [e.name for e in content.DOC_TYPES[8].entities]
        districts = ["Northreach", "Eastholm", "Southfold", "Westerby", "Midcairn"]
        years9 = [str(y) for y in range(2016, 2025)]
        defs: list[tuple[str, str, str, list[str], float, float, int, str, str]] = [
            (
                "glimmerfruit_2024",
                "Glimmerfruit harvest by district, 2024",
                "bar",
                districts,
                40,
                120,
                0,
                "tonnes",
                "Which district harvested the most Glimmerfruit in 2024?",
            ),
            (
                "glimmerfruit_2023",
                "Glimmerfruit harvest by district, 2023",
                "bar",
                districts,
                40,
                120,
                0,
                "tonnes",
                "Which district harvested the most Glimmerfruit in 2023?",
            ),
            (
                "clock_museum_2024",
                "Visitors to the Quillmere Clock Museum by month, 2024",
                "line",
                list(MONTH_ABBR),
                300,
                2400,
                0,
                "visitors",
                "In which month of 2024 did the Quillmere Clock Museum receive the most visitors?",
            ),
            (
                "clock_museum_2023",
                "Visitors to the Quillmere Clock Museum by month, 2023",
                "line",
                list(MONTH_ABBR),
                300,
                2400,
                0,
                "visitors",
                "In which month of 2023 did the Quillmere Clock Museum receive the most visitors?",
            ),
            (
                "brindle_sales",
                "Brindle Robotics unit sales by product, 2024",
                "hbar",
                ["Corvid-9", "Corvid-5", "Wrenbot", "Heron Arm", "Kelp Rover"],
                200,
                4000,
                0,
                "units sold",
                "Which Brindle Robotics product sold the most units in 2024?",
            ),
            (
                "saltwing_weekdays",
                "Saltwing Ferry Line passengers by weekday, 2024 average",
                "bar",
                [d[:3] for d in content.WEEKDAYS],
                120,
                900,
                0,
                "passengers",
                "On which day of the week does the Saltwing Ferry Line carry the most passengers?",
            ),
            (
                "island_rainfall",
                "Annual rainfall by island, 2024 (mm)",
                "bar",
                ["Kestrelmoor", "Orrinhal", "Duskwater", "Vellhaven", "Tamsin", "Corvane"],
                600,
                2400,
                0,
                "mm",
                "Which island had the highest annual rainfall in 2024?",
            ),
            (
                "ibis_sightings",
                "Saltmere ibis sightings reported by volunteers, 2016-2024",
                "line",
                years9,
                150,
                1300,
                0,
                "sightings",
                "In which year did volunteers report the most Saltmere ibis sightings?",
            ),
            (
                "lanterns",
                "Lanterns released at the Lantern Tide Festival, 2016-2024",
                "line",
                years9,
                800,
                6000,
                0,
                "lanterns",
                "In which year were the most lanterns released at the Lantern Tide Festival?",
            ),
            (
                "orrinhal_settlements",
                "Population of settlements on Orrinhal, 2024",
                "hbar",
                ["Port Ambrel", "Lowcairn", "Kettle Cove", "Brannockside", "Wythe End"],
                300,
                6000,
                0,
                "residents",
                "Which settlement on Orrinhal has the largest population?",
            ),
            (
                "tea_blends",
                "Tea blends sold in Port Ambrel shops, 2024 (share of sales)",
                "pie",
                ["Harbour Smoke", "Gorse Gold", "Fog Morning", "Ember Leaf"],
                10,
                40,
                0,
                "%",
                "Which tea blend had the largest share of sales in Port Ambrel shops in 2024?",
            ),
            (
                "fernmouth_waves",
                "Average wave height at the Fernmouth Beacon by month, 2024 (m)",
                "line",
                list(MONTH_ABBR),
                0.8,
                4.5,
                1,
                "m",
                "In which month of 2024 were the waves highest at the Fernmouth Beacon?",
            ),
            (
                "vellhaven_power",
                "Electricity sources on Vellhaven, 2024 (share)",
                "pie",
                ["Wind", "Tidal", "Solar", "Diesel"],
                8,
                40,
                0,
                "%",
                "What was the largest source of electricity on Vellhaven in 2024?",
            ),
            (
                "velmora_seasons",
                "Clear nights by season at the Velmora Observatory, 2024",
                "bar",
                ["Spring", "Summer", "Autumn", "Winter"],
                10,
                60,
                0,
                "clear nights",
                "In which season of 2024 did the Velmora Observatory have the most clear nights?",
            ),
            (
                "rail_punctuality",
                "Trains running on time by line, 2024 (%)",
                "hbar",
                rail,
                70,
                99,
                0,
                "% on time",
                "Which railway had the highest share of trains running on time in 2024?",
            ),
            (
                "fish_landings",
                "Fish landed by harbour, 2024 (tonnes)",
                "bar",
                ["Gullstead", "Port Ambrel", "Fernmouth", "Corvane Quay", "Halvric Bay"],
                200,
                1800,
                0,
                "tonnes",
                "Which harbour landed the most fish in 2024?",
            ),
            (
                "college_faculties",
                "Students by faculty at Quillmere College, 2024",
                "bar",
                ["Navigation", "Botany", "Music", "Engineering", "Law"],
                60,
                520,
                0,
                "students",
                "Which faculty at Quillmere College had the most students in 2024?",
            ),
            (
                "fernmouth_bikes",
                "Gullwing bicycle rentals in Fernmouth by month, 2024",
                "line",
                list(MONTH_ABBR),
                40,
                900,
                0,
                "rentals",
                "In which month of 2024 were bicycle rentals in Fernmouth highest?",
            ),
            (
                "kite_entries",
                "Kite entries by type at Duskwater Kite Week, 2024",
                "pie",
                ["Box", "Delta", "Dragon", "Stunt"],
                10,
                40,
                0,
                "%",
                "Which type of kite had the most entries at Duskwater Kite Week in 2024?",
            ),
            (
                "bird_vote",
                "Saltmere Bird of the Year vote, 2024",
                "hbar",
                ["Saltmere ibis", "Kestrelmoor cliff wren", "Vellhaven sand plover", "Corvane moth-hawk", "Storm lark"],
                200,
                3000,
                0,
                "votes",
                "Which bird won the Saltmere Bird of the Year vote in 2024?",
            ),
        ]
        prev_winner: dict[str, int] = {}
        for i, (slug, title, kind, labels, lo, hi, dec, unit, question) in enumerate(defs, start=1):
            family = re.sub(r"_20\d\d$", "", slug)
            vals = self._chart_values(rng, len(labels), lo, hi, dec, avoid_winner=prev_winner.get(family))
            if kind == "pie":
                total = sum(vals)
                vals = [float(max(1, round(100 * v / total))) for v in vals]
                w = max(range(len(vals)), key=vals.__getitem__)
                vals[w] += 100 - sum(vals)
            chart = Chart(path=f"corpus/images/chart_{i:02d}_{slug}.png", title=title, kind=kind, labels=labels, values=vals, ylabel=unit, decimals=dec)
            prev_winner[family] = chart.winner
            if labels == list(MONTH_ABBR):
                val = v_month(content.MONTHS[chart.winner])
            elif labels == [d[:3] for d in content.WEEKDAYS]:
                val = v_weekday(content.WEEKDAYS[chart.winner])
            else:
                val = v_word(labels[chart.winner])
            full = val.text
            num = f"{vals[chart.winner]:.{dec}f}"
            stmt = f'The chart "{title}" shows {full} with the highest value ({num} {unit}).'.replace(" %)", "%)")
            chart.facts.append(
                self._new_fact(modality="image", source=chart.path, statement=stmt, value=val, question=question, key=f"chart:{family}", entity=title)
            )
            self.world.charts.append(chart)
        self._link_family([c.facts[0] for c in self.world.charts])

    def _link_family(self, facts: list[Fact]) -> None:
        for f in facts:
            for g in facts:
                if f is not g and f.key == g.key:
                    f.distractor_ids.append(g.id)

    def _build_slides(self, rng: random.Random) -> None:
        palette = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown"]
        n = 0
        # (a) colour slides: four people, several objects each (same-person slides are labelled distractors)
        for count in (3, 3, 2, 2):
            person = self.names.person()
            objs = rng.sample(list(content.SLIDE_OBJECTS), count)
            cols = rng.sample(palette, count)
            group: list[Fact] = []
            for obj, col in zip(objs, cols, strict=True):
                n += 1
                first, last = person.split()
                path = f"corpus/images/slide_{n:02d}_{slugify(last)}_{slugify(obj)}.png"
                lines = [
                    TextLine(f"{person.upper()}'S", 34, (30, 30, 40)),
                    TextLine(obj.upper(), 46, (30, 30, 40)),
                    TextLine(f"IS {col.upper()}", 56, content.COLOURS[col]),
                ]
                slide = Slide(path=path, kind="colour", bg=(244, 241, 232), lines=lines, pictogram=obj, pict_color=content.COLOURS[col], accent=(60, 60, 70))
                f = self._new_fact(
                    modality="image",
                    source=path,
                    statement=f"{person}'s {obj} is {col}.",
                    value=v_word(col),
                    question=f"What colour is {person}'s {obj}?",
                    key=f"colour:{person}",
                    entity=person,
                )
                slide.facts.append(f)
                group.append(f)
                self.world.slides.append(slide)
            self._link_family(group)
        # (b) spec slides about entities from the text documents (attributes that no document states)
        ents = {dt.kind: dt.entities for dt in content.DOC_TYPES}
        flight = rng.randint(31, 74)
        weight = round(rng.uniform(1.2, 4.8), 1)
        parade_h, parade_m = rng.choice((17, 18, 19)), rng.choice((15, 30, 45))
        closed = rng.choice(content.WEEKDAYS[:5])
        bikes = rng.randint(12, 36)
        last_h, last_m = rng.choice((21, 22, 23)), rng.choice((10, 15, 20, 40))
        open_night = rng.choice([d for d in content.WEEKDAYS[:5] if d != closed])
        food = rng.choice((("MUD SHRIMP", "mud shrimp", "shrimp"), ("SAND EELS", "sand eels", "sand eels"), ("RAGWORMS", "ragworms", "ragworms")))
        specs: list[tuple[str, str, list[str], Value, str, str, str]] = [
            (
                "corvid9",
                "CORVID-9 SURVEY DRONE",
                ["MAX FLIGHT TIME", f"{flight} MIN"],
                v_int(flight, "minutes"),
                f"The Corvid-9 survey drone has a maximum flight time of {flight} minutes.",
                "What is the maximum flight time of the Corvid-9 survey drone?",
                "drone",
            ),
            (
                "tidewatch",
                "TIDEWATCH BAROMETER",
                ["WEIGHT", f"{weight:.1f} KG"],
                v_dec(weight, "kg"),
                f"The Tidewatch barometer weighs {weight:.1f} kg.",
                "How much does the Tidewatch barometer weigh?",
                "gauge",
            ),
            (
                "lantern_parade",
                ents["festival"][0].name.upper(),
                ["OPENING PARADE", f"STARTS {parade_h}:{parade_m:02d}"],
                Value(
                    f"{parade_h}:{parade_m:02d}",
                    (f"{parade_h}:{parade_m:02d}", f"{parade_h - 12}:{parade_m:02d} pm"),
                    (f"{parade_m:02d}",),
                    rf"(?<!\d)({parade_h}|{parade_h - 12})[:.]{parade_m:02d}(?!\d)",
                ),
                f"The opening parade of the {ents['festival'][0].name} starts at {parade_h}:{parade_m:02d}.",
                f"What time does the opening parade of the {ents['festival'][0].name} start?",
                "lantern",
            ),
            (
                "clock_museum",
                ents["museum"][1].name.upper(),
                ["CLOSED ON", f"{closed.upper()}S"],
                v_weekday(closed),
                f"The {ents['museum'][1].name} is closed on {closed}s.",
                f"On which day of the week is the {ents['museum'][1].name} closed?",
                "clock",
            ),
            (
                "saltwing_bikes",
                ents["ferry"][0].name.upper(),
                ["BICYCLES PER SAILING", f"MAX {bikes}"],
                v_int(bikes, "bicycles"),
                f"The {ents['ferry'][0].name} carries at most {bikes} bicycles per sailing.",
                f"How many bicycles can be carried per sailing on the {ents['ferry'][0].name}?",
                "bicycle",
            ),
            (
                "saffren_last_train",
                ents["railway"][2].name.upper(),
                ["LAST TRAIN", f"{last_h}:{last_m:02d}"],
                Value(
                    f"{last_h}:{last_m:02d}",
                    (f"{last_h}:{last_m:02d}", f"{last_h - 12}:{last_m:02d} pm"),
                    (f"{last_m:02d}",),
                    rf"(?<!\d)({last_h}|{last_h - 12})[:.]{last_m:02d}(?!\d)",
                ),
                f"The last train on the {ents['railway'][2].name} leaves at {last_h}:{last_m:02d}.",
                f"What time is the last train on the {ents['railway'][2].name}?",
                "train",
            ),
            (
                "velmora_open_night",
                ents["observatory"][0].name.upper(),
                ["PUBLIC OPEN NIGHT", f"EVERY {open_night.upper()}"],
                v_weekday(open_night),
                f"The {ents['observatory'][0].name} holds its public open night every {open_night}.",
                f"On which day of the week is the public open night at the {ents['observatory'][0].name}?",
                "telescope",
            ),
            (
                "ibis_food",
                ents["species"][0].name.upper(),
                ["FAVOURITE FOOD", food[0]],
                v_word(food[1], food[2]),
                f"The favourite food of the {ents['species'][0].name} is {food[1]}.",
                f"What is the favourite food of the {ents['species'][0].name}?",
                "bird",
            ),
        ]
        for slug, title, rest, val, stmt, question, pict in specs:
            n += 1
            path = f"corpus/images/slide_{n:02d}_{slug}.png"
            lines = [TextLine(title, 30, (255, 255, 255)), TextLine(rest[0], 32, (40, 40, 50)), TextLine(rest[1], 60, (200, 60, 30))]
            slide = Slide(path=path, kind="spec", bg=(250, 250, 247), lines=lines, pictogram=pict, pict_color=(70, 110, 160), accent=(36, 70, 120))
            slide.facts.append(
                self._new_fact(modality="image", source=path, statement=stmt, value=val, question=question, key=f"spec:{slug}", entity=title.title())
            )
            self.world.slides.append(slide)
        # (c) signposts: distances from a town to an entity
        sign_defs = [
            ("Gullstead", ents["lighthouse"][0].name),
            ("Lowcairn", ents["company"][0].name + " works"),
            ("Eddersby", "Duskwater Kite Week field"),
            ("Calloway", ents["museum"][0].name),
        ]
        for town, dest in sign_defs:
            n += 1
            km = round(rng.uniform(1.5, 19.5), 1)
            if not round(km * 10) % 10:
                km += 0.5
            path = f"corpus/images/slide_{n:02d}_sign_{slugify(town)}.png"
            lines = [TextLine(town.upper(), 40, (255, 255, 255)), TextLine(dest.upper(), 28, (255, 255, 255)), TextLine(f"{km:.1f} KM", 52, (255, 230, 120))]
            slide = Slide(path=path, kind="sign", bg=(196, 222, 240), lines=lines, pictogram="arrow", pict_color=(255, 255, 255), accent=(30, 100, 60))
            slide.facts.append(
                self._new_fact(
                    modality="image",
                    source=path,
                    statement=f"A signpost in {town} gives the distance to {dest} as {km:.1f} km.",
                    value=v_dec(km, "km"),
                    question=f"According to the signpost in {town}, how far is it to {dest}?",
                    key="sign",
                    entity=dest,
                )
            )
            self.world.slides.append(slide)

    def _build_photos(self, rng: random.Random) -> None:
        combos = [
            ("triangle", "blue", "sunset"),
            ("triangle", "red", "grass"),
            ("circle", "blue", "beach"),
            ("star", "yellow", "night"),
            ("square", "green", "fog"),
            ("hexagon", "purple", "underwater"),
            ("crescent", "orange", "night"),
            ("circle", "red", "snow"),
            ("diamond", "white", "underwater"),
            ("triangle", "yellow", "fog"),
            ("circle", "green", "sunset"),
            ("star", "blue", "grass"),
            ("square", "orange", "beach"),
            ("diamond", "purple", "snow"),
        ]
        facts = []
        for i, (shape, col, bg) in enumerate(combos, start=1):
            path = f"corpus/images/photo_{i:02d}.jpg"
            photo = Photo(path=path, shape=shape, color=col, background=bg, seed=rng.randrange(1 << 30))
            desc = f"a large {col} {shape}"
            f = self._new_fact(
                modality="image",
                source=path,
                statement=f"A photo-like image of {desc} on {content.BACKGROUNDS[bg]}.",
                value=v_word(shape),
                question=f"Which picture shows {desc} on {content.BACKGROUNDS[bg]}?",
                key=f"photo:{col}:{shape}",
                entity=desc,
            )
            photo.facts.append(f)
            facts.append(f)
            self.world.photos.append(photo)
        for f in facts:  # same colour or same shape = distractor
            _, col, shape = f.key.split(":")
            f.distractor_ids = [g.id for g in facts if g is not f and (g.key.split(":")[1] == col or g.key.split(":")[2] == shape)]

    def _build_query_slides(self) -> None:
        ents = {dt.kind: dt.entities for dt in content.DOC_TYPES}
        defs = [
            (
                "sign_velmora",
                [TextLine("WELCOME TO", 30, (255, 255, 255)), TextLine(ents["observatory"][0].name.upper(), 36, (255, 255, 255))],
                "telescope",
                (120, 70, 40),
            ),
            ("corvid9_photo", [TextLine("CORVID-9", 56, (30, 30, 40))], "drone", (90, 90, 100)),
            ("kestrelmoor_light", [TextLine(ents["lighthouse"][0].name.upper(), 36, (30, 30, 40))], "lighthouse", (200, 50, 50)),
        ]
        for slug, lines, pict, accent in defs:
            self.world.query_slides.append(
                Slide(path=f"{QUERY_MEDIA_DIR}/q_{slug}.png", kind="query", bg=(236, 240, 244), lines=lines, pictogram=pict, pict_color=accent, accent=accent)
            )

    # ----------------------------------------------------------------- audio

    def build_audio(self) -> None:
        rng = self._rng("audio")
        ents = {dt.kind: dt.entities for dt in content.DOC_TYPES}
        voices = ("en-gb+m3", "en-us+f3", "en-gb-scotland+m2", "en-us+m4", "en-gb-x-rp+f2", "en-029+m1", "en-us+f4", "en-gb+f1")
        islands = list(content.ISLANDS)
        rng.shuffle(islands)
        isl = iter(islands * 3)
        towns = [t for pair in content.ISLANDS.values() for t in pair]
        rng.shuffle(towns)
        twn = iter(towns * 3)
        teams = list(content.TEAMS)
        rng.shuffle(teams)

        def wd() -> str:
            return rng.choice(content.WEEKDAYS)

        items: list[tuple[str, str, str, str, Value]] = []  # slug, display text, spoken text, question, answer

        def add(slug: str, text: str, question: str, val: Value, spoken: str | None = None) -> None:
            items.append((slug, text, spoken if spoken is not None else text, question, val))

        ferry0, ferry2 = ents["ferry"][0].name, ents["ferry"][2].name
        t, g = next(twn), rng.randint(11, 29)
        add(
            "ferry_gate",
            f"Attention please. The evening sailing of the {ferry0} to {t} will leave from gate {g}.",
            f"From which gate does the evening sailing of the {ferry0} to {t} leave?",
            v_int(g),
        )
        t, i_, n = next(twn), next(isl), rng.randint(15, 55)
        add(
            "ferry_delay",
            f"We regret that the morning boat from {t} to {i_} is delayed by {n} minutes because of fog.",
            f"By how many minutes is the morning boat from {t} to {i_} delayed?",
            v_int(n, "minutes"),
        )
        i_, d = next(isl), wd()
        add(
            "ferry_cancel",
            f"Due to high winds, all sailings to {i_} are cancelled until {d} morning.",
            f"Until which day are all sailings to {i_} cancelled?",
            v_weekday(d),
        )
        t = next(twn)
        add(
            "lost_rucksack",
            f"A rucksack was found on the {ferry2}. Please collect it from the harbour office in {t}.",
            f"Where can the rucksack found on the {ferry2} be collected?",
            v_word(t),
        )
        t, n = next(twn), rng.randint(4, 12)
        add(
            "harbour_speed",
            f"The harbour master reminds all skippers that the speed limit inside {t} harbour is {n} knots.",
            f"What is the speed limit inside {t} harbour?",
            v_int(n, "knots"),
        )
        p, d, h, hall = self.names.person(), wd(), rng.randint(2, 5), rng.randint(2, 9)
        add(
            "corvid9_demo",
            f"Hi, it's {p} from {ents['company'][0].name}. The Corvid-9 demo has moved to {d} at {h} p.m. in hall {hall}.",
            "When is the Corvid-9 demo now taking place?",
            v_weekday(d),
        )
        p, i_, h = self.names.person(), next(isl), rng.randint(6, 8)
        add(
            "choir",
            f"Hello, this is {p}. Choir rehearsal on {i_} now starts at {h}:30 in the evening, in the old chapel.",
            f"What time does choir rehearsal on {i_} start now?",
            Value(
                f"{h}:30 pm",
                (f"{h}:30 pm", f"{h + 12}:30", f"{num_words(h)} thirty"),
                (f"{h}:30",),
                rf"(?<!\d)({h}|{h + 12})[:.]30(?!\d)|{num_words(h)}[\s-]thirty|half past {num_words(h)}",
            ),
            spoken=f"Hello, this is {p}. Choir rehearsal on {i_} now starts at {num_words(h)} thirty in the evening, in the old chapel.",
        )
        p = self.names.person()
        spot = rng.choice(content.KEY_SPOTS)
        add(
            "spare_key",
            f"Hi, it's {p}. I left the spare key to the boathouse under {spot[0]}.",
            f"Where did {p} leave the spare key to the boathouse?",
            v_word(spot[0], spot[1]),
        )
        p, n, d = self.names.person(), rng.randint(12, 48), wd()
        add(
            "museum_booking",
            f"This is {p} from the {ents['museum'][0].name}. Your group booking is confirmed for {n} people on {d}.",
            f"For how many people is the group booking at the {ents['museum'][0].name} confirmed?",
            v_int(n, "people"),
        )
        p, t, d = self.names.person(), next(twn), wd()
        add(
            "parcel",
            f"Message for {p}: your parcel from {t} will arrive on the afternoon ferry on {d}.",
            f"On which day will {p}'s parcel from {t} arrive?",
            v_weekday(d),
        )
        i_, n, dr = next(isl), rng.randint(25, 90), rng.choice(content.DIRECTIONS)
        add(
            "weather_wind",
            f"Now the weather for {i_}. Tomorrow expect winds of {n} kilometres per hour from the {dr}.",
            f"What wind speed is forecast for {i_} tomorrow?",
            v_int(n, "km/h"),
        )
        i_, h = next(isl), rng.randint(8, 11)
        add(
            "weather_fog",
            f"Fog will linger over {i_} until about {h} o'clock in the morning.",
            f"Until what time will fog linger over {i_}?",
            Value(f"{h} a.m.", (f"{h} a.m.", f"{h}:00", f"{h} o'clock"), (str(h),), rf"(?<!\d){h}(?!\d)|{num_words(h)}"),
        )
        t, n = next(twn), rng.randint(7, 19)
        add(
            "sea_temperature",
            f"The sea temperature off {t} this morning is {n} degrees.",
            f"What is the sea temperature off {t} this morning?",
            v_int(n, "degrees"),
        )
        i_, n = next(isl), rng.randint(2, 9)
        add(
            "frost",
            f"Tonight on {i_} temperatures will fall to minus {n} degrees, so protect your plants.",
            f"How cold will it get tonight on {i_}?",
            Value(f"minus {n} degrees", (f"minus {n} degrees", f"-{n}"), (str(n),), rf"(?<!\d){n}(?!\d)"),
        )
        bell, r, y, kg = rng.choice(content.BELLS), rng.randint(2, 9), rng.randint(1620, 1790), rng.randint(300, 1500)
        add(
            "bell",
            f"Room {r}. This bell, known as Old {bell}, was cast in {y} and weighs {kg} kilograms.",
            f"How much does the bell known as Old {bell} weigh?",
            v_int(kg, "kg"),
            spoken=f"Room {r}. This bell, known as Old {bell}, was cast in {year_words(y)} and weighs {kg} kilograms.",
        )
        p, n = self.names.person(), rng.randint(3, 19)
        add(
            "ship_model",
            f"The ship model in front of you took {p} {n} years to build, using only driftwood.",
            f"How many years did it take {p} to build the driftwood ship model?",
            v_int(n, "years"),
        )
        y, t = rng.randint(1801, 1899), next(twn)
        add(
            "tapestry",
            f"This tapestry shows the great storm of {y}, which flooded the harbour at {t}.",
            f"In which year did the great storm that flooded the harbour at {t} happen?",
            v_year(y),
            spoken=f"This tapestry shows the great storm of {year_words(y)}, which flooded the harbour at {t}.",
        )
        t, p = next(twn), self.names.person()
        add(
            "harbour_map",
            f"The old harbour map of {t} was drawn by {p}, the archipelago's first surveyor.",
            f"Who drew the old harbour map of {t}?",
            v_person(p),
        )
        t, n = next(twn), rng.randint(40, 160)
        add(
            "tower_clock",
            f"This tower clock from {t} has kept time without stopping for {n} years.",
            f"For how many years has the tower clock from {t} kept time without stopping?",
            v_int(n, "years"),
        )
        a, b = rng.randint(2, 5), rng.randint(0, 1)
        t = next(twn)
        add(
            "match_score",
            f"Final score at the {t} stadium: the {teams[0]} {a}, the {teams[1]} {b}.",
            f"What was the final score between the {teams[0]} and the {teams[1]}?",
            Value(f"{a}-{b}", (f"{a}-{b}", f"{a} to {b}"), (str(a), str(b)), rf"(?<!\d){a}\D{{1,30}}{b}(?!\d)"),
        )
        p, i_, h, m = self.names.person(), next(isl), rng.randint(2, 6), rng.randint(11, 58)
        add(
            "island_race",
            f"{p} won the round-the-island race on {i_} in {h} hours and {m} minutes.",
            f"How long did it take {p} to win the round-the-island race on {i_}?",
            Value(f"{h} hours and {m} minutes", (f"{h} hours and {m} minutes", f"{h}:{m:02d}"), (str(h), str(m)), rf"(?<!\d){h}\D{{1,30}}{m}(?!\d)"),
        )
        p, t = self.names.person(), next(twn)
        add("goalkeeper", f"The {teams[2]} have signed the goalkeeper {p} from {t}.", f"Which goalkeeper have the {teams[2]} signed?", v_person(p))
        dish, ing = rng.choice(content.DISHES_AUDIO)
        n = rng.randint(12, 45)
        add(
            "recipe_simmer",
            f"For a proper {dish}, simmer the {ing} for exactly {n} minutes.",
            f"For how many minutes should the {ing} be simmered for a proper {dish}?",
            v_int(n, "minutes"),
        )
        cake, n = rng.choice(content.CAKES), rng.randint(2, 6)
        add(
            "recipe_honey",
            f"Grandmother's tip: stir {n} spoonfuls of heather honey into the {cake} cake batter.",
            f"How many spoonfuls of heather honey go into the {cake} cake batter?",
            v_int(n, "spoonfuls"),
        )
        bread, n = rng.choice(content.BREADS), rng.choice(range(160, 240, 10))
        add("recipe_bake", f"Bake the {bread} at {n} degrees for forty minutes.", f"At what temperature should the {bread} be baked?", v_int(n, "degrees"))
        sp, i_, mo = ents["species"][2].name, next(isl), rng.choice(content.MONTHS)
        add("plover_return", f"Every {mo}, the {sp} returns to {i_} to breed.", f"In which month does the {sp} return to {i_}?", v_month(mo))
        p, obs, n = self.names.person(), ents["observatory"][3].name, rng.randint(120, 480)
        add(
            "meteors",
            f"Professor {p} reports that the {obs} recorded {n} meteors in a single night.",
            f"How many meteors did the {obs} record in a single night?",
            v_int(n, "meteors"),
        )
        rw, n = ents["railway"][1].name, rng.randint(900, 4800)
        add(
            "railway_record",
            f"On its busiest day ever, the {rw} carried {n} passengers.",
            f"How many passengers did the {rw} carry on its busiest day ever?",
            v_int(n, "passengers"),
        )
        for i, (slug, text, spoken, question, val) in enumerate(items, start=1):
            path = f"corpus/audio/audio_{i:02d}_{slug}.wav"
            clip = Audio(path=path, text=text, spoken=spoken, voice=voices[(i - 1) % len(voices)], speed=150 + 5 * ((i * 7) % 4))
            clip.facts.append(self._new_fact(modality="audio", source=path, statement=text, value=val, question=question, key=f"audio:{slug}", entity=slug))
            self.world.audios.append(clip)

    # ----------------------------------------------------------------- video

    def build_video(self) -> None:
        rng = self._rng("video")
        ents = {dt.kind: dt.entities for dt in content.DOC_TYPES}
        d, mo, t = rng.randint(2, 28), rng.choice(content.MONTHS), rng.choice(("Fernmouth", "Pellby", "Dunlin"))
        p1, p2 = self.names.person(), self.names.person()
        h, m = rng.randint(1, 4), rng.randint(10, 55)
        i3, n3 = rng.choice(("Corvane", "Tamsin", "Halvric")), rng.choice((30, 40, 50))
        x4 = rng.randint(14, 39) + 0.5
        st, mo5 = rng.choice(content.STATIONS), rng.choice([x for x in content.MONTHS if x != mo])
        n7 = rng.randint(40, 240) * 100
        x8 = round(rng.uniform(4.2, 13.8), 1)
        t9, h9 = rng.choice(("Brightwick", "Merrow End", "Ashcombe")), rng.choice((19, 20, 21))
        dog, lm = rng.choice(content.DOG_NAMES), rng.choice(content.LANDMARKS)
        lh = ents["lighthouse"][3].name
        defs: list[tuple[str, str, list[str], str, str, Value]] = [
            (
                "pier",
                "HARBOUR BULLETIN",
                [f"NEW PIER AT {t.upper()}", f"OPENS {d} {mo.upper()}"],
                f"The new pier at {t} opens on {d} {mo}.",
                f"When does the new pier at {t} open?",
                Value(f"{d} {mo}", (f"{d} {mo}", f"{mo} {d}"), (str(d), mo[:3]), rf"(?<!\d){d}(?!\d)\D{{0,6}}{mo[:3]}|{mo[:3]}\D{{0,8}}(?<!\d){d}(?!\d)"),
            ),
            (
                "kite_results",
                "DUSKWATER KITE WEEK",
                [f"WINNER: {p1.upper()}", f"AIRBORNE {h} H {m} MIN"],
                f"{p1} won Duskwater Kite Week, keeping a kite airborne for {h} h {m} min.",
                "Who won Duskwater Kite Week according to the results bulletin?",
                v_person(p1),
            ),
            (
                "speed_limit",
                "ISLAND COUNCIL NOTICE",
                [f"{i3.upper()} COAST ROAD", f"NEW LIMIT {n3} KM/H"],
                f"The new speed limit on the {i3} coast road is {n3} km/h.",
                f"What is the new speed limit on the {i3} coast road?",
                v_int(n3, "km/h"),
            ),
            (
                "lobster_price",
                "MARKET NEWS",
                ["LOBSTER PRICE TODAY", f"{x4:.2f} CROWNS/KG"],
                f"Today's lobster price is {x4:.2f} Saltmere crowns per kg.",
                "What is today's lobster price according to the market news bulletin?",
                v_money(x4),
            ),
            (
                "station_closed",
                "RAILWAY NOTICE",
                [f"{st.upper()} STATION", f"CLOSED UNTIL {mo5.upper()}"],
                f"{st} station is closed until {mo5}.",
                f"Until when is {st} station closed?",
                v_month(mo5),
            ),
            (
                "new_rector",
                "COLLEGE NEWS",
                ["QUILLMERE COLLEGE", f"NEW RECTOR: {p2.upper()}"],
                f"The new rector of Quillmere College is {p2}.",
                "Who is the new rector of Quillmere College?",
                v_person(p2),
            ),
            (
                "lighthouse_appeal",
                "LIGHTHOUSE APPEAL",
                [lh.upper(), f"{n7:,} CROWNS RAISED"],
                f"The repair appeal for the {lh} has raised {n7:,} Saltmere crowns.",
                f"How much money has the repair appeal for the {lh} raised?",
                v_int(n7, "Saltmere crowns"),
            ),
            (
                "storm_wave",
                "STORM REPORT",
                ["HIGHEST WAVE AT GULLSTEAD", f"{x8:.1f} M"],
                f"The highest wave recorded at Gullstead during the storm was {x8:.1f} m.",
                "How high was the highest wave recorded at Gullstead during the storm?",
                v_dec(x8, "m"),
            ),
            (
                "library_hours",
                "LIBRARY NEWS",
                [f"{t9.upper()} LIBRARY", f"NOW OPEN UNTIL {h9}:00"],
                f"The library in {t9} is now open until {h9}:00.",
                f"Until what time is the library in {t9} now open?",
                Value(f"{h9}:00", (f"{h9}:00", f"{h9 - 12} pm"), (f"{h9}:00",), rf"(?<!\d)({h9}[:.]00|{h9 - 12}(?::00)?\s?(pm|p\.m\.))"),
            ),
            (
                "lost_dog",
                "LOST AND FOUND",
                [f"DOG '{dog.upper()}' FOUND", f"NEAR {lm[0].upper()}"],
                f"The lost dog {dog} was found near {lm[0]}.",
                f"Where was the lost dog {dog} found?",
                v_word(lm[0], lm[1]),
            ),
        ]
        n = 0
        for slug, header, lines, stmt, question, val in defs:
            n += 1
            path = f"corpus/video/video_{n:02d}_{slug}.mp4"
            v = Video(path=path, kind="bulletin", header=header, lines=lines, duration=7.0, seed=rng.randrange(1 << 30))
            v.facts.append(
                self._new_fact(
                    modality="video",
                    source=path,
                    statement=stmt,
                    value=val,
                    question=question,
                    location={"t_start": 3.0, "t_end": 6.0},
                    key=f"video:{slug}",
                    entity=header.title(),
                )
            )
            self.world.videos.append(v)
        motions = [("circle", "red", "left_to_right"), ("circle", "red", "right_to_left"), ("square", "green", "bottom_to_top"), ("star", "yellow", "diagonal")]
        words = {
            "left_to_right": "from left to right",
            "right_to_left": "from right to left",
            "bottom_to_top": "from the bottom to the top",
            "diagonal": "diagonally from the top-left to the bottom-right corner",
        }
        group = []
        for shape, col, mot in motions:
            n += 1
            path = f"corpus/video/video_{n:02d}_motion.mp4"
            v = Video(path=path, kind="motion", shape=shape, color=col, motion=mot, duration=5.0, seed=rng.randrange(1 << 30))
            desc = f"a {col} {shape} moves {words[mot]} across a plain light background"
            direction_rx = {
                "left_to_right": r"left\W+to\W+(the\W+)?right|rightwards|to the right",
                "right_to_left": r"right\W+to\W+(the\W+)?left|leftwards|to the left",
                "bottom_to_top": r"bottom\W+to\W+(the\W+)?top|upwards?|ris(es|ing)",
                "diagonal": r"diagonal",
            }[mot]
            f = self._new_fact(
                modality="video",
                source=path,
                statement=f"A short clip in which {desc}.",
                value=Value(words[mot], (words[mot],), (shape,), direction_rx, spoken=words[mot]),
                question=f"Find the video in which {desc}.",
                location={"t_start": 0.0, "t_end": 5.0},
                key=f"motion:{col}:{shape}",
                entity=desc,
            )
            v.facts.append(f)
            group.append(f)
            self.world.videos.append(v)
        self._link_family(group)

    def build_query_audio(self) -> None:
        ents = {dt.kind: dt.entities for dt in content.DOC_TYPES}
        qs = [
            ("quellmar_employees", f"How many people does {ents['company'][1].name} employ?", "en-gb+f4"),
            ("saffren_month", f"In which month is the {ents['festival'][1].name} held?", "en-us+m2"),
        ]
        for slug, text, voice in qs:
            self.world.query_audios.append(Audio(path=f"{QUERY_MEDIA_DIR}/q_{slug}.wav", text=text, spoken=text, voice=voice, speed=150))

    # ----------------------------------------------------------------- queries and dialogues

    def _doc_fact(self, kind: str, idx: int, key: str) -> Fact:
        dt = next(d for d in content.DOC_TYPES if d.kind == kind)
        doc = next(d for d in self.world.docs if d.entity is dt.entities[idx] and not d.path.endswith("_2019_edition.md"))
        return next(f for f in doc.facts if f.key == key)

    def build_queries(self) -> None:
        rng = self._rng("queries")
        w = self.world
        qs: list[tuple[str, str, str, str | None, list[Fact], str]] = []  # type, lang, text, attachment, facts, keyword override
        reserved: set[str] = set()

        def take(f: Fact) -> Fact:
            reserved.add(f.id)
            return f

        base_docs = [d for d in w.docs if not d.path.endswith("_2019_edition.md")]
        # near-duplicate (6): pairs 1-2 ask both editions, pair 3 the archived one, pair 4 the revised one
        nd_queries: list[Fact] = []
        for i, (kind, idx, key) in enumerate(NEAR_DUPLICATES):
            base = take(self._doc_fact(kind, idx, key))
            variant = take(w.fact(base.distractor_ids[0]))
            nd_queries += [base, variant] if i < 2 else ([variant] if i == 2 else [base])
        # cross-modal media -> text (attachments)
        ims = w.query_slides
        cross: list[tuple[str, str | None, Fact, str]] = [
            (
                "When was the main mirror installed at the observatory named on this sign?",
                ims[0].path,
                take(self._doc_fact("observatory", 0, "mirror_year")),
                "",
            ),
            ("Which company makes the product shown in this picture?", ims[1].path, take(self._doc_fact("company", 0, "flagship")), "Brindle"),
            ("How many steps lead up to the lamp room of the lighthouse in this picture?", ims[2].path, take(self._doc_fact("lighthouse", 0, "steps")), ""),
            ("Please answer the question asked in this recording.", w.query_audios[0].path, take(self._doc_fact("company", 1, "employees")), ""),
            ("Please answer the question asked in this recording.", w.query_audios[1].path, take(self._doc_fact("festival", 1, "month")), ""),
        ]
        # cross-modal text -> image / audio / video
        charts = [c.facts[0] for c in w.charts]
        pick_charts = [charts[0], *rng.sample(charts[2:], 3)]
        colour = [f for s in w.slides if s.kind == "colour" for f in s.facts]
        spec = [f for s in w.slides if s.kind in ("spec", "sign") for f in s.facts]
        photos = [f for p in w.photos for f in p.facts]
        audio = [f for a in w.audios for f in a.facts]
        bulletins = [f for v in w.videos if v.kind == "bulletin" for f in v.facts]
        motion = [f for v in w.videos if v.kind == "motion" for f in v.facts]
        to_media = [
            *pick_charts,
            *rng.sample(colour, 2),
            rng.choice(spec[1:]),
            photos[0],
            rng.choice(photos[1:]),
            *rng.sample(audio, 6),
            *rng.sample(bulletins, 4),
            motion[0],
        ]
        # multilingual (5 es + 4 ja) on English text facts with numeric/name answers
        ml_pool = [
            f
            for d in base_docs
            for f in d.facts
            if f.id not in reserved and f.es and f.ja and f.template is not None and f.template.value.kind in ("int", "dec", "year", "person", "money")
        ]
        rng.shuffle(ml_pool)
        ml: list[Fact] = []
        for f in ml_pool:
            if len(ml) < 9 and all(g.entity != f.entity for g in ml):
                ml.append(take(f))
        # single-modal text -> text (30): at most one per document
        docs = list(base_docs)
        rng.shuffle(docs)
        single: list[Fact] = []
        for d in docs:
            cands = [f for f in d.facts if f.id not in reserved]
            if cands and len(single) < 30:
                single.append(take(rng.choice(cands)))
        # negatives: unused attributes of existing entities (7) + absent entities (3)
        negs: list[tuple[str, str]] = []  # (text, entity)
        neg_docs = list(base_docs)
        rng.shuffle(neg_docs)
        for d in neg_docs:
            dt = next(x for x in content.DOC_TYPES if x.kind == d.kind)
            have = {f.key for f in d.facts}
            unused = [t for t in dt.templates if t.key not in have and t.value.kind in ("int", "dec", "year", "person", "money")]
            if unused and len(negs) < 7:
                t = rng.choice(unused)
                negs.append((t.question.format(**entity_slots(dt, d.entity)), d.entity.name))
        for kind, name in content.ABSENT_ENTITIES:
            dt = next(x for x in content.DOC_TYPES if x.kind == kind)
            t = rng.choice([t for t in dt.templates if t.value.kind in ("int", "year", "person")])
            negs.append((t.question.format(**entity_slots(dt, content.Entity(name, dt.entities[0].island))), name))

        for f in single:
            qs.append(("single_modal", "en", f.question, None, [f], ""))
        for text, att, f, kw in cross:
            qs.append(("cross_modal", "en", text, att, [f], kw))
        for f in to_media:
            qs.append(("cross_modal", "en", f.question, None, [f], ""))
        for text, _ in negs:
            qs.append(("negative", "en", text, None, [], ""))
        for f in nd_queries:
            qs.append(("near_duplicate", "en", f.question, None, [f], ""))
        for i, f in enumerate(ml):
            lang = "es" if i < 5 else "ja"
            qs.append(("multilingual", lang, f.es if lang == "es" else f.ja, None, [f], ""))
        for i, (typ, lang, text, att, facts, kw_override) in enumerate(qs, start=1):
            if kw_override:
                keywords, regex = [kw_override], word_rx(kw_override)
            elif facts:
                keywords, regex = list(facts[0].value.keywords), facts[0].value.regex
            else:
                keywords, regex = [], ""
            w.queries.append(
                Query(
                    id=f"Q{i:03d}",
                    type=typ,
                    lang=lang,
                    text=text,
                    attachment=att,
                    expected_sources=sorted({f.source for f in facts}),
                    answer_keywords=keywords,
                    answer_regex=regex,
                    must_refuse=typ == "negative",
                    fact_ids=[f.id for f in facts],
                )
            )

    def build_dialogues(self) -> None:
        w = self.world
        out: list[dict[str, Any]] = []
        for dt in content.DOC_TYPES:
            idx = DIALOGUE_HOSTS[dt.kind]
            turns = []
            for j, key in enumerate(dt.dialogue):
                f = self._doc_fact(dt.kind, idx, key)
                text = f.question if j == 0 else (f.followup or f.question)
                turns.append(self._turn(text, f))
            out.append({"id": f"D{len(out) + 1:02d}", "turns": turns})
        flagship = self._doc_fact("company", 0, "flagship")
        flight = next(f for s in w.slides for f in s.facts if f.key == "spec:corvid9")
        demo = next(f for a in w.audios for f in a.facts if f.key == "audio:corvid9_demo")
        out.append(
            {
                "id": f"D{len(out) + 1:02d}",
                "turns": [
                    self._turn(flagship.question, flagship),
                    self._turn("What is its maximum flight time?", flight),
                    self._turn("And when is its demo now taking place?", demo),
                ],
            }
        )
        w.dialogues = out

    @staticmethod
    def _turn(text: str, f: Fact) -> dict[str, Any]:
        return {"user": text, "expected_sources": [f.source], "answer_keywords": list(f.value.keywords), "answer_regex": f.value.regex, "fact_ids": [f.id]}

    # ----------------------------------------------------------------- probe + sample

    def build_probe_and_sample(self) -> None:
        w = self.world

        def doc(kind: str, idx: int, variant: bool = False) -> Doc:
            dt = next(d for d in content.DOC_TYPES if d.kind == kind)
            return next(d for d in w.docs if d.entity is dt.entities[idx] and d.path.endswith("_2019_edition.md") == variant)

        def by_key(items: list[Any], key: str) -> Any:
            return next(it for it in items if any(f.key == key for f in it.facts))

        probe_docs = [
            doc("lighthouse", 0),
            doc("observatory", 0),
            doc("company", 0),
            doc("festival", 0),
            doc("ferry", 0),
            doc("museum", 2),
            doc("crop", 0),
            doc("railway", 0),
        ]
        probe_imgs = [w.charts[0], w.charts[5], w.slides[0], by_key(w.slides, "spec:corvid9"), w.photos[0]]
        probe_aud = [
            by_key(w.audios, "audio:corvid9_demo"),
            by_key(w.audios, "audio:ferry_gate"),
            by_key(w.audios, "audio:weather_wind"),
            by_key(w.audios, "audio:bell"),
        ]
        probe_vid = [w.videos[0], w.videos[1], w.videos[10]]
        items: list[dict[str, Any]] = []
        queries: list[dict[str, Any]] = []
        groups: list[tuple[str, list[Any], int]] = [("text", probe_docs, 8), ("image", probe_imgs, 4), ("audio", probe_aud, 2), ("video", probe_vid, 2)]
        for modality, objs, n_q in groups:
            for j, obj in enumerate(objs):
                pid = f"P{len(items) + 1:02d}"
                items.append({"id": pid, "path": obj.path, "modality": modality})
                if j < n_q:
                    f = obj.facts[0]
                    queries.append({"text": f.question, "expected": [pid]})
        w.probe = {"items": items, "queries": queries}

        sample_docs = [doc(dt.kind, DIALOGUE_HOSTS[dt.kind]) for dt in content.DOC_TYPES] + [
            doc("company", 0),
            doc("lighthouse", 1),
            doc("lighthouse", 1, True),
        ]
        sample_imgs = [w.charts[0], w.charts[1], w.slides[0], w.slides[1], by_key(w.slides, "spec:corvid9"), w.photos[0], w.photos[1], w.photos[2]]
        sample_aud = [*probe_aud, by_key(w.audios, "audio:match_score")]
        w.sample_subset = [o.path for o in (*sample_docs, *sample_imgs, *sample_aud, *probe_vid)]


def build_world(seed: int) -> World:
    b = Builder(seed)
    b.build_text()
    b.build_images()
    b.build_audio()
    b.build_video()
    b.build_query_audio()
    b.build_queries()
    b.build_dialogues()
    b.build_probe_and_sample()
    return b.world
