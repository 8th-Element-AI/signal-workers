"""Safety lens — PII detection via Presidio + content safety classification.

Processes `model_call` spans. For each batch of spans we:

  1. Collect every input + output text (skipping empties).
  2. Run ONE batched Presidio call across them (`analyze_batch(texts, batch_size=4)`)
     — concurrent detection, identical texts deduped automatically.
  3. Run SafetyObservabilityClassifier on each unique input text not already cached.
  4. Stash the per-text results in in-memory content-hash caches so a later
     batch hitting the same prompt (which is the common case) doesn't re-analyze.
  5. Per span, build_context looks up the cached results for *this span's*
     input + output, summarizes into pii fields + content safety scores.

No `compute()` override — per-row entity-type detail is attached via the
generic `meta_fn` hook on `MetricSpec`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict

from ..base import parse_meta
from ..spec import MetricSpec, SpecWorker
from ..patterns import ctx_value
from ..predicates import llm_call

log = logging.getLogger("signal.worker.safety")

LENS = "safety"

def _spec(metric, applies, pattern, inputs, unit, window="1h",
          threshold=False, per_span=True, meta_fn=None):
    return MetricSpec(
        metric=metric, lens=LENS, applies=applies, pattern=pattern,
        inputs=inputs, unit=unit, window=window, threshold=threshold,
        per_span=per_span, meta_fn=meta_fn,
    )


# ---- meta_fn: attach entity types + which side the PII was on to each row ----
def _pii_meta(span, ctx):
    if not ctx.get("pii_types"):
        return None
    meta = {"types": ctx["pii_types"]}
    loc = ctx.get("pii_location")
    if loc:
        meta["location"] = loc                              # "input" | "output" | "both"
    return meta


def _cs_meta(span, ctx):
    if not ctx.get("cs_labels"):
        return None
    meta = {"labels": ctx["cs_labels"]}
    if ctx.get("cs_triggered"):
        meta["triggered_models"] = ctx["cs_triggered"]
    if ctx.get("cs_skipped"):
        meta["skipped_models"] = ctx["cs_skipped"]
    if ctx.get("cs_routing"):
        meta["routing"] = ctx["cs_routing"]
    return meta


SPECS = [
    # ---- PII ----
    _spec("pii_count",            llm_call, ctx_value("pii_count"),
          ["metadata.input + metadata.output -> presidio.entity_count"],
          unit="count", window="1h", threshold=True, meta_fn=_pii_meta),
    _spec("pii_detected",         llm_call, ctx_value("pii_detected"),
          ["pii_count > 0"], unit="ratio", window="1h", threshold=True),
    _spec("pii_distinct_types",   llm_call, ctx_value("pii_type_count"),
          ["distinct entity types detected across input + output"],
          unit="count", window="1h"),

    # ---- Content Safety ----
    _spec("content_safety_flagged",           llm_call, ctx_value("cs_flagged"),
          ["metadata.input -> content_safety labels"],
          unit="ratio", window="1h", threshold=True, meta_fn=_cs_meta),
    _spec("content_safety_prompt_injection",  llm_call, ctx_value("cs_prompt_injection"),
          ["metadata.input -> content_safety prompt_injection score"],
          unit="score", window="1h", threshold=True),
    _spec("content_safety_harmful_content",   llm_call, ctx_value("cs_harmful_content"),
          ["metadata.input -> content_safety harmful_content score"],
          unit="score", window="1h", threshold=True),
    _spec("content_safety_sexual",            llm_call, ctx_value("cs_sexual"),
          ["metadata.input -> content_safety sexual score"],
          unit="score", window="1h"),
]


class SafetyWorker(SpecWorker):
    """Safety lens worker — runs Presidio across spans in batches.

    The PII module is imported as a normal package (`deidentifier`); install it
    in editable mode in the same environment (`pip install -e ./PII`). The
    `sys.path` hack from the earlier version is gone.

    Per-batch flow:
      process_batch(spans)
        -> collect (text, hash) pairs not already cached
        -> pii_engine.analyze_batch(unique_texts, batch_size=4)   # concurrent
        -> populate the LRU cache by hash
        -> for each span: super().compute(span) -> build_context reads cache
    """
    lens = LENS
    specs = SPECS
    span_types = ("model_call",)

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cache_max = cfg.signal_pii_cache_max
        self.batch_concurrency = cfg.signal_pii_batch
        self.ner_model = cfg.signal_pii_ner_model
        self.cs_config = cfg.signal_content_safety_config or None
        self.cs_cache_max = cfg.signal_content_safety_cache_max

        self._pii_engine = None
        self._cs_classifier = None
        self._cache: "OrderedDict[str, object]" = OrderedDict()       # hash -> PII AnalysisResult
        self._cs_cache: "OrderedDict[str, object]" = OrderedDict()    # hash -> content safety result
        self._cache_lock = threading.Lock()
        self._cs_cache_lock = threading.Lock()

    # ---- lazy PII engine (mirrors PricingCache property pattern) ----
    @property
    def pii_engine(self):
        if self._pii_engine is None:
            from deidentifier import PresidioEngine
            log.info("[safety] loading PresidioEngine (ner_model=%s)", self.ner_model)
            self._pii_engine = PresidioEngine.get_instance(ner_model=self.ner_model)
        return self._pii_engine

    # ---- lazy content safety classifier ----
    @property
    def cs_classifier(self):
        if self._cs_classifier is None:
            from safety_observability import SafetyObservabilityClassifier
            log.info("[safety] loading SafetyObservabilityClassifier (config=%s)", self.cs_config)
            self._cs_classifier = SafetyObservabilityClassifier(self.cs_config)
        return self._cs_classifier

    # ---- LRU helpers ----
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _cache_get(self, h: str):
        with self._cache_lock:
            if h in self._cache:
                self._cache.move_to_end(h)            
                return self._cache[h]
            return None

    def _cache_put(self, h: str, result):
        with self._cache_lock:
            self._cache[h] = result
            self._cache.move_to_end(h)
            while len(self._cache) > self.cache_max:
                self._cache.popitem(last=False)       

    def _cs_cache_get(self, h: str):
        with self._cs_cache_lock:
            if h in self._cs_cache:
                self._cs_cache.move_to_end(h)
                return self._cs_cache[h]
            return None

    def _cs_cache_put(self, h: str, result):
        with self._cs_cache_lock:
            self._cs_cache[h] = result
            self._cs_cache.move_to_end(h)
            while len(self._cs_cache) > self.cs_cache_max:
                self._cs_cache.popitem(last=False)

    # ---- batch hook: pre-fill the cache for an entire batch in ONE call ----
    def process_batch(self, spans: list) -> list:
        t_batch_start = time.time()

        # 1. Collect texts that aren't already cached, deduped across the whole batch.
        to_analyze: dict[str, str] = {}                # hash -> text
        cs_to_analyze: dict[str, str] = {}             # hash -> input text (content safety)
        model_call_count = 0
        for span in spans:
            if span.get("span_type") not in self.span_types:
                continue
            model_call_count += 1
            md = parse_meta(span.get("metadata"))
            # PII: check both input and output
            for text in (md.get("input"), md.get("output")):
                if not text:
                    continue
                h = self._hash(text)
                if self._cache_get(h) is None and h not in to_analyze:
                    to_analyze[h] = text
            # Content safety: input only
            input_text = md.get("input")
            if input_text:
                h = self._hash(input_text)
                if self._cs_cache_get(h) is None and h not in cs_to_analyze:
                    cs_to_analyze[h] = input_text

        pii_cache_hits = model_call_count - len(to_analyze)
        cs_cache_hits = model_call_count - len(cs_to_analyze)

        log.info("[safety] batch=%d model_call_spans=%d | pii: %d to_analyze %d cache_hits | cs: %d to_analyze %d cache_hits",
                 len(spans), model_call_count,
                 len(to_analyze), pii_cache_hits,
                 len(cs_to_analyze), cs_cache_hits)

        # 2. One concurrent PII call covers everything we don't already have.
        t_pii_start = time.time()
        if to_analyze:
            hashes = list(to_analyze.keys())
            texts = [to_analyze[h] for h in hashes]
            log.info("[safety] pii analyze_batch: %d unique texts (concurrency=%d)",
                     len(texts), self.batch_concurrency)
            results = self.pii_engine.analyze_batch(texts, batch_size=self.batch_concurrency)
            for h, r in zip(hashes, results):
                self._cache_put(h, r)
        t_pii_ms = round((time.time() - t_pii_start) * 1000, 1)

        # 3. Content safety: classify unique input texts sequentially.
        # PyTorch's DeBERTa moderation model uses global meta/fake tensor state that is
        # not thread-safe — parallel classification causes RuntimeError even with locks.
        # FastText handles 99%+ of inputs in ~1ms so serial is fast enough.
        t_cs_start = time.time()
        if cs_to_analyze:
            log.info("[safety] content_safety classify: %d unique input texts",
                     len(cs_to_analyze))
            for h, text in cs_to_analyze.items():
                try:
                    result = self.cs_classifier.classify(text)
                    self._cs_cache_put(h, result)
                except Exception:
                    log.exception("[safety] content_safety classify failed for hash=%s", h)
        t_cs_ms = round((time.time() - t_cs_start) * 1000, 1)

        # 4. Now defer to the standard per-span engine; build_context reads the cache.
        t_emit_start = time.time()
        rows = super().process_batch(spans)
        t_emit_ms = round((time.time() - t_emit_start) * 1000, 1)

        t_total_ms = round((time.time() - t_batch_start) * 1000, 1)
        log.info(
            "[safety] latency report | total=%.1fms | pii=%.1fms | content_safety=%.1fms | emit=%.1fms | rows=%d",
            t_total_ms, t_pii_ms, t_cs_ms, t_emit_ms, len(rows)
        )
        return rows

    # ---- build a per-span context from the cached input/output results ----
    def build_context(self, span: dict) -> dict:
        md = parse_meta(span.get("metadata"))
        input_text = md.get("input") or ""
        output_text = md.get("output") or ""

        in_res = self._cache_get(self._hash(input_text)) if input_text else None
        out_res = self._cache_get(self._hash(output_text)) if output_text else None

        in_count = in_res.entity_count if in_res else 0
        out_count = out_res.entity_count if out_res else 0
        total = in_count + out_count

        # Distinct types across input + output
        types: dict[str, int] = {}
        if in_res:
            for t, c in in_res.entities.items():
                types[t] = types.get(t, 0) + c
        if out_res:
            for t, c in out_res.entities.items():
                types[t] = types.get(t, 0) + c

        # Which side the PII was on
        if in_count and out_count:
            location = "both"
        elif in_count:
            location = "input"
        elif out_count:
            location = "output"
        else:
            location = None

        # Content safety — input only
        cs_res = self._cs_cache_get(self._hash(input_text)) if input_text else None
        cs_scores = cs_res["scores"] if cs_res else {}
        cs_labels = cs_res["labels"] if cs_res else []
        cs_routing = cs_res["routing"] if cs_res else None
        cs_triggered = cs_res["triggered_models"] if cs_res else []
        cs_skipped = cs_res["skipped_models"] if cs_res else []

        return {
            "pii_count":                  float(total),
            "pii_detected":               1.0 if total > 0 else 0.0,
            "pii_type_count":             float(len(types)),
            "pii_types":                  sorted(types.keys()),
            "pii_location":               location,
            "cs_flagged":                 1.0 if cs_labels else 0.0,
            "cs_prompt_injection":        float(cs_scores.get("prompt_injection", 0.0)),
            "cs_harmful_content":         float(cs_scores.get("harmful_content", 0.0)),
            "cs_sexual":                  float(cs_scores.get("sexual", 0.0)),
            "cs_labels":                  cs_labels,
            "cs_routing":                 cs_routing,
            "cs_triggered":               cs_triggered,
            "cs_skipped":                 cs_skipped,
        }
    