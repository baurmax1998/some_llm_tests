import json
import streamlit as st
from json_schema_playground.model_backend import DEFAULT_MODEL, MLXModelBackend
from json_schema_playground.schema_engine import OutlinesSchemaConstraint, validate_supported_schema
from json_schema_playground.generation import generate

st.set_page_config(page_title="JSON-Schema Decoding", layout="wide")
st.title("Lokales JSON-Schema-Decoding-Playground")
DEFAULT_SCHEMA = json.dumps({
    "type": "object", "properties": {
        "sentiment": {"type": "string", "enum": ["positiv", "neutral", "negativ"]},
        "confidence": {"type": "integer"}
    }, "required": ["sentiment", "confidence"], "additionalProperties": False
}, ensure_ascii=False, indent=2)

prompt = st.text_area("Prompt", "Bewerte: Das Produkt gefällt mir sehr.")
schema_text = st.text_area("JSON Schema", DEFAULT_SCHEMA, height=280)
model_id = st.text_input("Modell-ID", DEFAULT_MODEL)
a, b, c, d = st.columns(4)
temperature = a.number_input("Temperature", min_value=0.0, value=0.7, step=0.1)
top_p = b.number_input("Top-p", min_value=0.01, max_value=1.0, value=0.9, step=0.05)
seed = c.number_input("Seed", value=0, step=1)
limit = d.number_input("Tokenlimit", min_value=1, value=256, step=1)
start = st.button("Start", type="primary")
cancel = st.button("Abbrechen", disabled=not start)
output_box = st.empty()
status_box = st.empty()

if start:
    try:
        schema = json.loads(schema_text)
        validate_supported_schema(schema)
        status_box.info("Modell wird geladen …")
        backend = MLXModelBackend(model_id)
        constraint = OutlinesSchemaConstraint(schema, backend.tokenizer, backend.vocab_size)
        def update(t):
            output_box.code(t.text, language="json")
            status_box.info(f"läuft – Schritt {len(t.steps)}")
        result = generate(backend, constraint, schema, prompt, float(temperature),
                          float(top_p), int(seed), int(limit), update,
                          should_cancel=lambda: cancel)
        st.session_state["last_trace"] = result
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

trace = st.session_state.get("last_trace")
if trace:
    output_box.code(trace.text, language="json")
    {"abgeschlossen": status_box.success, "unvollständig": status_box.warning,
     "fehlgeschlagen": status_box.error, "abgebrochen": status_box.warning}.get(
        trace.status, status_box.info)(f"Status: {trace.status}" + (f" – {trace.error}" if trace.error else ""))
    if trace.steps:
        index = st.selectbox("Generierungsschritt", range(len(trace.steps)),
                             format_func=lambda i: f"{i}: Token {trace.steps[i].selected_token_id}")
        step = trace.steps[index]
        st.metric("allowed_mass", f"{step.allowed_mass:.6f}")
        st.caption(f"Erlaubt: {step.allowed_count} · Modell: {step.model_seconds:.4f}s · Maske: {step.mask_seconds:.4f}s")
        rows = {}
        for label, entries in (("ursprünglich", step.raw_top20),
                               ("nach Grammatik", step.constrained_top20),
                               ("Sampling", step.sampling_top20)):
            for x in entries:
                row = rows.setdefault(x.token_id, {"Token-ID": x.token_id, "Token": x.token,
                                                   "gesperrt": not x.allowed})
                row[label] = x.probability
        st.dataframe(list(rows.values()), use_container_width=True, hide_index=True)
        mode = "deterministisch (Greedy)" if step.deterministic else "Zufallsauswahl"
        st.write(f"Auswahl: **{mode}**, raw p={step.selected_raw_probability:.6g}, final p={step.selected_final_probability:.6g}")
