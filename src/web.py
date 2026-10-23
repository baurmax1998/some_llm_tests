"""FastAPI-Host und statischer Compiler für SIC-Webprojekte.

Die Webschicht bleibt bewusst klein: Konfiguration, Seiten und Endpunktlogik
sind SIC-Programme. Python übernimmt nur HTTP, Hot-Reload und das Schreiben
der erzeugten Browserdateien.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import threading
from typing import Any

from .lang import Evaluator, LangCalculationError, LangError, compile_program, standard_registry
from .lang.types import ListType, MapType, NullableType, Primitive, RecordType, Type, prune


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    method: str
    path: str
    source: Path


@dataclass(frozen=True)
class WebConfig:
    title: str
    host: str
    port: int
    output: Path
    pages: tuple[Path, ...]
    endpoints: tuple[EndpointConfig, ...]
    root: Path


def _inside(root: Path, raw: str, label: str) -> Path:
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} verlässt das Projektverzeichnis: {raw}") from exc
    return path


def _string(record: dict[str, Any], name: str, context: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{name} muss ein nicht leerer String sein")
    return value


def load_web_config(path: str | Path = "web.sic") -> WebConfig:
    """Wertet die SIC-Konfiguration aus und validiert ihre Web-Metadaten."""
    config_path = Path(path).resolve()
    root = config_path.parent
    source = config_path.read_text(encoding="utf-8")
    registry = standard_registry(root)
    value = Evaluator(compile_program(source, registry), registry).run().value
    if not isinstance(value, dict):
        raise ValueError("Das Goal der Web-Konfiguration muss einen Record liefern")

    title = _string(value, "title", "app")
    host = _string(value, "host", "app")
    port = value.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("app.port muss ein Int zwischen 1 und 65535 sein")
    output_raw = _string(value, "output", "app")
    output = _inside(root, output_raw, "app.output")

    raw_pages = value.get("pages")
    raw_endpoints = value.get("endpoints")
    if not isinstance(raw_pages, list) or not all(isinstance(item, str) for item in raw_pages):
        raise ValueError("app.pages muss List<String> sein")
    if not isinstance(raw_endpoints, list) or not all(isinstance(item, dict) for item in raw_endpoints):
        raise ValueError("app.endpoints muss eine Liste von Records sein")

    pages = tuple(_inside(root, item, "page") for item in raw_pages)
    endpoints: list[EndpointConfig] = []
    names: set[str] = set()
    routes: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_endpoints):
        context = f"app.endpoints[{index}]"
        name = _string(item, "name", context)
        method = _string(item, "method", context).upper()
        route = _string(item, "path", context)
        source_path = _inside(root, _string(item, "source", context), "endpoint.source")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"{context}.method wird nicht unterstützt: {method}")
        if not route.startswith("/"):
            raise ValueError(f"{context}.path muss mit / beginnen")
        if name in names:
            raise ValueError(f"Endpunktname {name!r} ist doppelt")
        if (method, route) in routes:
            raise ValueError(f"Route {method} {route} ist doppelt")
        names.add(name)
        routes.add((method, route))
        endpoints.append(EndpointConfig(name, method, route, source_path))
    return WebConfig(title, host, port, output, pages, tuple(endpoints), root)


def _coerce_http_value(value: Any, expected: Type | None) -> Any:
    """Konvertiert nur eindeutige HTTP-Strings anhand des geprüften SIC-Typs."""
    if expected is None:
        return value
    expected = prune(expected)
    if isinstance(expected, NullableType):
        return None if value is None else _coerce_http_value(value, expected.inner)
    if isinstance(expected, Primitive) and isinstance(value, str):
        if expected.name == "Int":
            try:
                return int(value)
            except ValueError:
                return value
        if expected.name == "Float":
            try:
                return float(value)
            except ValueError:
                return value
        if expected.name == "Bool" and value.lower() in {"true", "false"}:
            return value.lower() == "true"
    if isinstance(expected, RecordType) and isinstance(value, dict):
        return {
            name: _coerce_http_value(item, expected.fields.get(name))
            for name, item in value.items()
        }
    if isinstance(expected, MapType) and isinstance(value, dict):
        return {
            _coerce_http_value(key, expected.key): _coerce_http_value(item, expected.value)
            for key, item in value.items()
        }
    if isinstance(expected, ListType) and isinstance(value, list):
        return [_coerce_http_value(item, expected.element) for item in value]
    return value


class HotSicProgram:
    """Hält die letzte kompilierte Fassung und lädt Änderungen automatisch."""

    def __init__(self, source: Path, root: Path, execution_lock: threading.RLock):
        self.source = source
        self.root = root
        self.execution_lock = execution_lock
        self._load_lock = threading.RLock()
        self._stamp: int | None = None
        self._checked = None
        self._registry = None
        self._load()

    def _load(self) -> None:
        stamp = self.source.stat().st_mtime_ns
        if stamp == self._stamp and self._checked is not None:
            return
        with self._load_lock:
            stamp = self.source.stat().st_mtime_ns
            if stamp == self._stamp and self._checked is not None:
                return
            source = self.source.read_text(encoding="utf-8")
            registry = standard_registry(self.root)
            checked = compile_program(source, registry)
            self._registry = registry
            self._checked = checked
            self._stamp = stamp

    def execute(self, arguments: dict[str, Any]) -> Any:
        self._load()
        converted = {
            name: _coerce_http_value(value, self._checked.inputs.get(name))
            for name, value in arguments.items()
        }
        with self.execution_lock:
            # Evaluator-Zustand ist absichtlich pro Request frisch.
            return Evaluator(self._checked, self._registry, converted).run().value


def _browser_runtime(endpoints: tuple[EndpointConfig, ...]) -> str:
    public = {
        endpoint.name: {"method": endpoint.method, "path": endpoint.path}
        for endpoint in endpoints
    }
    encoded = json.dumps(public, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""
const definitions = {encoded};
function definition(name) {{
  const value = definitions[name];
  if (!value) throw new Error(`Unbekanntes SIC-Goal: ${{name}}`);
  return value;
}}
async function call(name, values = {{}}) {{
  const endpoint = definition(name);
  const payload = {{...values}};
  let path = endpoint.path.replace(/\\{{([^}}]+)\\}}/g, (_, key) => {{
    if (!(key in payload)) throw new Error(`Fehlender Pfadwert: ${{key}}`);
    const value = encodeURIComponent(payload[key]);
    delete payload[key];
    return value;
  }});
  const options = {{method: endpoint.method, headers: {{Accept: "application/json"}}}};
  if (endpoint.method === "GET") {{
    const query = new URLSearchParams(payload).toString();
    if (query) path += (path.includes("?") ? "&" : "?") + query;
  }} else {{
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }}
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const result = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {{
    const error = new Error(result.detail || `HTTP ${{response.status}}`);
    error.response = result;
    throw error;
  }}
  return result;
}}
function show(name, value) {{
  document.querySelectorAll(`[data-sic-output="${{name}}"]`).forEach(node => {{
    node.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }});
}}
document.addEventListener("submit", async event => {{
  const form = event.target.closest("form[data-sic-submit]");
  if (!form) return;
  event.preventDefault();
  const name = form.dataset.sicSubmit;
  const values = Object.fromEntries(new FormData(form).entries());
  try {{
    const result = await call(name, values);
    show(name, result);
    document.dispatchEvent(new CustomEvent("sic:result", {{detail: {{name, result}}}}));
  }} catch (error) {{
    show(name, error.response || error.message);
    document.dispatchEvent(new CustomEvent("sic:error", {{detail: {{name, error}}}}));
  }}
}});
document.addEventListener("click", async event => {{
  const button = event.target.closest("[data-sic-call]");
  if (!button) return;
  const name = button.dataset.sicCall;
  const result = await call(name);
  show(name, result);
  document.dispatchEvent(new CustomEvent("sic:result", {{detail: {{name, result}}}}));
}});
window.sic = {{call, endpoints: definitions}};
""".strip()


def build_web(config_path: str | Path = "web.sic") -> list[Path]:
    """Kompiliert SIC-Seitengoals zu eigenständigen HTML-Dateien."""
    config = load_web_config(config_path)
    config.output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    runtime = _browser_runtime(config.endpoints)
    for source_path in config.pages:
        registry = standard_registry(config.root)
        page = Evaluator(
            compile_program(source_path.read_text(encoding="utf-8"), registry), registry
        ).run().value
        if not isinstance(page, dict):
            raise ValueError(f"{source_path}: Das Page-Goal muss einen Record liefern")
        output_name = _string(page, "output", str(source_path))
        title = page.get("title", config.title)
        body = page.get("html", "")
        css = page.get("css", "")
        script = page.get("js", "")
        if not all(isinstance(value, str) for value in (title, body, css, script)):
            raise ValueError(f"{source_path}: title, html, css und js müssen Strings sein")
        destination = _inside(config.output, output_name, "page.output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = "\n".join([
            "<!doctype html>", '<html lang="de">', "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>", f"<style>{css}</style>",
            "</head>", "<body>", body,
            f"<script type=\"module\">{runtime}\n\n{script}</script>",
            "</body>", "</html>", "",
        ])
        destination.write_text(document, encoding="utf-8")
        generated.append(destination)
    return generated


def create_app(config_path: str | Path = "web.sic"):
    """Erzeugt eine FastAPI-App; SIC-Endpunkte werden bei Änderungen neu geladen."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - verständlich bei Minimalinstallation
        raise RuntimeError("FastAPI fehlt; Projektabhängigkeiten erneut installieren") from exc

    config_path = Path(config_path).resolve()
    config = load_web_config(config_path)
    build_web(config_path)
    app = FastAPI(title=config.title)
    execution_lock = threading.RLock()  # schützt die vorläufige JSON-Dateipersistenz

    def response_for(value: Any):
        if (
            isinstance(value, dict) and type(value.get("status")) is int
            and "body" in value
        ):
            headers = value.get("headers", {})
            if not isinstance(headers, dict):
                headers = {}
            return JSONResponse(value["body"], status_code=value["status"], headers=headers)
        return JSONResponse(value)

    def make_handler(program: HotSicProgram):
        async def handler(request: Request):
            body: Any = None
            raw = await request.body()
            if raw:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError:
                        return JSONResponse({"detail": "Ungültiges JSON"}, status_code=400)
                else:
                    body = raw.decode("utf-8")
            query = dict(request.query_params)
            path_parameters = dict(request.path_params)
            request_value = {
                "method": request.method,
                "path": request.url.path,
                "query": query,
                "pathParams": path_parameters,
                "headers": dict(request.headers),
                "body": body,
            }
            arguments: dict[str, Any] = {"request": request_value, **query, **path_parameters}
            if isinstance(body, dict):
                arguments.update(body)
            try:
                value = await run_in_threadpool(program.execute, arguments)
                return response_for(value)
            except LangCalculationError as exc:
                detail = exc.payload if exc.payload is not None else exc.name
                return JSONResponse({"error": exc.name, "detail": detail}, status_code=422)
            except LangError as exc:
                return JSONResponse({"detail": str(exc)}, status_code=422)
            except (OSError, ValueError) as exc:
                return JSONResponse({"detail": str(exc)}, status_code=500)
        # ``Request`` wird lokal importiert; FastAPI benötigt hier das echte
        # Objekt statt der durch ``annotations`` verzögerten Zeichenkette.
        handler.__annotations__["request"] = Request
        return handler

    for endpoint in config.endpoints:
        program = HotSicProgram(endpoint.source, config.root, execution_lock)
        app.add_api_route(
            endpoint.path, make_handler(program), methods=[endpoint.method], name=endpoint.name
        )

    index = config.output / "index.html"
    if index.exists():
        @app.get("/", include_in_schema=False)
        async def index_page():
            return FileResponse(index)

    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=config.output), name="static")
    return app


def serve_web(config_path: str | Path = "web.sic", *, reload: bool = False) -> None:
    """Startet den Webserver. SIC-Endpunkte benötigen keinen Prozess-Restart."""
    import uvicorn

    config_path = Path(config_path).resolve()
    config = load_web_config(config_path)
    if reload:
        # Uvicorns Prozess-Reload benötigt einen Importstring; SIC-Hot-Reload ist
        # ohnehin aktiv, daher wird diese Option nur als Hinweis behandelt.
        print("Hinweis: SIC-Endpunkte werden bereits pro Anfrage auf Änderungen geprüft.")
    uvicorn.run(create_app(config_path), host=config.host, port=config.port)
