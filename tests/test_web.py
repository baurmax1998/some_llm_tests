from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.lang import LangTypeError
from src.web import build_web, create_app, load_web_config


def web_project(tmp_path: Path) -> Path:
    config = tmp_path / "web.sic"
    config.write_text(
        '''
        goal app: App::web = {
            title: "Test App",
            host: "127.0.0.1",
            port: 8123,
            output: ".out",
            modules: [
                {module: "ui", package: "components", source: "card.sic"}
            ],
            pages: ["page.sic"],
            endpoints: [
                {name: "greet", method: "POST", path: "/api/greet", source: "greet.sic"}
            ]
        }
        ''',
        encoding="utf-8",
    )
    (tmp_path / "card.sic").write_text(
        '''
        goal card: Html::web = element::web("article", {"class": "card"}, [
            element::web("h1", {"class": "title"}, [text::web(name: String)])
        ])
        ''',
        encoding="utf-8",
    )
    (tmp_path / "page.sic").write_text(
        '''
        goal page: Page::web = {
            output: "index.html",
            title: "Start",
            html: card::ui::components(name: "SIC"),
            css: "",
            js: "sic.call(\\"greet\\", {name: \\"Ada\\"});"
        }
        ''',
        encoding="utf-8",
    )
    (tmp_path / "greet.sic").write_text(
        'goal response: String = "Hallo " + (name: String)\n', encoding="utf-8"
    )
    return config


def test_minimal_typed_app_uses_web_defaults(tmp_path):
    config_path = tmp_path / "app.sic"
    config_path.write_text(
        '''
        goal app: App::web = {
            port: 8000,
            pages: [],
            endpoints: []
        }
        ''',
        encoding="utf-8",
    )
    config = load_web_config(config_path)
    assert config.title == "SIC App"
    assert config.host == "127.0.0.1"
    assert config.output == tmp_path / ".out"
    assert config.modules == ()


def test_sic_web_config_and_static_build(tmp_path):
    config_path = web_project(tmp_path)
    config = load_web_config(config_path)
    assert config.port == 8123
    assert config.endpoints[0].name == "greet"

    generated = build_web(config_path)
    assert generated == [tmp_path / ".out" / "index.html"]
    document = generated[0].read_text(encoding="utf-8")
    assert '<article class="card"><h1 class="title">SIC</h1></article>' in document
    assert 'window.sic = {call' in document
    assert '"greet":{"method":"POST","path":"/api/greet"}' in document


def test_web_goal_requires_standard_library_type(tmp_path):
    config_path = web_project(tmp_path)
    page_path = tmp_path / "page.sic"
    page_path.write_text(
        '''
        goal page = {
            output: "index.html",
            title: "Start",
            html: text::web(""),
            css: "",
            js: ""
        }
        ''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="View::web oder Page::web annotiert"):
        build_web(config_path)


def test_page_type_requires_all_fields(tmp_path):
    config_path = web_project(tmp_path)
    page_path = tmp_path / "page.sic"
    page_path.write_text(
        '''
        goal page: Page::web = {
            output: "index.html",
            title: "Start",
            html: text::web(""),
            css: ""
        }
        ''',
        encoding="utf-8",
    )
    with pytest.raises(LangTypeError, match="Record fehlt Feld 'js'"):
        build_web(config_path)


def test_websocket_recomputes_live_view_after_file_effect(tmp_path):
    config_path = tmp_path / "app.sic"
    config_path.write_text(
        '''
        goal app: App::web = {
            port: 8000,
            pages: ["page.sic"],
            endpoints: [
                {name: "setValue", method: "POST", path: "/value", source: "set.sic"}
            ]
        }
        ''',
        encoding="utf-8",
    )
    (tmp_path / "page.sic").write_text(
        'goal view: View::web = Text::web(read_text::files::std("value.txt"))',
        encoding="utf-8",
    )
    (tmp_path / "set.sic").write_text(
        'goal saved: Bool = write_text::files::std(value: String, "value.txt")',
        encoding="utf-8",
    )
    (tmp_path / "value.txt").write_text("vorher", encoding="utf-8")

    app = create_app(config_path)
    with TestClient(app) as client:
        with client.websocket_connect("/__sic_ws?page=/") as websocket:
            assert websocket.receive_json() == {"type": "replace", "html": "vorher"}
            websocket.send_json({
                "type": "call", "id": "1", "name": "setValue",
                "values": {"value": "nachher"},
            })
            assert websocket.receive_json() == {
                "type": "result", "id": "1", "result": True,
            }
            assert websocket.receive_json() == {"type": "replace", "html": "nachher"}


def test_websocket_transports_post_put_and_delete_actions(tmp_path):
    config_path = tmp_path / "app.sic"
    config_path.write_text(
        '''
        goal app: App::web = {
            port: 8000,
            pages: ["page.sic"],
            endpoints: [
                {name: "postValue", method: "POST", path: "/values", source: "method.sic"},
                {name: "putValue", method: "PUT", path: "/values/{id}", source: "method.sic"},
                {name: "deleteValue", method: "DELETE", path: "/values/{id}", source: "method.sic"}
            ]
        }
        ''',
        encoding="utf-8",
    )
    (tmp_path / "page.sic").write_text(
        'goal view: View::web = Text::web("bereit")', encoding="utf-8"
    )
    (tmp_path / "method.sic").write_text(
        'goal response: String = request.method: String', encoding="utf-8"
    )

    app = create_app(config_path)
    with TestClient(app) as client:
        with client.websocket_connect("/__sic_ws?page=/") as websocket:
            assert websocket.receive_json()["type"] == "replace"
            calls = [
                ("postValue", {"value": "a"}, "POST"),
                ("putValue", {"id": 7, "value": "b"}, "PUT"),
                ("deleteValue", {"id": 7}, "DELETE"),
            ]
            for index, (name, values, method) in enumerate(calls, 1):
                websocket.send_json({
                    "type": "call", "id": str(index),
                    "name": name, "values": values,
                })
                assert websocket.receive_json() == {
                    "type": "result", "id": str(index), "result": method,
                }


def test_fastapi_forwards_json_to_hot_sic_goal(tmp_path):
    config_path = web_project(tmp_path)
    app = create_app(config_path)
    with TestClient(app) as client:
        assert client.post("/api/greet", json={"name": "Ada"}).json() == "Hallo Ada"

        # Die Route bleibt bestehen, nur die SIC-Logik wird ausgetauscht.
        source = tmp_path / "greet.sic"
        source.write_text(
            'goal response: String = "Hi " + (name: String) + "!"\n', encoding="utf-8"
        )
        assert client.post("/api/greet", json={"name": "Ada"}).json() == "Hi Ada!"
