from pathlib import Path

from fastapi.testclient import TestClient

from src.web import build_web, create_app, load_web_config


def web_project(tmp_path: Path) -> Path:
    config = tmp_path / "web.sic"
    config.write_text(
        '''
        goal app = {
            title: "Test App",
            host: "127.0.0.1",
            port: 8123,
            output: "dist",
            pages: ["page.sic"],
            endpoints: [
                {name: "greet", method: "POST", path: "/api/greet", source: "greet.sic"}
            ]
        }
        ''',
        encoding="utf-8",
    )
    (tmp_path / "page.sic").write_text(
        '''
        goal page = {
            output: "index.html",
            title: "Start",
            html: "<h1>SIC</h1><form data-sic-submit=\\"greet\\"></form>",
            css: "h1 { color: red; }",
            js: "sic.call(\\"greet\\", {name: \\"Ada\\"});"
        }
        ''',
        encoding="utf-8",
    )
    (tmp_path / "greet.sic").write_text(
        'goal response: String = "Hallo " + (name: String)\n', encoding="utf-8"
    )
    return config


def test_sic_web_config_and_static_build(tmp_path):
    config_path = web_project(tmp_path)
    config = load_web_config(config_path)
    assert config.port == 8123
    assert config.endpoints[0].name == "greet"

    generated = build_web(config_path)
    assert generated == [tmp_path / "dist" / "index.html"]
    document = generated[0].read_text(encoding="utf-8")
    assert "<h1>SIC</h1>" in document
    assert 'window.sic = {call' in document
    assert '"greet":{"method":"POST","path":"/api/greet"}' in document


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
