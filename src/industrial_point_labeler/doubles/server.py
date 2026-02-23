"""HTTP server for the doubles dissection tool."""
import http.server
import argparse
import json
import webbrowser
from pathlib import Path
from functools import partial


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, app_state=None, **kwargs):
        self.app = app_state
        super().__init__(*args, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def translate_path(self, path):
        if path == '/' or path == '/index.html':
            return str(Path(__file__).parent / 'viewer.html')
        if path.startswith('/doubles/'):
            return str(self.app["data_dir"] / path[len('/doubles/'):])
        return super().translate_path(path)

    def do_POST(self):
        if self.path == '/save_splits':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            splits_path = self.app["splits_path"]
            existing = {}
            if splits_path.exists():
                with open(splits_path) as f:
                    existing = json.load(f)
            existing.update(data)

            with open(splits_path, 'w') as f:
                json.dump(existing, f, indent=2)

            total_groups = sum(len(gs) for gs in existing.values())
            print(f'Saved splits: {len(existing)} segments, {total_groups} groups -> {splits_path}')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'path': str(splits_path)}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        if 'POST' in str(args):
            http.server.SimpleHTTPRequestHandler.log_message(self, format, *args)


def main():
    parser = argparse.ArgumentParser(description="Doubles dissection tool")
    parser.add_argument("--data-dir", required=True, help="Directory with manifest.json and PLY files")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    splits_path = data_dir / "splits.json"

    app_state = {
        "data_dir": data_dir,
        "splits_path": splits_path,
    }
    handler = partial(Handler, app_state=app_state)
    server = http.server.HTTPServer(('', args.port), handler)
    url = f'http://localhost:{args.port}'
    print(f'Doubles dissection tool: {url}')
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
