import usocket


class Response:

    def __init__(self, f):
        self.raw = f
        self.status_code = None
        self.reason = None
        self.headers = []
        self.encoding = "utf-8"
        self._cached = None

    def close(self):
        if self.raw:
            self.raw.close()
            self.raw = None
        self._cached = None

    @property
    def content(self):
        if self._cached is None:
            try:
                self._cached = self.raw.read()
            finally:
                self.raw.close()
                self.raw = None
        return self._cached

    @property
    def text(self):
        return str(self.content, self.encoding)

    def json(self):
        import ujson
        return ujson.loads(self.content)


def request(method, url, data=None, json=None, stream=None, headers=None, store_resp_headers=False,
            timeout=15.0, bufsize=256):

    if headers is None:
        headers = {}

    try:
        proto, dummy, host, path = url.split("/", 3)
    except ValueError:
        proto, dummy, host = url.split("/", 2)
        path = ""
    if proto == "http:":
        port = 80
    elif proto == "https:":
        import ussl
        port = 443
    else:
        raise ValueError("Unsupported protocol: " + proto)

    if ":" in host:
        host, port = host.split(":", 1)
        port = int(port)

    ai = usocket.getaddrinfo(host, port, 0, usocket.SOCK_STREAM)
    ai = ai[0]

    s = usocket.socket(ai[0], ai[1], ai[2])
    s.settimeout(timeout)
    try:
        s.connect(ai[-1])
        if proto == "https:":
            s = ussl.wrap_socket(s, server_hostname=host)
        httpline = b"%s /%s HTTP/1.0\r\n" % (method, path)
        if "Host" not in headers:
            headers['Host'] = host
        if json is not None:
            assert data is None
            import ujson
            data = ujson.dumps(json)
            headers['Content-Type'] = 'application/json'
        if data:
            if isinstance(data, str):
                data = data.encode()
            headers['Content-Length'] = len(data)
        # Iterate over keys to avoid tuple alloc
        s.write(httpline)
        for k, v in headers.items():
            s.write('{}: {}\r\n'.format(k, str(v)).encode())
        s.write(b"\r\n")
        if stream is not None:
            while True:
                chunk = stream.read(bufsize)
                if not chunk:
                    break
                if isinstance(chunk, str) and not isinstance(chunk, bytes):
                    chunk = chunk.encode()
                socket_sent = 0
                while socket_sent < len(chunk):  # Ensure we're sending all the data
                    socket_sent += s.write(chunk[socket_sent:])
        elif data:
            s.write(data)

        l = s.readline()
        # print(l)
        l = l.split(None, 2)
        status = int(l[1])
        reason = ""
        resp_headers = []
        if len(l) > 2:
            reason = l[2].rstrip()
        while True:
            l = s.readline()
            if not l or l == b"\r\n":
                break
            # print(l)
            if l.startswith(b"Transfer-Encoding:"):
                if b"chunked" in l:
                    raise ValueError("Unsupported " + l)
            elif l.startswith(b"Location:") and not 200 <= status <= 299:
                raise NotImplementedError("Redirects not yet supported")
            if store_resp_headers:
                resp_headers.append(l)
    except OSError:
        s.close()
        raise

    resp = Response(s)
    resp.status_code = status
    resp.reason = reason
    resp.headers = resp_headers
    return resp


def head(url, **kw):
    return request("HEAD", url, **kw)


def get(url, **kw):
    return request("GET", url, **kw)


def post(url, **kw):
    return request("POST", url, **kw)


def put(url, **kw):
    return request("PUT", url, **kw)


def patch(url, **kw):
    return request("PATCH", url, **kw)


def delete(url, **kw):
    return request("DELETE", url, **kw)
