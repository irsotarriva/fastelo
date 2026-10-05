"""Remote storage for runs that are stopped and resumed: an S3-compatible bucket or a local folder.

    store = open_store("s3://my-bucket")      # bucket from the URL, credentials from the environment:
                                              #   S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    store = open_store("file://mirror")     # same interface on a local folder (tests, dry runs)

Keys are "/"-separated paths inside the bucket. Everything is idempotent so a job restarted after
a stop can call the same functions again: `sync_down` skips files whose size already
matches, uploads go to a temporary key and are then copied into place, and markers (`mark`,
`is_marked`) record finished work.
"""
import json
import os
import shutil
import threading
import time


class LocalStore:
    def __init__(self, root):
        self.root = root
        self.mtime = {}   # key -> last-modified time (epoch seconds), filled by list()

    def _p(self, key):
        return os.path.join(self.root, *key.split("/"))

    def put_file(self, local, key):
        dst = self._p(key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(local, dst + ".part")
        os.replace(dst + ".part", dst)

    def get_file(self, key, local):
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        shutil.copyfile(self._p(key), local + ".part")
        os.replace(local + ".part", local)

    def list(self, prefix):
        """{key: size} of every object under `prefix`."""
        base = self._p(prefix.rstrip("/"))
        out = {}
        if os.path.isfile(base):
            return {prefix: os.path.getsize(base)}
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.endswith(".part"):
                    continue
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                out[rel] = os.path.getsize(full)
                self.mtime[rel] = os.path.getmtime(full)
        return out

    def exists(self, key):
        return os.path.isfile(self._p(key))

    def delete(self, key):
        if self.exists(key):
            os.remove(self._p(key))


class S3Store:
    def __init__(self, bucket):
        import boto3
        from botocore.config import Config
        self.bucket = bucket
        self.mtime = {}   # key -> last-modified time (epoch seconds), filled by list()
        self.s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"],
                               config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))

    def put_file(self, local, key):
        self.s3.upload_file(local, self.bucket, key)  # multipart for big files; atomic on completion

    def get_file(self, key, local):
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        self.s3.download_file(self.bucket, key, local + ".part")
        os.replace(local + ".part", local)

    def list(self, prefix):
        out = {}
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                out[obj["Key"]] = obj["Size"]
                self.mtime[obj["Key"]] = obj["LastModified"].timestamp()
        return out

    def exists(self, key):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def open_store(url):
    if url.startswith("file://"):
        return LocalStore(url[len("file://"):])
    if url.startswith("s3://"):
        return S3Store(url.split("://", 1)[1].strip("/"))
    raise ValueError(f"unsupported store URL {url!r} (use s3://bucket or file://path)")


# ------------------------------------------------------------------------------ helpers
def sync_down(store, prefix, local_dir):
    """Download every object under `prefix` into local_dir (same relative layout), skipping files
    that already exist with the right size and are not older than the remote copy (a checkpoint
    keeps its size from one save to the next). Returns the number of files fetched."""
    n = 0
    prefix = prefix.rstrip("/") + "/"
    for key, size in sorted(store.list(prefix).items()):
        local = os.path.join(local_dir, *key[len(prefix):].split("/"))
        if (os.path.exists(local) and os.path.getsize(local) == size
                and os.path.getmtime(local) >= store.mtime.get(key, 0)):
            continue
        store.get_file(key, local)
        n += 1
    return n


def sync_up(store, local_dir, prefix):
    """Upload every file under local_dir that is missing or has a different size remotely."""
    remote = store.list(prefix.rstrip("/") + "/")
    n = 0
    for dirpath, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith((".part", ".tmp")):
                continue
            full = os.path.join(dirpath, f)
            key = prefix.rstrip("/") + "/" + os.path.relpath(full, local_dir).replace(os.sep, "/")
            if remote.get(key) != os.path.getsize(full):
                store.put_file(full, key)
                n += 1
    return n


def mark(store, key, info=None):
    """Record finished work (a small JSON object)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dict(info or {}, time=time.strftime("%Y-%m-%d %H:%M:%S")), f)
    store.put_file(f.name, key)
    os.remove(f.name)


def is_marked(store, key):
    return store.exists(key)


class BackgroundUploader:
    """Uploads files in a background thread; a newer request for the same key replaces an older
    pending one (only the latest checkpoint matters). `flush()` waits until everything is sent."""

    def __init__(self, store):
        self.store, self.pending, self.lock = store, {}, threading.Condition()
        self.busy, self.errors = False, []
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, local, key):
        with self.lock:
            self.pending[key] = local
            self.lock.notify()

    def _loop(self):
        while True:
            with self.lock:
                while not self.pending:
                    self.lock.wait()
                key, local = next(iter(self.pending.items()))
                del self.pending[key]
                self.busy = True
            try:
                # copy first so the trainer can overwrite its local file while we upload
                snap = local + ".upload"
                shutil.copyfile(local, snap)
                self.store.put_file(snap, key)
                os.remove(snap)
            except Exception as e:  # noqa: BLE001 - keep training; report on flush
                self.errors.append(f"{key}: {e}")
            with self.lock:
                self.busy = False
                self.lock.notify_all()

    def flush(self, timeout=3600):
        end = time.time() + timeout
        with self.lock:
            while (self.pending or self.busy) and time.time() < end:
                self.lock.wait(timeout=1)
        if self.errors:
            raise RuntimeError("uploads failed: " + "; ".join(self.errors[-3:]))


def main(argv=None):
    """python -m fastelo.cloud push LOCAL_DIR PREFIX --remote s3://bucket [--exclude aux.npy ...]
       python -m fastelo.cloud pull PREFIX LOCAL_DIR --remote s3://bucket
       python -m fastelo.cloud ls PREFIX --remote s3://bucket"""
    import argparse
    import fnmatch
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull", "ls"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--remote", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="file name patterns to skip on push")
    args = ap.parse_args(argv)
    store = open_store(args.remote)
    if args.cmd == "ls":
        items = store.list(args.a)
        print(f"{len(items)} objects, {sum(items.values()) / 1e9:.2f} GB")
        return
    if args.cmd == "pull":
        print(f"fetched {sync_down(store, args.a, args.b)} files")
        return
    prefix = args.b.strip("/")
    join = lambda rel: f"{prefix}/{rel}" if prefix else rel
    remote = store.list(prefix + "/" if prefix else "")
    n = total = 0
    for dirpath, _, files in os.walk(args.a):
        for f in sorted(files):
            if f.endswith((".part", ".tmp")) or any(fnmatch.fnmatch(f, pat) for pat in args.exclude):
                continue
            full = os.path.join(dirpath, f)
            key = join(os.path.relpath(full, args.a).replace(os.sep, "/"))
            if remote.get(key) == os.path.getsize(full):
                continue
            store.put_file(full, key)
            n += 1
            total += os.path.getsize(full)
            print(f"{key} ({os.path.getsize(full) / 1e6:.0f} MB)", flush=True)
    print(f"pushed {n} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
