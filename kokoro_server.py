#!/usr/bin/env python3
"""Persistent Kokoro TTS server - streams sentence chunks.

Protocol (one connection per request):
  client -> one JSON line: {"text": ..., "voice": "af_heart", "speed": 1.0}
  server -> 1 status byte (0=ok, 1=error)
  then frames: [4-byte LE length][S16LE pcm@24000] repeated
  terminator: 4-byte zero length
"""
import json
import struct
import socketserver

import numpy as np
import torch
torch.set_num_threads(6)

from kokoro import KPipeline

PORT = 8081
SAMPLE_RATE = 24000
pipeline = KPipeline(lang_code="a")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline()
            if not line:
                return
            req = json.loads(line.decode())
            text = req.get("text", "")
            voice = req.get("voice", "af_heart")
            speed = float(req.get("speed", 1.0))

            self.wfile.write(struct.pack("B", 0))
            for res in pipeline(text, voice=voice, speed=speed, split_pattern=r"[.!?]"):
                pcm = (res.audio.numpy() * 32767).astype(np.int16).tobytes()
                self.wfile.write(struct.pack("<I", len(pcm)))
                self.wfile.write(pcm)
                self.wfile.flush()
            self.wfile.write(struct.pack("<I", 0))
            self.wfile.flush()
        except Exception:
            import traceback
            traceback.print_exc()
            try:
                self.wfile.write(struct.pack("B", 1))
                self.wfile.flush()
            except Exception:
                pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as srv:
        print(f"kokoro stream server ready on port {PORT}", flush=True)
        srv.serve_forever()