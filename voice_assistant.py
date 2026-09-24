#!/usr/bin/env python3
"""Speech-to-speech assistant: Mic -> Whisper -> Qwen -> Kokoro -> Speaker"""

import subprocess
import os
import sys
import threading
import queue
import json
import time
import socket
import struct
import http.client
import urllib.request

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL = "base.en"
QWEN_API = "http://127.0.0.1:8080/v1/chat/completions"
KOKORO_VOICE = "af_bella"
KOKORO_PORT = 8081
STREAM_CHUNK_MIN = 40

whisper_model = None
conversation_history = []
_aplay_proc = None
_wlock = threading.Lock()
_sentence_q = queue.Queue()
_press = threading.Event()
_interrupt = threading.Event()
_tts_sock = None
_tts_worker_t = None


_line_q = queue.Queue()


def _stdin_reader():
    for line in sys.stdin:
        _line_q.put(line.strip())
        _press.set()


def _read(timeout=None):
    try:
        return _line_q.get(timeout=timeout)
    except queue.Empty:
        return None


def _drain():
    try:
        while True:
            _line_q.get_nowait()
    except queue.Empty:
        pass


def load_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("Whisper ready.")


def record_audio():
    print("●", end="", flush=True)
    frames = []
    stop = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stop.is_set():
            frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=1024, callback=callback):
        _read()
        stop.set()

    if not frames:
        return None
    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio):
    with _wlock:
        segments, _ = whisper_model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    if text:
        print(f"User: {text}")
    return text


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("connection closed")
        data += chunk
    return data


def tts_chunks(text):
    global _tts_sock
    sock = socket.create_connection(("127.0.0.1", KOKORO_PORT), timeout=5)
    sock.settimeout(30)
    _tts_sock = sock
    try:
        req = json.dumps({"text": text, "voice": KOKORO_VOICE, "speed": 1.25}).encode() + b"\n"
        sock.sendall(req)
        if recv_exact(sock, 1) != b"\x00":
            raise RuntimeError("kokoro synthesis failed")
        while True:
            (n,) = struct.unpack("<I", recv_exact(sock, 4))
            if n == 0:
                break
            yield recv_exact(sock, n)
    finally:
        if _tts_sock is sock:
            _tts_sock = None
        sock.close()


def get_aplay():
    global _aplay_proc
    if _aplay_proc is None or _aplay_proc.poll() is not None:
        _aplay_proc = subprocess.Popen(
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", "24000", "-c", "1", "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    return _aplay_proc


def write_pcm(pcm):
    if _interrupt.is_set():
        return
    proc = get_aplay()
    try:
        proc.stdin.write(pcm)
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        proc = get_aplay()
        try:
            proc.stdin.write(pcm)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass


def speak(text):
    print("Speaking...")
    for pcm in tts_chunks(text):
        write_pcm(pcm)


def _tts_worker():
    while True:
        sentence = _sentence_q.get()
        if sentence is None:
            break
        if _interrupt.is_set():
            continue
        try:
            for pcm in tts_chunks(sentence):
                if _interrupt.is_set():
                    break
                write_pcm(pcm)
        except (EOFError, ConnectionError, OSError) as e:
            if not _interrupt.is_set():
                print(f"[tts error: {e}]", flush=True)


def _next_sentence_boundary(s):
    for i, ch in enumerate(s):
        if ch in ".!?":
            nxt = s[i + 1:i + 2]
            if not nxt or nxt in (" \t\n\r") or nxt.isupper():
                if any(c.isalpha() for c in s[:i]):
                    return i
    return None


def ask_qwen_stream(prompt):
    system_msg = "You are a helpful voice assistant. Keep each reply short, plain, and spoken-style. No lists or formatting."

    payload = {
        "messages": [{"role": "system", "content": system_msg}]
                  + conversation_history
                  + [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
    }
    conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=300)
    full = ""
    buf = ""
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
            except Exception:
                continue
            if not content:
                continue
            full += content
            yield ("token", content)
            buf += content
            while True:
                idx = _next_sentence_boundary(buf)
                if idx is None:
                    break
                seg = buf[:idx + 1].strip()
                if seg:
                    yield ("sentence", seg)
                buf = buf[idx + 1:]
            s = buf.rstrip(" \t\n")
            if len(s) >= STREAM_CHUNK_MIN and s[-1] in " \t" and any(c.isalpha() for c in s):
                yield ("sentence", s.strip())
                buf = ""
        tail = buf.strip()
        if tail:
            yield ("sentence", tail)
    finally:
        conn.close()

    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": full.strip()})
    if len(conversation_history) > 20:
        conversation_history[:] = conversation_history[-20:]
    yield ("done", full.strip())


def stream_assistant(text):
    global _tts_worker_t
    _interrupt.clear()
    _press.clear()
    while True:
        try:
            _sentence_q.get_nowait()
        except queue.Empty:
            break

    if _tts_worker_t and _tts_worker_t.is_alive():
        _tts_worker_t.join(1.0)

    stop_watch = threading.Event()

    def watcher():
        while not stop_watch.is_set():
            if _press.wait(timeout=0.1):
                _press.clear()
                _interrupt.set()
                proc = _aplay_proc
                if proc and proc.poll() is None:
                    proc.kill()
                sock = _tts_sock
                if sock:
                    try:
                        sock.close()
                    except OSError:
                        pass
                return

    threading.Thread(target=watcher, daemon=True).start()

    first = True
    result = None
    worker = threading.Thread(target=_tts_worker, daemon=True)
    _tts_worker_t = worker
    worker.start()
    try:
        for kind, val in ask_qwen_stream(text):
            if _interrupt.is_set():
                break
            if kind == "token":
                if first:
                    print("\nModel: ", end="", flush=True)
                    first = False
                print(val, end="", flush=True)
            elif kind == "sentence":
                _sentence_q.put(val)
            else:
                print()
                result = val
    finally:
        _sentence_q.put(None)
        if _interrupt.is_set():
            worker.join(3.0)
        else:
            worker.join()
        stop_watch.set()
    return result


def ensure_server():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3) as resp:
            if '"ok"' in resp.read().decode():
                return
    except Exception:
        pass
    print("Starting llama-server...")
    subprocess.run([os.path.expanduser("~/code/speak/start_server.sh")])
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3) as resp:
                if '"ok"' in resp.read().decode():
                    print("llama-server ready.")
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready")


def kokoro_alive():
    try:
        with socket.create_connection(("127.0.0.1", KOKORO_PORT), timeout=2):
            return True
    except OSError:
        return False


def ensure_kokoro():
    if kokoro_alive():
        return
    print("Starting kokoro server...")
    subprocess.run([os.path.expanduser("~/code/speak/start_kokoro.sh")])
    for _ in range(120):
        if kokoro_alive():
            print("kokoro server ready.")
            return
        time.sleep(1)
    raise RuntimeError("kokoro server did not become ready")


def main():
    print("Voice Assistant - Ctrl+C to exit")
    print("■ Enter to record. Enter-Enter switches to the >_ typing prompt.")
    print(">_ type a message + Enter to send. Enter-Enter switches back to voice.")
    print("During a reply, pressing Enter cuts it off and starts recording.")
    print("Commands (type at either prompt): /new resets context, /voice, /type, /help")
    threading.Thread(target=_stdin_reader, daemon=True).start()
    load_whisper()
    ensure_server()
    ensure_kokoro()
    mode = "record"
    while True:
        try:
            if _interrupt.is_set():
                _interrupt.clear()
                _drain()
                audio = record_audio()
                text = transcribe(audio) if audio is not None else ""
            else:
                print("■" if mode == "record" else ">_", end="", flush=True)
                line = _read()
                if not line:
                    second = _read(0.35)
                    if second == "":
                        print()
                        mode = "record" if mode == "type" else "type"
                        continue
                    if second is not None:
                        _line_q.put(second)
                    if mode == "type":
                        continue
                    audio = record_audio()
                    text = transcribe(audio) if audio is not None else ""
                else:
                    if line.startswith("/"):
                        cmd = line.split()[0].lower()
                        if cmd == "/new":
                            conversation_history[:] = []
                            print()
                            print("(context reset)")
                        elif cmd == "/voice":
                            mode = "record"
                            print("(voice mode)")
                        elif cmd == "/type":
                            mode = "type"
                            print("(typing mode)")
                        else:
                            print("(commands: /new /voice /type /help)")
                        continue
                    text = line
                    print(f"User: {text}")
            if not text:
                print("No audio recorded.")
                continue
            stream_assistant(text)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
