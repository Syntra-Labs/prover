#!/bin/bash

# Usage / help check
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [target_source_file]"
    echo "Compiles target source file (.cpp, .lean, .rs) and runs llama-cli logic analyzer."
    exit 0
fi

# Locate repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Target source file defaults to src/process.cpp relative to repository root
TARGET_FILE="${1:-"${REPO_ROOT}/src/process.cpp"}"

# Resolve relative paths
if [[ "$TARGET_FILE" != /* ]]; then
    if [ -f "$TARGET_FILE" ]; then
        TARGET_FILE="$(pwd)/$TARGET_FILE"
    elif [ -f "${REPO_ROOT}/$TARGET_FILE" ]; then
        TARGET_FILE="${REPO_ROOT}/$TARGET_FILE"
    fi
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: target source file '$TARGET_FILE' does not exist." >&2
    exit 1
fi

# Create a new temporary file in build directory
BUILD_DIR="${REPO_ROOT}/build"
mkdir -p "$BUILD_DIR"
LGCC_TEMP="${BUILD_DIR}/lgccTemp"
touch "$LGCC_TEMP"

# Append source file to temporary file
cat "$TARGET_FILE" > "$LGCC_TEMP"

# Detect file extension and branch to appropriate compiler
EXT="${TARGET_FILE##*.}"

case "$EXT" in
    cpp|cc|cxx)
        # compile the file with cmake and make in the build directory
        # append the output to the temporary file
        cd "$BUILD_DIR" || exit 1
        cmake "$REPO_ROOT" >> "$LGCC_TEMP" 2>&1 || true
        make >> "$LGCC_TEMP" 2>&1 || true
        ;;
    lean)
        # invoke lean on the source file
        if command -v lean >/dev/null 2>&1; then
            lean "$TARGET_FILE" >> "$LGCC_TEMP" 2>&1 || true
        else
            echo "Error: 'lean' compiler not found in PATH." >&2
            exit 1
        fi
        ;;
    rs)
        # invoke rustc or cargo test --no-run if applicable
        CARGO_DIR="$(dirname "$TARGET_FILE")"
        CARGO_TOML=""
        while [ "$CARGO_DIR" != "/" ] && [ "$CARGO_DIR" != "." ]; do
            if [ -f "$CARGO_DIR/Cargo.toml" ]; then
                CARGO_TOML="$CARGO_DIR/Cargo.toml"
                break
            fi
            CARGO_DIR="$(dirname "$CARGO_DIR")"
        done

        if [ -n "$CARGO_TOML" ] && command -v cargo >/dev/null 2>&1; then
            cargo test --no-run --manifest-path "$CARGO_TOML" >> "$LGCC_TEMP" 2>&1 || true
        elif command -v rustc >/dev/null 2>&1; then
            rustc "$TARGET_FILE" -o /dev/null >> "$LGCC_TEMP" 2>&1 || rustc "$TARGET_FILE" >> "$LGCC_TEMP" 2>&1 || true
        else
            echo "Error: neither 'cargo' nor 'rustc' found in PATH." >&2
            exit 1
        fi
        ;;
    *)
        echo "Error: unsupported file extension '.$EXT' for '$TARGET_FILE'. Supported extensions: .cpp, .lean, .rs" >&2
        exit 1
        ;;
esac

# Safe invocation of llama-cli logical error detector
MODEL_PATH="${REPO_ROOT}/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf"
if [ ! -f "$MODEL_PATH" ] && [ -f "$HOME/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf" ]; then
    MODEL_PATH="$HOME/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf"
fi

if command -v llama-cli >/dev/null 2>&1 && [ -f "$MODEL_PATH" ]; then
    llama-cli -m "$MODEL_PATH" -ngl 12 -t 8 -c 4096 --no-display-prompt -sys "You are a strict logical error detector. Focus exclusively on logic bugs (e.g., off-by-one, edge cases). Ignore styling, naming, and syntax. Output exactly one line per error in this format: [line N] <concise description>. Do not rewrite code. Do not include conversational filler or markdown." -f "$LGCC_TEMP"
else
    echo "Notice: llama-cli logic analyzer skipped (llama-cli binary or model at '$MODEL_PATH' not available)."
fi
