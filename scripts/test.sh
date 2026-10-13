#!/bin/bash

# Create a new temporary file
LGCC_TEMP="/home/yassine/code/prover/build/lgccTemp"
touch "$LGCC_TEMP"
# append to it the source file
# FIXME(legacy-test-hardcoded | tier:C): Parameterize source file input in legacy test script.
# Description: Replace hardcoded process.cpp path with argument variable.
# Criterion: review: "scripts/test.sh accepts target source file as $1"
cat "/home/yassine/code/prover/src/process.cpp" > "$LGCC_TEMP"

# FIXME(legacy-test-multilang | tier:C): Add multi-language compilation support.
# Description: Expand beyond cpp cmake build to support Lean and Rust sources.
# Criterion: review: "scripts/test.sh branches on file extension to call appropriate compiler"
# compile the file with cmake and 
# append the output to the temporary file
cd /home/yassine/code/prover/build/
cmake .. >> "$LGCC_TEMP"
make >> "$LGCC_TEMP"

llama-cli -m ~/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf -ngl 12 -t 8 -c 4096 --no-display-prompt -sys "You are a strict logical error detector. Focus exclusively on logic bugs (e.g., off-by-one, edge cases). Ignore styling, naming, and syntax. Output exactly one line per error in this format: [line N] <concise description>. Do not rewrite code. Do not include conversational filler or markdown." -f ./lgccTemp

# alias lgcc='llama-cli -m ~/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf -ngl 12 -t 8 -c 4096 --no-display-prompt -sys "You are a strict logical error detector for source code. Identify only genuine logic bugs that can cause incorrect runtime behavior (off-by-one, wrong loop bounds/conditions, missing/wrong edge cases, inverted conditionals, wrong operators/precedence that change semantics, incorrect data-range/state assumptions, ordering/race issues affecting correctness, logic-caused infinite loops, etc.). Strictly ignore style, formatting, naming, syntax/type errors, performance, readability, best practices, docs, and non-logic security issues. Output each distinct logic error on its own line exactly as: [line N] <one short sentence describing the bug and why it is incorrect>. Use the actual line number from the input. Do not suggest fixes or rewrite code. Output no other text whatsoever (no greetings, explanations, markdown, bullets, code blocks, or summaries)." -f'
