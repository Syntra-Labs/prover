#!/bin/bash

# Create a new temporary file
touch /home/yassine/.lgccTemp
# append to it the source file
# FIXME: don't hardcord the source code
cat "/home/yassine/code/prover/src/process.cpp" > /home/yassine/.lgccTemp

# FIXME: for now we're assuming that the files will be cpp, but we should add support for other languages in the future
# compile the file with cmake and 
# append the output to the temporary file
cd /home/yassine/code/prover/build/
cmake .. >> /home/yassine/.lgccTemp
make >> /home/yassine/.lgccTemp

llama-cli -m ~/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf -ngl 12 -t 8 -c 4096 --no-display-prompt -sys "You are a strict logical error detector. Focus exclusively on logic bugs (e.g., off-by-one, edge cases). Ignore styling, naming, and syntax. Output exactly one line per error in this format: [line N] <concise description>. Do not rewrite code. Do not include conversational filler or markdown." -f ./lgccTemp

# alias lgcc='llama-cli -m ~/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf -ngl 12 -t 8 -c 4096 --no-display-prompt -sys "You are a strict logical error detector for source code. Identify only genuine logic bugs that can cause incorrect runtime behavior (off-by-one, wrong loop bounds/conditions, missing/wrong edge cases, inverted conditionals, wrong operators/precedence that change semantics, incorrect data-range/state assumptions, ordering/race issues affecting correctness, logic-caused infinite loops, etc.). Strictly ignore style, formatting, naming, syntax/type errors, performance, readability, best practices, docs, and non-logic security issues. Output each distinct logic error on its own line exactly as: [line N] <one short sentence describing the bug and why it is incorrect>. Use the actual line number from the input. Do not suggest fixes or rewrite code. Output no other text whatsoever (no greetings, explanations, markdown, bullets, code blocks, or summaries)." -f'
