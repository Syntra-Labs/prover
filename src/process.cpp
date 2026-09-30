// #include <pybind11/embed.h>
#include "process.hpp"

/**
int preparePybind11() {
    py::scoped_interpreter python{}; // uniform initialization intoduced in c++11
    py::module_::import("numpy");
}
**/

void callModel(bool prover) {

    // TODO: load bechmark dataset and read inpu
    // parse the json data set and store the statements
    const bool FORMAL = true;
    const bool INFORMAL = false;
    std::vector<std::string> statements = parseJsonl(FORMAL);

    // print statements to test
    /**
    for (std::string statement : statements) {
        std::cout << statement << std::endl;
    }
    **/

    const std::string PROVER = "~/code/prover/vendor/models/deepseek-prover-v2-7b/deepseek-prover-v2-7B-BF16.gguf";
    const std::string CODER = "~/code/prover/vendor/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q6_k.gguf";
    std::string model;
    if (prover == true) {
        model = PROVER;
    } else {
        model = CODER;
    }

    /**
    Upon which condition does the line of code after 'popen(command)' runs?
    If popen() does not return control to the subsequent instructions because  are blocked
    **/

    const int MEM_SIZE = 1024;
    int file_descriptor = shm_open("/my_shared_memory", O_CREAT | O_RDWR, 0666);
    ftruncate(file_descriptor, MEM_SIZE);
    char* mem = const_cast<char*>(mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor, 0));

    // create a child process
    pid_t pid = fork();
    std::string llamacli = "/home/yassine/.local/bin/llama-cli";
    std::string parameters[] = {
        "-m", model,
        "-ngl", "12",
        "-t", "8",
        "-c", "2048",
        "-p", statements[0],
    };

    const int STDOUT = 1;
    dup2(file_descriptor,STDOUT);

    if (execv(llamacli,parameters) == -1) {
        std::cerr << "execv() failed" << std::endl;
    }
 
    std::regex eof(R"([ Prompt: \d+.\d+ t/s | Generation: \d+.\d+ t/s ])");

    // each character is a byte, therefore there will be MEM_SIZE number of characters in the shared buffer
    std::string output(mem,MEM_SIZE);
    if (std::regex_match(output,eof)) {
        // kill the process running llama-cli
    }

    return;
}

int main() {
    bool prover = true;
    bool coder = false;
    callModel(coder);
    return 0;
}
