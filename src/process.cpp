#include "process.hpp"
#define PROVER_MODEL_ID 1
#define CODER_MODEL_ID 0
#define FORMAL_STATEMENT_ID true
#define INFORMAL_STATEMENT_ID false

struct flags {
    enum class MODEL { PROVER = 0x1, CODER = 0x0 };
    enum class STATEMENT { FORMAL = 0x2, INFORMAL = 0x0 };
};

int process_statement(char flags) {
    // create the child process
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        std::cerr << "Pipe failed";
        return 1;
    }

    pid_t pid = fork();
    pid_t child_id;
    
    if ( pid == -1 ) {
        // FIXME: is this check sufficient? or can fork return other negative numbers other than -1?
        std::cerr << "failed to fork" << std::endl;
        return 1;
    } else if ( pid  == 0 ) { 
        // Inside child process.
        close(pipefd[0]); // close read end of pipe
        dup2(pipefd[1],pipefd[0]);
    
        // Build path and argv for execv()'s second argument.
        char* path = "/home/yassine/.local/bin/llama-cli"; // FIXME: this path should not be hardcoded
        int model_id;
        int statement_type_id;
        
        // Arguments
        const char* option1 = "-m";
        const char* switch1 = (flags & PROVER_MODEL_ID) ? "/home/yassine/code/prover/vendor/models/deepseek-prover-v2-7B-BF16.gguf" : "/home/yassine/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf";
        
        const char* option2 = "-t";
        const char* switch2 = "8";
        
        const char* option3 = "-c";
        const char* switch3 = "2048";
        
        // Set statement option
        flags & 2 ? statement_type_id = FORMAL_STATEMENT_ID : statement_type_id = INFORMAL_STATEMENT_ID;
        std::vector<std::string> statements = parseJsonl(statement_type_id);
        if (statements.empty()) {
            std::cerr << "Failed to process benchmark statement" << std::endl;
            return 1;
        }
        const char* option4 = "-p";
        const char* switch4 = statements[0].c_str();
        
        // create a child process
        pid_t pid = fork();
        std::string llamacli = "/home/yassine/.local/bin/llama-cli";

        char * const argv = { option1, switch1, option2, switch2, option3, switch3, option4, switch4, nullptr };

        execv(path,argv);
        std::cerr << "Not good" << std::endl;
        perror("execv");
        return 1;
    } else {
        // Inside parent process.
        close(pipefd[1]); // close the write end of the pipe
        child_id = pid;
        std::cout << pid << " is the child pid." << std::endl;
        
        // read the output of the child
        std::string output;
        int condition = 1;
        char buffer[128];
        while ( condition > 0 ) {
            condition = read(pipefd[0],buffer,128);
            if ( condition < 0 ) {
                perror("process.cpp: read(pipefd[0],buffer,128)");
                return 1;
            }
            for (int i = 0; i < 128; i++ ) {
                output.push_back(buffer[i]);
            }
        }
        
        if (output.empty()) {
            std::cerr << "No output was received from the process." << std::endl;
            return 1;
        }

        std::regex end_of_turn("R\\[ Prompt: \\d+\\.\\d+ t/s \\| Generation: \\d+\\.\\d+ t/s \\]");
        if (std::regex_match(output,end_of_turn)) {
            if ( kill(child_id,SIGKILL) == -1 ) {
                std::cerr << "Failed to kill child process." << std::endl;
            }
        }
        close(pipefd[0]); // close the read end of the pipe
        return 0;
    }
}

int main() {
    char flags = 0x3;
    process_statement(flags);
    return 0;
}
