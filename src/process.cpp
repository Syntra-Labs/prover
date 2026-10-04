#include "process.hpp"
#define PROVER_MODEL_ID 1
#define CODER_MODEL_ID 0
#define FORMAL_STATEMENT_ID true
#define INFORMAL_STATEMENT_ID false

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
        dup2(pipefd[1],STDOUT_FILENO);
        
        // FIXME: warning: ISO C++ forbids converting a string constant to ‘char*’ [-Wwrite-strings]
        //
        // To fix this, create copies of the string constants into char*, and add the null terminator. Ensuring the original string constants remain safe from modification.

        // Build path and argv for execv()'s second argument.
        char * path = "/home/yassine/.local/bin/llama-cli"; // FIXME: this path should not be hardcoded
        int model_id;
        int statement_type_id;

        char * option0 = "llama-cli";
        
        // Arguments
        char * option1 = "-m";
        char * switch1 = (flags & PROVER_MODEL_ID) ? const_cast<char*>("/home/yassine/code/prover/vendor/models/deepseek-prover-v2-7b-q4_k_m-imat.gguf") : const_cast<char*>("/home/yassine/code/prover/vendor/models/qwen2.5-coder-7b-instruct-q6_k.gguf");
        
        char * option2 = "-t";
        char * switch2 = "8";
        
        char * option3 = "-c";
        char * switch3 = "2048";
        
        // Set statement option
        flags & 2 ? statement_type_id = FORMAL_STATEMENT_ID : statement_type_id = INFORMAL_STATEMENT_ID;
        std::vector<std::string> statements = parseJsonl(statement_type_id);
        if (statements.empty()) {
            // rr record is telling  us that the program ended here, with empty statements.
            // But actually running the binary shows us the statement we expect. What could be leading to this mismatch?
            std::cerr << "Failed to process benchmark statement" << std::endl;
            return 1;
        }
        char * option4 = "-p";

        // so I can't append a c-like string (char *) to another.
        // This kind of makes sense. What does the addition operation mean?
        // With pointers, instead of concatenating them using the + binary operator, you should instead use pointer offsets.

        // Can I then do seomthing like *(switch4 + offset) = some character?

        // Well, all I know is that I'll most likely have to build it character by character.

        // we know the size of the char* should statement[i].size() + 2, 
        // reserving two characters two enclose the statement within quotation marks.
        //
        // actually we need to save one final space for the null terminator, hence + 3

        int statement_size = statements[0].size() + 3;
        char switch4[ statement_size ];
        switch4[0] = '\"';
        switch4[statement_size - 2] = '\"';
        switch4[statement_size - 1] = '\0'; // null terminator!
        for ( int k = 1; k < statement_size - 2; k++ ) {
            switch4[k] = statements[0][k-1];
        }

        // switch4 += const_cast<char*>(statements[0].c_str());
        // const_cast<char*>("\"" + statements[0] + "\"");
        
        // create a child process
        // FIXME: why fork() here if we're already doing that outside the if-statement chain? I'll comment it out for now.
        // pid_t pid = fork();

        char *const argv[10] = { option0, option1, switch1, option2, switch2, option3, switch3, option4, switch4, nullptr };

        std::cout << "Executing command..." << std::endl;
        /**
        for (int i = 0; i < 9 ; i++ ) {
            std::cout << argv[i] << std::endl;
        }
        **/
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
        int condition;
        char buffer[128];
        do {
            condition = read(pipefd[0],buffer,128);
            // FIXME: read() returns the number of bytes read
            // therefore change the name of condition to a more aptly descriptive name
            if ( condition < 0 ) {
                perror("process.cpp: read(pipefd[0],buffer,128)");
                std::cerr << "Error: " << std::strerror(errno) << " (errno: " << ")\n";
                return 1;
            }
            for ( int i = 0; i < condition; i++ ) {
                output.push_back(buffer[i]);
            }
        } while (condition > 0);
        
        if (output.empty()) {
            std::cerr << "No output was received from the process." << std::endl;
            return 1;
        }

        std::regex end_of_turn("R\\[ Prompt: \\d+\\.\\d+ t/s \\| Generation: \\d+\\.\\d+ t/s \\]");
        if (std::regex_search(output,end_of_turn)) {
            std::cout << "Initiating process termination." << std::endl;
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
