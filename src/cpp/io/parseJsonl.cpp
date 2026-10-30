#include "parseJsonl.hpp"


/**
 * Parse the dataset, and return a list of statements
 * which will be fed to the prover model
 */
std::vector<std::string> parseJsonl(bool formal) {

    std::string path = "../benchmarks/miniF2F_v2/datasets/miniF2F_v2s.jsonl";
    std::ifstream ifs(path,std::ios::in);

    std::vector<std::string> statements;

    std::string line;
    while(std::getline(ifs,line)) {

        std::string name;
        std::string split;
        std::string informal_statement;
        std::string formal_statement;
        std::string header;
        std::string informal_proof;
        std::string formal_proof;

        int field_idx = 0;

        const int NAME = 0;
        const int SPLIT = 1;
        const int INFORMAL_STATEMENT = 2;
        const int FORMAL_STATEMENT = 3;
        const int HEADER = 4;
        const int INFORMAL_PROOF = 5;
        const int FORMAL_PROOF = 6;
        
        bool quote = false; 
        int quoteCount = 0;
        int commaCount = 0;


        for (int i = 1 ; i < line.size()-1; i ++) { // skip the first and last characters of the line, which will allways be '{' and '}' respectively.
            char c = line[i];
            if (c == '"') {
                quote == true ? (void)quoteCount++ : (void)0 ; // end of the quote, increment quoteCount
                quote = !quote;
                continue;
            }
            if (c == ',' && quote == false) {
                field_idx++;
                quoteCount = 0;
                continue;
            }

            if (!(quoteCount == 1 && quote == true)) {
                continue; // don't store these characters
            }

            // switch the field based on flag
            switch (field_idx) {
                case NAME:
                    name.push_back(c);
                    break;
                case SPLIT:
                    split.push_back(c);
                    break;
                case INFORMAL_STATEMENT:
                    informal_statement.push_back(c);
                    break;
                case FORMAL_STATEMENT:
                    formal_statement.push_back(c);
                    break;
                case HEADER:
                    header.push_back(c);
                    break;
                case INFORMAL_PROOF:
                    informal_proof.push_back(c);
                    break;
                case FORMAL_PROOF:
                    formal_proof.push_back(c);
                    break; 
            }

        } // for
        formal ? statements.push_back(formal_statement) : statements.push_back(informal_statement);
    } // while
    return statements;
} // parsejsonl
