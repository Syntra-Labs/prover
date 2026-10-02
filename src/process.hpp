#ifndef PROCESS_H
#define PROCESS_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <regex>
#include <vector>
#include "parseJsonl.hpp"

extern "C" {
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
}

void process_statement();
int main();

#endif // PROCESS_H
