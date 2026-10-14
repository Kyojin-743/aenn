#pragma once

#include <iostream>
#include <sstream>
#include <cstdlib>

class AssertStream {
public:
    AssertStream(bool condition, const char* cond_str, const char* file, int line) 
        : condition_failed_(!condition) {
        if (condition_failed_) {
            stream_ << "\n[  FAILED  ] " << file << ":" << line 
                    << "\n  Assertion: " << cond_str << "\n  Message:   ";
        }
    }

    // Stream operator template to handle any streamable type
    template <typename T>
    AssertStream& operator<<(const T& val) {
        if (condition_failed_) {
            stream_ << val;
        }
        return *this;
    }

    // Destructor executes when the stream expression terminates
    ~AssertStream() {
        if (condition_failed_) {
            stream_ << "\n" << std::endl;
            std::cerr << stream_.str();
            std::exit(EXIT_FAILURE); // Or throw std::runtime_error(stream_.str());
        }
    }

private:
    bool condition_failed_;
    std::ostringstream stream_;
};


// GoogleTest-like ASSERT macro
#define ASSERT(condition) AssertStream((condition), #condition, __FILE__, __LINE__)