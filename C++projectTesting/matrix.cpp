#include "matrix.hpp"
#include <stdexcept>

void matrix::checkBound(int it, int lower, int upper) {
    if (it < lower || it >= upper) {
        throw std::range_error(
                std::to_string(it) + " not in range ["
                + std::to_string(lower) + ","
                + std::to_string(upper) + ")"
        );
    }
}

std::vector<int> matrix::makeSlice(int start, int end, int step, int upperBound) {
    if (start < 0) {
        start = upperBound + start;
    }
    if (end < 0) {
        end = upperBound + end;
    }
    if (step < 0) {
        int tmp = start;
        start = end;
        end = tmp;
    }
    int i = start;
    std::vector<int> vec;
    while ((step > 0 && i >= start && i <= end) || (step < 0 && i >= end && i <= start)) {
        vec.push_back(i);
        i += step;
    }
    return vec;
}