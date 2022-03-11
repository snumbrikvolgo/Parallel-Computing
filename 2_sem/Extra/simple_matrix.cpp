#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <exception>
#include <iostream>
#include <omp.h>

class Random {
public:
    static float getFloat(float const min = 0.0f, float const max = 1.0f) {
        return m_distribution(m_gen) * (max - min) + min;
    }

private:
    static std::mt19937 m_gen;
    static std::uniform_real_distribution<float> m_distribution;
};

std::mt19937 Random::m_gen = std::mt19937(std::random_device()());
std::uniform_real_distribution<float> Random::m_distribution = std::uniform_real_distribution<float>(0.0f, 1.0f);

class Timer {
public:
    static void start() {
        m_start = std::chrono::system_clock::now();
    }

    static auto stop() {
        m_stop = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_stop - m_start);
        return duration.count();
    }

private:
    static std::chrono::time_point<std::chrono::system_clock> m_start;
    static std::chrono::time_point<std::chrono::system_clock> m_stop;
};

std::chrono::time_point<std::chrono::system_clock> Timer::m_start = std::chrono::system_clock::now();
std::chrono::time_point<std::chrono::system_clock> Timer::m_stop = std::chrono::system_clock::now();

class Matrix {
public:
    static Matrix create(size_t const nRows, size_t const nColumns, bool const init = true) {
        auto result = Matrix(nRows, nColumns);
        if (!init) {
            return result;
        }

        auto size = result.getSize();
        for (size_t i = 0; i < size; ++i) {
            result[i] = Random::getFloat();
        }
        return result;
    }

    size_t getSize() const {
        return m_nColumns * m_nRows;
    }

    float &operator[](size_t const idx) {
        return m_data[idx];
    }

    float const &operator[](size_t const idx) const {
        return m_data[idx];
    }

    size_t m_nRows = 0;
    size_t m_nColumns = 0;

private:
    Matrix(size_t const nRows, size_t const nColumns) :
            m_nRows{nRows},
            m_nColumns{nColumns},
            m_data{std::make_unique<float[]>(nRows * nColumns)} {}

    std::unique_ptr<float[]> m_data;

public:
    friend Matrix matmulBlocks(Matrix &a, Matrix &b, int chunkSize);
};

float matDiff(Matrix const &a, Matrix const &b) {
    if (a.m_nRows != b.m_nRows || a.m_nColumns != b.m_nColumns) {
        throw std::invalid_argument("dimensions not compatible");
    }

    auto const size = a.getSize();

    float sum_delta = 0, sum = 0;
    for (auto i = 0u; i < size; ++i) {
        sum_delta += std::abs(a[i] - b[i]);
        sum += std::abs(a[i]) + std::abs(b[i]);
    }
    return sum_delta / sum;
}

// dumb way to multiply matrices O(n^3)
Matrix matmulSimple(Matrix const &a, Matrix const &b, int chunkSize) {
    if (a.m_nColumns != b.m_nRows) {
        throw std::invalid_argument("dimensions not compatible");
    }

    auto const resNRows = a.m_nRows;
    auto const resNColumns = b.m_nColumns;

    auto res = Matrix::create(resNRows, resNColumns, false);

    auto const multLen = a.m_nColumns;
#pragma omp parallel for default(shared) schedule(dynamic, chunkSize)
    for (auto r = 0u; r < resNRows; ++r) {
        for (auto c = 0u; c < resNColumns; ++c) {
            res[r * resNColumns + c] = 0;
            for (auto i = 0u; i < multLen; ++i) {
                res[r * resNColumns + c] += a[r * multLen + i] * b[i * multLen + c];
            }
        }
    }
    return res;
}

// less dumb way to multiply matricies
Matrix matmulByRows(Matrix const &a, Matrix const &b, int chunkSize) {
    if (a.m_nColumns != b.m_nRows) {
        throw std::invalid_argument("dimensions not compatible");
    }

    auto const resNRows = a.m_nRows;
    auto const resNColumns = b.m_nColumns;

    auto res = Matrix::create(resNRows, resNColumns, false);

    auto const multLen = a.m_nColumns;
#pragma omp parallel for default(shared) schedule(dynamic, chunkSize)
    for (auto r = 0u; r < resNRows; ++r) {
        for (auto c = 0u; c < resNColumns; ++c) {
            res[r * resNColumns + c] = 0;
        }
        for (auto i = 0u; i < multLen; ++i) {
            for (auto c = 0u; c < resNColumns; ++c) {
                res[r * resNColumns + c] += a[r * multLen + i] * b[i * resNColumns + c];
            }
        }
    }

    return res;
}

// cache-friendly way to multiply matricies
Matrix matmulBlocks(Matrix &a, Matrix &b, int chunkSize) {
    if (a.m_nColumns != b.m_nRows) {
        throw std::invalid_argument("dimensions not compatible");
    }

    auto const resNRows = a.m_nRows;
    auto const resNColumns = b.m_nColumns;
    auto const multLen = a.m_nColumns;

    if (!((resNRows % 4 == 0) && (resNColumns % 4 == 0) && (multLen % 4 == 0))) {
        throw std::invalid_argument("can't divide matricies into blocks");
    }

    auto const nBlocksRow = resNRows / 4;
    auto const nBlocksColumn = resNColumns / 4;
    auto const blocksMultLen = multLen / 4;

    struct float4_t {
        float data[4];

        float &operator[](size_t idx) {
            return data[idx];
        }

        float const &operator[](size_t idx) const {
            return data[idx];
        }

        float4_t operator*(float v) const {
            float4_t res;
            for (auto i = 0u; i < 4; ++i) {
                res[i] = v * data[i];
            }
            return res;
        }

        float4_t operator+(float4_t const &rhs) const {
            float4_t res;
            for (auto i = 0u; i < 4; ++i) {
                res[i] = rhs[i] + data[i];
            }
            return res;
        }
    };

    auto const blockReorderLhs = [&nBlocksRow, &nBlocksColumn, &blocksMultLen, &chunkSize](float4_t *in,
                                                                                           float4_t *out) {
        for (auto br = 0u; br < nBlocksRow; ++br) {
            for (auto bc = 0u; bc < blocksMultLen; ++bc) {
                float4_t *offs = in + bc + 4 * blocksMultLen * br;
                for (auto i = 0u; i < 4; ++i)
                    *(out++) = offs[i * blocksMultLen];
            }
        }
    };

    auto const blockReorderRhs = [&nBlocksRow, &nBlocksColumn, &blocksMultLen, &chunkSize](float4_t *in,
                                                                                           float4_t *out) {
        for (auto bc = 0u; bc < nBlocksColumn; ++bc) {
            for (auto br = 0u; br < blocksMultLen; ++br) {
                float4_t *offs = in + bc + 4 * nBlocksColumn * br;
                for (auto i = 0u; i < 4; ++i)
                    *(out++) = offs[i * nBlocksColumn];
            }
        }
    };

    auto const blockReorderRes = [&nBlocksRow, &nBlocksColumn, &blocksMultLen, &chunkSize](float4_t *in,
                                                                                           float4_t *out) {
        for (auto bc = 0u; bc < nBlocksColumn; ++bc) {
            for (auto br = 0u; br < nBlocksRow; ++br) {
                float4_t *offs = out + bc + 4 * nBlocksColumn * br;
                for (auto i = 0u; i < 4; ++i)
                    offs[i * nBlocksColumn] = *(in++);
            }
        }
    };

    auto tmpLhs = Matrix::create(a.m_nColumns, a.m_nRows, false);
    blockReorderLhs(reinterpret_cast<float4_t *>(a.m_data.get()), reinterpret_cast<float4_t *>(tmpLhs.m_data.get()));

    auto tmpRhs = Matrix::create(b.m_nColumns, b.m_nRows, false);
    blockReorderRhs(reinterpret_cast<float4_t *>(b.m_data.get()), reinterpret_cast<float4_t *>(tmpRhs.m_data.get()));

    auto tmpRes = Matrix::create(resNRows, resNColumns, false);
#pragma omp parallel for default(shared) schedule(dynamic, chunkSize)
    for (auto br = 0u; br < nBlocksRow; ++br) {
        for (auto bc = 0u; bc < nBlocksColumn; ++bc) {
            auto r_ptr = reinterpret_cast<float4_t *>(tmpLhs.m_data.get() + 16 * blocksMultLen * br);
            auto c_ptr = reinterpret_cast<float4_t *>(tmpRhs.m_data.get() + 16 * blocksMultLen * bc);
            auto res_ptr = reinterpret_cast<float4_t *>(tmpRes.m_data.get() + 16 * (br + bc * nBlocksRow));

            float4_t tmp[4], tmp2[4];
            for (auto i = 0u; i < 4; ++i) {
                tmp[i] = {0, 0, 0, 0};
            }

            for (auto i = 0u; i < 4 * blocksMultLen; i += 4) {
                for (auto j = 0u; j < 4; ++j) {
                    tmp2[j] = c_ptr[i + j];
                }
                for (auto j = 0u; j < 4; ++j) {
                    for (auto k = 0u; k < 4; ++k) {
                        tmp[j] = tmp[j] + tmp2[k] * r_ptr[i + j][k];
                    }
                }
            }

            for (auto i = 0u; i < 4; ++i) {
                res_ptr[i] = tmp[i];
            }
        }
    }

    auto res = Matrix::create(resNRows, resNColumns, false);
    blockReorderRes(reinterpret_cast<float4_t *>(tmpRes.m_data.get()), reinterpret_cast<float4_t *>(res.m_data.get()));
    return res;
}

int main(int argc, char *argv[]) {

    // set dynamic to false just in case if implementation will allocate different number of threads
    // on every other run
    omp_set_dynamic(static_cast<int>(false));

    // output maximum available number of threads
    auto const maxThreads = omp_get_max_threads();
//    auto const maxThreads = 1;
    std::printf("Max threads %d\n", maxThreads);

    auto const maxMatDimSize = 1024;
    // iterate chunk sizes for `omp schedule(dynamic, chunk size)`
    for (unsigned i = 1024; i <= maxMatDimSize; i *= 2) {

        // create two square matrices with random floats
        auto a = Matrix::create(i, i);
        auto b = Matrix::create(i, i);

        auto timeSimple = 0;
        Timer::start();
        auto resSimple = matmulSimple(a, b, 1);
        timeSimple = Timer::stop();

        auto timeByRows = 0;
        Timer::start();
        auto resByRows = matmulByRows(a, b, 1);
        timeByRows = Timer::stop();

        auto timeBlocks = 0;
        Timer::start();
        auto resBlocks = matmulBlocks(a, b, 1);
        timeBlocks = Timer::stop();

        std::printf("Matrix size %d\n"
                    "\tSimple %d ms.\n"
                    "\tByRows %d ms.\n"
                    "\tByBlocks %d ms.\n",
                    i, timeSimple, timeByRows, timeBlocks);
    }

    // Example output from my machine:
    //    Max threads 8
    //-------------------------------- P A R A L L E L----------------------------------------------------------------
    //Matrix size 4
    //	Simple 1 ms. \\ 1
    //	ByRows 1 ms. \\ 0
    //	ByBlocks 0 ms. \\ 0
    //Matrix size 8
    //	Simple 0 ms. \\ 0 ms.
    //	ByRows 1 ms. \\ 1
    //	ByBlocks 0 ms. \\ 0
    //Matrix size 16
    //	Simple 2 ms. \\ 0
    //	ByRows 1 ms. \\ 0
    //	ByBlocks 1 ms. \\ 0
    //Matrix size 32
    //	Simple 1 ms. \\ 0
    //	ByRows 1 ms. \\ 0
    //	ByBlocks 0 ms. \\ 0
    //Matrix size 64
    //	Simple 3 ms. \\ 2
    //	ByRows 5 ms. \\ 2
    //	ByBlocks 1 ms. \\ 0
    //Matrix size 128
    //	Simple 31 ms. \\ 17
    //	ByRows 28 ms. \\ 18
    //	ByBlocks 10 ms. \\ 6
    //Matrix size 256
    //	Simple 104 ms. \\ 112
    //	ByRows 100 ms. \\ 107
    //	ByBlocks 27 ms. \\ 31
    //Matrix size 512
    //	Simple 780 ms. \\ 725
    //	ByRows 744 ms. \\ 726
    //	ByBlocks 224 ms. \\ 203
    //Matrix size 1024
    //	Simple 7031 ms. \\ 7117
    //	ByRows 6776 ms. \\ 6297
    //	ByBlocks 1712 ms. \\ 1909

    // 1) obviously chunk size equal to whole job (1024) will effectively make this program single-threaded.
    //    this is the least efficient run.
    // 2) chunk sizes 4-64 is middle-ground. They are reasonably effective for each multiplication algorithm.
    // 3) the most effective chunk size is 1 due to the fact that multiplying one row is big-enough task by itself.

    return 0;
}
