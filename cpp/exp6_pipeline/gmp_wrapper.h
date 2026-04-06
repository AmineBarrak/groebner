/*
 * gmp_wrapper.h — Lightweight C++ wrapper around GMP's C API (mpz_t)
 *
 * Provides an mpz_class-compatible interface when libgmpxx is unavailable.
 * Used on HPC systems where GMP was built without --enable-cxx.
 */
#pragma once

#ifdef USE_GMP_C_API

#include <gmp.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <utility>

class mpz_class {
    mpz_t val_;
public:
    mpz_class()                    { mpz_init(val_); }
    mpz_class(long v)             { mpz_init_set_si(val_, v); }
    mpz_class(unsigned long v)    { mpz_init_set_ui(val_, v); }
    mpz_class(int v)              { mpz_init_set_si(val_, (long)v); }
    mpz_class(long long v)        { mpz_init(val_); mpz_set_si(val_, (long)v); }
    mpz_class(const char* s)      { mpz_init_set_str(val_, s, 10); }
    mpz_class(const std::string& s) { mpz_init_set_str(val_, s.c_str(), 10); }
    mpz_class(const mpz_class& o) { mpz_init_set(val_, o.val_); }
    mpz_class(mpz_class&& o) noexcept { mpz_init(val_); mpz_swap(val_, o.val_); }
    ~mpz_class()                  { mpz_clear(val_); }

    mpz_class& operator=(const mpz_class& o) {
        if (this != &o) mpz_set(val_, o.val_);
        return *this;
    }
    mpz_class& operator=(mpz_class&& o) noexcept {
        mpz_swap(val_, o.val_);
        return *this;
    }
    mpz_class& operator=(long v) { mpz_set_si(val_, v); return *this; }
    mpz_class& operator=(int v)  { mpz_set_si(val_, (long)v); return *this; }

    // Access raw mpz_t
    mpz_ptr   get_mpz_t()       { return val_; }
    mpz_srcptr get_mpz_t() const { return val_; }

    // Arithmetic operators
    mpz_class operator+(const mpz_class& o) const {
        mpz_class r; mpz_add(r.val_, val_, o.val_); return r;
    }
    mpz_class operator-(const mpz_class& o) const {
        mpz_class r; mpz_sub(r.val_, val_, o.val_); return r;
    }
    mpz_class operator*(const mpz_class& o) const {
        mpz_class r; mpz_mul(r.val_, val_, o.val_); return r;
    }
    mpz_class operator/(const mpz_class& o) const {
        mpz_class r; mpz_tdiv_q(r.val_, val_, o.val_); return r;
    }
    mpz_class operator-() const {
        mpz_class r; mpz_neg(r.val_, val_); return r;
    }

    mpz_class& operator+=(const mpz_class& o) { mpz_add(val_, val_, o.val_); return *this; }
    mpz_class& operator-=(const mpz_class& o) { mpz_sub(val_, val_, o.val_); return *this; }
    mpz_class& operator*=(const mpz_class& o) { mpz_mul(val_, val_, o.val_); return *this; }

    // Comparison
    bool operator==(const mpz_class& o) const { return mpz_cmp(val_, o.val_) == 0; }
    bool operator!=(const mpz_class& o) const { return mpz_cmp(val_, o.val_) != 0; }
    bool operator<(const mpz_class& o)  const { return mpz_cmp(val_, o.val_) < 0; }
    bool operator>(const mpz_class& o)  const { return mpz_cmp(val_, o.val_) > 0; }
    bool operator<=(const mpz_class& o) const { return mpz_cmp(val_, o.val_) <= 0; }
    bool operator>=(const mpz_class& o) const { return mpz_cmp(val_, o.val_) >= 0; }

    bool operator==(long v) const { return mpz_cmp_si(val_, v) == 0; }
    bool operator!=(long v) const { return mpz_cmp_si(val_, v) != 0; }
    bool operator>(long v)  const { return mpz_cmp_si(val_, v) > 0; }
    bool operator<(long v)  const { return mpz_cmp_si(val_, v) < 0; }
    bool operator==(int v)  const { return mpz_cmp_si(val_, (long)v) == 0; }
    bool operator!=(int v)  const { return mpz_cmp_si(val_, (long)v) != 0; }
    bool operator>(int v)   const { return mpz_cmp_si(val_, (long)v) > 0; }
    bool operator<(int v)   const { return mpz_cmp_si(val_, (long)v) < 0; }

    // Conversion
    double get_d()   const { return mpz_get_d(val_); }
    long   get_si()  const { return mpz_get_si(val_); }

    std::string get_str(int base = 10) const {
        char* s = mpz_get_str(nullptr, base, val_);
        std::string result(s);
        free(s);
        return result;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const mpz_class& z) {
        os << z.get_str();
        return os;
    }
};

// Free functions matching gmpxx interface
inline mpz_class abs(const mpz_class& x) {
    mpz_class r;
    mpz_abs(r.get_mpz_t(), x.get_mpz_t());
    return r;
}

#else
// Use the real gmpxx
#include <gmpxx.h>
#endif
