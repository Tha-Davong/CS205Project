//
// Created by Vithlaithla Long on 7/6/21.
//

#ifndef CS205PROJECT_TEMPLATEUTIL_H
#define CS205PROJECT_TEMPLATEUTIL_H
#include <type_traits>
#include <complex>

#define IF(...) typename std::enable_if< (__VA_ARGS__), bool  >::type = true


template< class T, class U >
constexpr bool is_same() {return std::is_same<T, U>::value;};

template<typename T>
struct is_complex_t : public std::false_type {};

template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template<typename T>
constexpr bool is_complex = is_complex_t<T>::value;

template<typename T >
constexpr bool is_arithmetic_t =  std::is_arithmetic<T>::value;

#endif //CS205PROJECT_TEMPLATEUTIL_H