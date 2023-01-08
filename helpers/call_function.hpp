#pragma once

#include <array>
#include <iostream>
#include <tuple>
#include <utility>

namespace helpers
{
    template <typename F, std::size_t N, typename T, std::size_t... Indices>
    auto call_function_helper(F func, const std::array<T, N> &v, std::index_sequence<Indices...>)
    {
        return func(std::get<Indices>(v)...);
    }

    template <typename F, std::size_t N, typename T>
    auto call_function(F func, const std::array<T, N> &v)
    {
        return call_function_helper<N>(func, v, std::make_index_sequence<N>());
    }
}
