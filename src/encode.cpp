#include "encode.hpp"
#include "rapidcsv.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
int main()
{
    auto csv_path = "/pytorch/data/task1.csv";
    auto json_path = "/pytorch/data/task1.json";
    // io::CSVReader<0> in("data_44cols.csv");
    rapidcsv::Document doc(csv_path);
    int window_size = 200;
    int lookahead_buffer_size = 15;
    // std::vector<std::string> row = doc.GetRow<std::string>(1);
    std::vector<std::string> row = doc.GetColumn<std::string>(1);
    // fmt::print("第零行{}: \n", row);
    // row = doc.GetRow<std::string>(1);
    // fmt::print("第一行{}: \n", row);

    auto encode = LZEncoder_t(window_size, lookahead_buffer_size);
    auto output = encode.encode(row);
    // fmt::print("压缩后字节流{}: \n", output);
    auto decode_output = encode.decode(output);
    auto hamming = CRCCoder_t();
    auto encoded_hamming = hamming.encode(output);
    auto decoded_hamming = hamming.decode(encoded_hamming);
    fmt::print("解压后字符串向量{}: \n", decode_output);

    return 0;
}