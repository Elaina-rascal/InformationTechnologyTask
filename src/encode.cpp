#include "encode.hpp"
#include "rapidcsv.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string>
#include <vector>
int main()
{
    auto csv_path = "/pytorch/data/task1.csv";
    // io::CSVReader<0> in("data_44cols.csv");
    rapidcsv::Document doc(csv_path);

    std::vector<std::string> row = doc.GetRow<std::string>(0);
    fmt::print("第零行{}: \n", row);
    row = doc.GetRow<std::string>(1);
    fmt::print("第一行{}: \n", row);

    auto encode = LZEncoder_t(20, 8);
    auto output = encode.encode(row);
    fmt::print("压缩后字节流{}: \n", output);
    auto decode_output = encode.decode(output);
    fmt::print("解压后字符串向量{}: \n", decode_output);
    return 0;
}