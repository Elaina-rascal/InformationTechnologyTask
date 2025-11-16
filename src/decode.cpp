#include "encode.hpp"
#include "rapidcsv.h"
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 辅助函数：从文件读取二进制数据
std::vector<uint8_t> LoadBinaryFile(const std::string &path)
{
    if (!fs::exists(path))
    {
        throw std::runtime_error("二进制文件不存在: " + path);
    }

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
    {
        throw std::runtime_error("无法打开二进制文件: " + path);
    }

    // 获取文件大小并预分配缓冲区
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!ifs.read(reinterpret_cast<char *>(data.data()), size))
    {
        throw std::runtime_error("读取二进制文件失败: " + path);
    }

    return data;
}

int main()
{

    // 路径配置
    const std::string csv_path = "/pytorch/data/task1.csv";
    const std::string json_config_path = "/pytorch/data/task1_config.json";
    const std::string bin_dir = "/pytorch/data/bin/";

    // 1. 读取配置文件
    std::ifstream config_file(json_config_path);
    if (!config_file.is_open())
    {
        throw std::runtime_error("无法打开配置文件: " + json_config_path);
    }

    json config;
    config_file >> config;
    config_file.close();

    // 解析配置参数
    const int window_size = config["window_size"];
    const int lookahead_buffer_size = config["lookahead_buffer_size"];
    const int crc_polynomial = config["crc_polynomial"];
    const size_t crc_block_size = config["crc_block_size"];

    // 初始化解码器（假设编码器类同时包含解码方法，或有对应的解码器类）
    CRCCoder_t CRC(crc_polynomial, crc_block_size);
    LZEncoder_t LZ(window_size, lookahead_buffer_size); // 假设包含decode方法

    // 2. 读取原始CSV文件
    rapidcsv::Document doc(csv_path);

    // 3. 处理所有列
    size_t total_mismatches = 0;
    size_t total_elements = 0;

    for (const auto &[column_name, bin_filename] : config["columns"].items())
    {
        fmt::print("\n处理列: {}\n", column_name);

        // 3.1 读取二进制编码文件
        const std::string bin_path = bin_dir + bin_filename.get<std::string>();
        std::vector<uint8_t> crc_encoded = LoadBinaryFile(bin_path);

        // 3.2 解码流程（先CRC解码，再LZ解码）
        std::vector<uint8_t> lz_encoded = CRC.Decode(crc_encoded);
        std::vector<std::string> decoded_data = LZ.Decode(lz_encoded);

        // 3.3 获取原始数据
        std::vector<std::string> original_data = doc.GetColumn<std::string>(column_name);

        // 3.4 比较原始数据与解码数据
        uint32_t column_mismatches = 0;

        if (decoded_data.size() != original_data.size())
        {
            fmt::print("警告: 数据长度不匹配！原始: {} 解码: {}\n",
                       original_data.size(), decoded_data.size());
            // 以较短的长度进行比较
            uint32_t compare_len = std::min(original_data.size(), decoded_data.size());
            total_elements += compare_len;

            for (auto i = 0; i < compare_len; ++i)
            {
                if (decoded_data[i] != original_data[i])
                {
                    column_mismatches++;
                    if (column_mismatches <= 5)
                    { // 只显示前5个不匹配项
                        fmt::print("位置 {} 不匹配 - 原始: '{}' 解码: '{}'\n",
                                   i, original_data[i], decoded_data[i]);
                    }
                }
            }
        }
        // 统计结果
        total_mismatches += column_mismatches;
        fmt::print("列 {} 不匹配率: {:.2f}%\n",
                   column_name, (column_mismatches * 100.0) / original_data.size());
    }

    // 4. 输出总体比较结果
    fmt::print("总元素数: {}\n", total_elements);
    fmt::print("总不匹配数: {}\n", total_mismatches);

    return 0;
}