#include "encode.hpp"
#include "rapidcsv.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// 辅助函数：将二进制数据写入文件
void write_binary(const std::string &path, const std::vector<uint8_t> &data)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("无法打开二进制文件: " + path);
    }
    ofs.write(reinterpret_cast<const char *>(data.data()), data.size());
}

int main()
{
    const std::string csv_path = "/pytorch/data/task1.csv";
    const std::string json_config_path = "/pytorch/data/task1_config.json";
    const std::string bin_dir = "/pytorch/data/bin/"; // 二进制文件存放目录

    // 创建二进制目录（如果不存在）
    std::filesystem::create_directories(bin_dir);

    rapidcsv::Document doc(csv_path);
    const int window_size = 200;
    const int lookahead_buffer_size = 15;
    const int crc_polynomial = 0x07;
    const size_t crc_block_size = 2048;

    CRCCoder_t CRC(crc_polynomial, crc_block_size);
    LZEncoder_t LZ(window_size, lookahead_buffer_size);

    // JSON仅保存配置和二进制文件索引
    nlohmann::json json_config;
    json_config["window_size"] = window_size;
    json_config["lookahead_buffer_size"] = lookahead_buffer_size;
    json_config["crc_polynomial"] = crc_polynomial;
    json_config["crc_block_size"] = crc_block_size;
    json_config["columns"] = nlohmann::json::object(); // 存储列名到二进制文件的映射

    // 处理前3列
    for (int column_id = 0; column_id < doc.GetColumnCount(); ++column_id)
    {
        const std::string column_name = doc.GetColumnName(column_id);
        std::vector<std::string> column_data = doc.GetColumn<std::string>(column_id);

        // 编码流程
        std::vector<uint8_t> lz_encoded = LZ.encode(column_data);
        std::vector<uint8_t> crc_encoded = CRC.encode(lz_encoded);

        // 保存二进制数据到单独文件
        const std::string bin_filename = fmt::format("col_{}.bin", column_id);
        const std::string bin_path = bin_dir + bin_filename;
        write_binary(bin_path, crc_encoded);

        // 在JSON中记录映射关系
        json_config["columns"][column_name] = bin_filename;
    }

    // 保存配置JSON
    std::ofstream json_file(json_config_path);
    if (json_file.is_open())
    {
        json_file << json_config.dump(4); // 带缩进的可读性配置
        json_file.close();
        fmt::print("成功保存配置和二进制文件\n");
    }
    else
    {
        throw std::runtime_error("无法打开配置JSON文件");
    }

    return 0;
}