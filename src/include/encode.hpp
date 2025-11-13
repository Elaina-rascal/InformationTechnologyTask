#include <cstdint>
#include <string>
#include <vector>
#define DEBUG true
#if DEBUG
#include <fmt/core.h>
#include <fmt/ranges.h>
#endif

class LZEncoder_t
{
public:
    LZEncoder_t(int window_size, int lookahead_buffer_size)
    {
        // 11位偏移最大支持2047，窗口大小不超过此值
        _window_size = std::min(window_size, 2048);
        // 最大匹配长度仍为18（4位存储+3）
        _lookahead_buffer_size = std::min(lookahead_buffer_size, 18);
    }

    std::vector<uint8_t> encode(const std::vector<std::string> &input)
    {
        std::vector<uint8_t> output;

        // 拼接逻辑保持不变
        std::string concatenated;
        for (size_t i = 0; i < input.size(); ++i)
        {
            if (i > 0)
            {
                concatenated += '@';
            }
            concatenated += input[i];
        }
#if DEBUG
        fmt::print("拼接后的字符串: {}\n", concatenated);
#endif

        // LZ77编码（使用标志位区分）
        size_t pos = 0;
        const size_t input_len = concatenated.size();
        while (pos < input_len)
        {
            size_t match_offset = 0;
            size_t match_length = 0;
            size_t start_window = (pos >= _window_size) ? (pos - _window_size) : 0;
            size_t end_window = pos;

            for (size_t i = start_window; i < end_window; ++i)
            {
                size_t length = 0;
                while (length < static_cast<size_t>(_lookahead_buffer_size) &&
                       pos + length < input_len &&
                       concatenated[i + length] == concatenated[pos + length])
                {
                    ++length;
                }
                if (length > match_length || (length == match_length && (pos - i) < match_offset))
                {
                    match_length = length;
                    match_offset = pos - i;
                }
            }

            // 11位偏移（0~2047）+ 标志位区分
            if (match_length >= _min_match_length && match_length <= _max_match_length && match_offset <= _max_offset)
            {
                // 第一个字节：最高位0（标志位）+ 偏移高3位
                uint8_t byte1 = static_cast<uint8_t>((match_offset >> 8) & 0x7F); // 0x7F确保最高位为0
                // 第二个字节：偏移低8位
                uint8_t byte2 = static_cast<uint8_t>(match_offset & 0xFF);
                // 重新打包：byte1保持标志位，byte2高4位存长度
                byte2 = static_cast<uint8_t>(((match_length - _min_match_length) << 4) | (byte2 & 0x0F));
                output.push_back(byte1);
                output.push_back(byte2);
                pos += match_length;
            }
            else
            {
                // 单字符标志：最高位1（0x80），低7位无意义
                output.push_back(0x80);
                output.push_back(static_cast<uint8_t>(concatenated[pos]));
                ++pos;
            }
        }

        return output;
    }

    std::vector<std::string> decode(const std::vector<uint8_t> &input)
    {
        std::string concatenated;
        size_t pos = 0;
        const size_t input_len = input.size();

        // LZ77解码（基于标志位判断）
        while (pos + 1 < input_len)
        {
            uint8_t byte1 = input[pos];
            uint8_t byte2 = input[pos + 1];
            pos += 2;

            // 最高位为1：单字符
            if ((byte1 & 0x80) != 0)
            {
                concatenated += static_cast<char>(byte2);
            }
            // 最高位为0：偏移+长度编码
            else
            {
                // 解析11位偏移：byte1低7位 + byte2低4位
                size_t match_offset = static_cast<size_t>((byte1 & 0x7F) << 8) | (byte2 & 0x0F);
                // 解析4位长度（+3还原）
                size_t match_length = static_cast<size_t>((byte2 >> 4) & 0x0F) + _min_match_length;

                // 容错处理：无效偏移或长度时按单字符处理
                if (match_offset == 0 || match_offset > concatenated.size() ||
                    match_length < _min_match_length || match_length > _max_match_length)
                {
                    concatenated += static_cast<char>(byte2);
                    continue;
                }

                // 正确复制匹配内容（支持自引用）
                size_t start_pos = concatenated.size() - match_offset;
                for (size_t i = 0; i < match_length; ++i)
                {
                    concatenated += concatenated[start_pos + i];
                }
            }
        }
#if DEBUG
        fmt::print("解码后拼接的字符串: {}\n", concatenated);
#endif

        // 分割逻辑保持不变
        std::vector<std::string> output;
        std::string current;
        for (char c : concatenated)
        {
            if (c == '@')
            {
                output.push_back(current);
                current.clear();
            }
            else
            {
                current += c;
            }
        }
        output.push_back(current);

        return output;
    }

private:
    uint32_t _window_size = 0;
    uint32_t _lookahead_buffer_size = 0;
    uint8_t const _min_match_length = 3;
    uint32_t const _max_offset = 2047; // 11位最大偏移（2^11 - 1）
    uint32_t const _max_match_length = 18;
};

// 因为是字符串,数据肯定小于128 可以直接用最高位做标志位，用最高位表示是单字符还是偏移+长度