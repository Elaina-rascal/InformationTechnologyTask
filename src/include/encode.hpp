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
        : _window_size(window_size), _lookahead_buffer_size(lookahead_buffer_size) {}

    std::vector<uint8_t> encode(const std::vector<std::string> &input)
    {
        std::vector<uint8_t> output;

        // 修正拼接逻辑：用@分隔元素，空字符串直接拼接（占一个分隔符位置）
        std::string concatenated;
        for (size_t i = 0; i < input.size(); ++i)
        {
            if (i > 0)
            {
                concatenated += '@'; // 仅在元素间添加分隔符
            }
            concatenated += input[i]; // 空字符串添加空内容
        }
#if DEBUG
        fmt::print("拼接后的字符串: {}\n", concatenated);
#endif

        // LZ77编码（修正范围限制）
        size_t pos = 0;
        const size_t input_len = concatenated.size();
        while (pos < input_len)
        {
            size_t match_offset = 0;
            size_t match_length = 0;

            size_t start_window = (pos >= static_cast<size_t>(_window_size)) ? (pos - _window_size) : 0;
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
                if (length > match_length)
                {
                    match_length = length;
                    match_offset = pos - i;
                }
            }

            // 限制偏移（12位）和长度（4位+3）
            if (match_length >= 3 && match_length <= 18 && match_offset <= 4095)
            {
                output.push_back(static_cast<uint8_t>(match_offset >> 4));
                output.push_back(static_cast<uint8_t>(((match_offset & 0x0F) << 4) | (match_length - 3)));
                pos += match_length;
            }
            else
            {
                output.push_back(0);
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

        // LZ77解码（容错处理）
        while (pos + 1 < input_len)
        {
            uint8_t byte1 = input[pos];
            uint8_t byte2 = input[pos + 1];
            pos += 2;

            if (byte1 == 0)
            {
                concatenated += static_cast<char>(byte2);
            }
            else
            {
                size_t match_offset = (static_cast<size_t>(byte1) << 4) | (byte2 >> 4);
                size_t match_length = (byte2 & 0x0F) + 3;

                // 容错：偏移非法时按单字符处理
                if (match_offset > concatenated.size() || match_length == 0)
                {
                    concatenated += static_cast<char>(byte2);
                    continue;
                }

                size_t start_pos = concatenated.size() - match_offset;
                for (size_t i = 0; i < match_length; ++i)
                {
                    if (start_pos + i >= concatenated.size())
                        break; // 避免越界
                    concatenated += concatenated[start_pos + i];
                }
            }
        }
#if DEBUG
        fmt::print("解码后拼接的字符串: {}\n", concatenated);
#endif
        // 修正分割逻辑：按@分割，连续@对应空字符串
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
        output.push_back(current); // 最后一个元素

        return output;
    }

private:
    int _window_size;
    int _lookahead_buffer_size;
};