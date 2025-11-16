#include <cstdint>
#include <string>
#include <vector>
#define DEBUG false
#if DEBUG
#include <fmt/core.h>
#include <fmt/ranges.h>
#endif

class LZEncoder_t
{
public:
    LZEncoder_t(int window_size, int lookahead_buffer_size)
    {
        // 11位偏移最大支持2047（2^11-1），窗口大小不超过此值
        _window_size = std::min(window_size, 2048);
        // 最大匹配长度18（4位存储+3，0~15对应3~18）
        _lookahead_buffer_size = std::min(lookahead_buffer_size, 18);
    }

    std::vector<uint8_t> Encode(const std::vector<std::string> &input)
    {
        std::vector<uint8_t> output;

        // 拼接输入字符串，用@分隔
        std::string concatenated;
        for (size_t i = 0; i < input.size(); ++i)
        {
            if (i > 0)
                concatenated += '@';
            concatenated += input[i];
        }
#if DEBUG
        fmt::print("拼接后的字符串: {}\n", concatenated);
#endif

        size_t pos = 0;
        const size_t input_len = concatenated.size();
        while (pos < input_len)
        {
            uint16_t match_offset = 0; // 11位偏移（0~2047）
            uint8_t match_length = 0;  // 实际长度（3~18）
            size_t start_window = (pos >= _window_size) ? (pos - _window_size) : 0;
            size_t end_window = pos;

            // 查找窗口内最长匹配
            for (size_t i = start_window; i < end_window; ++i)
            {
                uint8_t length = 0;
                // 限制匹配长度不超过前瞻缓冲区和输入边界
                while (length < _lookahead_buffer_size &&
                       pos + length < input_len &&
                       concatenated[i + length] == concatenated[pos + length])
                {
                    ++length;
                }
                // 优先选更长的匹配，长度相同时选偏移更小的
                if (length > match_length || (length == match_length && (pos - i) < match_offset))
                {
                    match_length = length;
                    match_offset = pos - i; // 计算偏移（当前位置 - 匹配起始位置）
                }
            }

            // 若找到有效匹配（长度≥3，偏移≤2047），编码为偏移+长度
            if (match_length >= _min_match_length && match_length <= _max_match_length && match_offset <= _max_offset)
            {
                uint8_t len_encoded = match_length - _min_match_length; // 编码长度（0~15）

                // 第一字节：标志位0（最高位） + 偏移高7位（bit10~bit4）
                uint8_t byte1 = (match_offset >> 4) & 0x7F; // 0x7F确保最高位为0

                // 第二字节：偏移低4位（bit3~bit0） + 编码长度（bit7~bit4）
                uint8_t byte2 = ((match_offset & 0x0F) << 4) | (len_encoded & 0x0F);

                output.push_back(byte1);
                output.push_back(byte2);
                pos += match_length;
            }
            // 单字符编码（两字节，标志位1）
            else
            {
                // 第一字节：标志位1（最高位），低7位填充0（保留位）
                uint8_t byte1 = 0x80; // 0x80 = 1000 0000（标志位置1）
                // 第二字节：存储字符的ASCII值
                uint8_t byte2 = static_cast<uint8_t>(concatenated[pos]);

                output.push_back(byte1);
                output.push_back(byte2);
                ++pos;
            }
        }

        return output;
    }

    std::vector<std::string> Decode(const std::vector<uint8_t> &input)
    {
        std::string concatenated;
        size_t pos = 0;
        const size_t input_len = input.size();

        // 按两字节一组解码
        while (pos + 1 < input_len)
        {
            uint8_t byte1 = input[pos];
            uint8_t byte2 = input[pos + 1];
            pos += 2;

            // 标志位为1：单字符（第二字节存储字符）
            if (byte1 & 0x80)
            {
                concatenated += static_cast<char>(byte2);
            }
            // 标志位为0：偏移+长度编码
            else
            {
                // 解析11位偏移：byte1低7位（偏移高7位） + byte2高4位（偏移低4位）
                uint16_t match_offset = ((byte1 & 0x7F) << 4) | ((byte2 >> 4) & 0x0F);

                // 解析4位长度：byte2低4位 + 3（还原实际长度）
                uint8_t match_length = (byte2 & 0x0F) + _min_match_length;

                // 容错处理：无效偏移/长度时跳过（避免越界）
                if (match_offset == 0 || match_offset > concatenated.size() ||
                    match_length < _min_match_length || match_length > _max_match_length)
                {
                    continue;
                }

                // 复制匹配内容（支持自引用，如循环字符串）
                auto start_pos = concatenated.size() - match_offset;
                for (uint8_t i = 0; i < match_length; ++i)
                {
                    concatenated += concatenated[start_pos + i];
                }
            }
        }
#if DEBUG
        fmt::print("解码后拼接的字符串: {}\n", concatenated);
#endif

        // 分割回原始字符串数组（按@分割）
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
        output.push_back(current); // 添加最后一个字符串

        return output;
    }

private:
    uint32_t _window_size = 0;
    uint32_t _lookahead_buffer_size = 0;
    const uint8_t _min_match_length = 3;   // 最小匹配长度（低于此值用单字符编码）
    const uint32_t _max_offset = 2047;     // 11位最大偏移（2^11 - 1）
    const uint32_t _max_match_length = 18; // 最大匹配长度（4位编码+3）
};
#include <algorithm>

class CRCCoder_t
{
public:
    // 构造函数：传入CRC多项式和块大小（默认2048字节）
    CRCCoder_t(uint8_t polynomial = 0x07, size_t block_size = 2048)
    {
        _m_block_size = block_size;
        _m_polynomial = polynomial;
        // 确保块大小至少为1
        if (_m_block_size == 0)
        {
            _m_block_size = 1;
        }
    }

    // 编码：输入原始数据，返回带CRC校验的数据
    std::vector<uint8_t> Encode(const std::vector<uint8_t> &input) const
    {
        std::vector<uint8_t> output;
        if (input.empty())
            return output;

        // 预分配空间：原始数据 + 校验块数（每块1字节）
        size_t total_blocks = (input.size() + _m_block_size - 1) / _m_block_size;
        output.reserve(input.size() + total_blocks);

        size_t pos = 0;
        const uint8_t *data = input.data();

        while (pos < input.size())
        {
            size_t current_len = std::min(_m_block_size, input.size() - pos);
            // 复制数据块
            output.insert(output.end(), data + pos, data + pos + current_len);
            // 计算并附加CRC
            output.push_back(CalculateCRC(data + pos, current_len));
            pos += current_len;
        }

        return output;
    }

    // 解码：输入带CRC的数据，验证通过返回原始数据，失败返回空
    std::vector<uint8_t> Decode(const std::vector<uint8_t> &input) const
    {
        std::vector<uint8_t> output;
        if (input.empty())
            return output;

        size_t pos = 0;
        const uint8_t *data = input.data();
        size_t total_len = input.size();

        while (pos < total_len)
        {
            // 计算当前块的最大可能数据长度（剩余数据减1字节CRC）
            size_t remaining = total_len - pos;
            if (remaining < 1)
                return {}; // 数据不完整（缺少CRC）

            size_t current_data_len = std::min(_m_block_size, remaining - 1);
            uint8_t received_crc = data[pos + current_data_len];

            // 验证CRC
            uint8_t calculated_crc = CalculateCRC(data + pos, current_data_len);
            if (calculated_crc != received_crc)
            {
                return {}; // CRC校验失败
            }

            // 提取有效数据
            output.insert(output.end(), data + pos, data + pos + current_data_len);
            pos += current_data_len + 1; // 跳过数据和CRC
        }

        return output;
    }

private:
    uint8_t _m_polynomial; // CRC多项式
    size_t _m_block_size;  // 分块大小

    // 实时计算8位CRC（无表法）
    uint8_t CalculateCRC(const uint8_t *data, size_t len) const
    {
        uint8_t crc = 0x00;
        for (size_t i = 0; i < len; ++i)
        {
            crc ^= data[i];
            for (int bit = 0; bit < 8; ++bit)
            {
                if (crc & 0x80)
                {
                    crc = (crc << 1) ^ _m_polynomial;
                }
                else
                {
                    crc <<= 1;
                }
                crc &= 0xFF; // 保持8位
            }
        }
        return crc;
    }
};
// 因为是字符串,数据肯定小于128 可以直接用最高位做标志位，用最高位表示是单字符还是偏移+长度