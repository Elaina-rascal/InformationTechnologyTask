#ifndef __ENCODE_HPP__
#define __ENCODE_HPP__
class LZEncoder_t
{
public:
    LZEncoder_t(int windows_size, int lookahead_buffer_size)
    {
        _window_size = windows_size;
        _lookahead_buffer_size = lookahead_buffer_size;
    }

private:
    int _window_size = 0;
    int _lookahead_buffer_size = 0;
};
#endif