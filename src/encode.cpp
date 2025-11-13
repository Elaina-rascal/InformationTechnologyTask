#include "rapidcsv.h"
#include <string>
#include <vector>
int main()
{
    auto csv_path = "/pytorch/data/task1.csv";
    // io::CSVReader<0> in("data_44cols.csv");
    rapidcsv::Document doc(csv_path);

    std::vector<std::string> row = doc.GetRow<std::string>(0);
    for (size_t i = 0; i < row.size(); ++i)
    {
        printf("第 %d 行: %s\n", static_cast<int>(i + 1), row[i].c_str());
    }
    return 0;
}