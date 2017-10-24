#include <fstream>
#include <string>

#include "imatrix.h"

class LabelReader
{
public:
	LabelReader();
	LabelReader(const LabelReader &obj);
	LabelReader(const std::string &path);
	~LabelReader() = default;
	void next();
	void catch_up(int i);
	Matrix2D<float, 10, 1> current;
	int index;
	float defaultval = -1;
private:
	std::ifstream file;
	std::string m_path;
};
