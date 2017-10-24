#include <fstream>
#include <string>

#include "imatrix.h"

class ImageReader
{
public:
	ImageReader();
	ImageReader(const ImageReader &obj);
	ImageReader(const std::string &path);
	~ImageReader() = default;
	void next();
	void catch_up(int i);
	Matrix2D<float, 29, 29> current;
	int index = 0;
	float defaultval = -1;
private:
	std::ifstream file;
	std::string m_path;
};
