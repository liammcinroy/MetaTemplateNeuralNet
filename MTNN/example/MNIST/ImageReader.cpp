#include "ImageReader.h"

ImageReader::ImageReader()
{
	file = std::ifstream("", std::ios::in | std::ios::binary);
	current = Matrix2D<float, 29, 29>(defaultval);
}

ImageReader::ImageReader(const ImageReader &obj)
{
	file = std::ifstream(obj.m_path, std::ios::in | std::ios::binary);
	m_path = obj.m_path;
	current = Matrix2D<float, 29, 29>(defaultval);
	char c;
	for (size_t i = 0; i < 16; ++i)
		file >> c;
	next();
}

ImageReader::ImageReader(const std::string &path)
{
	file = std::ifstream(path, std::ios::in | std::ios::binary);
	m_path = path;
	current = Matrix2D<float, 29, 29>(defaultval);
	char c;
	for (size_t i = 0; i < 16; ++i)
		file >> c;
	next();
}

void ImageReader::next()
{
	for (size_t j = 0; j < 29; ++j)
	{
		current.at(0, j) = defaultval;
		current.at(28, j) = defaultval;
	}

	char c;
	for (size_t i = 0; i < 28; ++i)
	{
		current.at(i, 0) = defaultval;
		current.at(i, 28) = defaultval;


		for (size_t j = 1; j < 29; ++j)
		{
			c = file.peek();
			if (c != '\0')
				current.at(i, j) = 1;
			else
				current.at(i, j) = defaultval;
			file.get();
		}
	}
	++index;
}

void ImageReader::catch_up(int i)
{
	int diff = i - index;
	index = i;
	for (int j = 0; j < diff * 28 * 28; ++j)
		file.get();
}
