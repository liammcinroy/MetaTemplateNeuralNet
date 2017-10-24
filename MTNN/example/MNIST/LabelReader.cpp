#include "LabelReader.h"

LabelReader::LabelReader()
{
	file = std::ifstream("", std::ios::in | std::ios::binary);
	current = Matrix2D<float, 10, 1>(defaultval);
	char c;
}

LabelReader::LabelReader(const LabelReader &obj)
{
	file = std::ifstream(obj.m_path, std::ios::in | std::ios::binary);
	m_path = obj.m_path;
	current = Matrix2D<float, 10, 1>(defaultval);
	char c;
	for (size_t i = 0; i < 8; ++i)
		file >> c;
	next();
}

LabelReader::LabelReader(const std::string &path)
{
	file = std::ifstream(path, std::ios::in | std::ios::binary);
	m_path = path;
	current = Matrix2D<float, 10, 1>(defaultval);
	char c;
	for (size_t i = 0; i < 8; ++i)
		file >> c;
	next();
}

void LabelReader::next()
{
	char label;
	label = file.peek();
	file.get();
	for (size_t i = 0; i < 10; ++i)
	{
		if (i == label)
			current.at(i, 0) = 1;
		else
			current.at(i, 0) = defaultval;
	}
	++index;
}

void LabelReader::catch_up(int i)
{
	int diff = i - index;
	index = i;
	for (int j = 0; j < diff; ++j)
		file.get();
}
