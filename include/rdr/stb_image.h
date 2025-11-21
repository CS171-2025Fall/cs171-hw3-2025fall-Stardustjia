#ifndef RDR_STB_IMAGE_H
#define RDR_STB_IMAGE_H

// 禁用一些在MSVC下可能出现的无关警告
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

// 定义这个宏会把实现代码也包含进来
// 你需要从网上下载 stb_image.h 文件，并把它放在 include/rdr/stb/ 目录下
#include "stb/stb_image.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif  // RDR_STB_IMAGE_H