/// A patch to resolve the correct filesystem include between <filesystem> and <experimental/filesystem>
/// Both namespaces are converted to the new namespace `fs`
/// See: https://en.cppreference.com/w/cpp/preprocessor/include
/// See: https://stackoverflow.com/a/53366603/5760608
#ifndef FILESYSTEM_PATCH
#define FILESYSTEM_PATCH

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "ERROR: no filesystem support"
#endif

#endif // FILESYSTEM_PATCH