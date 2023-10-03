/// A patch to resolve the correct any include between <any> and <experimental/any>
/// Both namespaces are converted to the new namespace `any`
/// See: https://en.cppreference.com/w/cpp/preprocessor/include
/// See: https://stackoverflow.com/a/53366603/5760608
#ifndef ANY_PATCH
#define ANY_PATCH

#if __has_include(<any>)
#include <any>
namespace any = std::any;
#elif __has_include(<experimental/any>)
#include <experimental/any>
namespace any = std::experimental::any;
#else
#error "ERROR: no any support"
#endif

#endif // ANY_PATCH