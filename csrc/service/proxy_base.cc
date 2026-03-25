#include "service/proxy_base.h"

#include "common/generator.h"
#include "muduo_base/my_uuid.h"

using namespace std;
using namespace uevent;

ProxyBase::ProxyBase()
    : ctx_id_(NumGenerator::ctx_id()), retcode_(kSuccess), in_conn_(nullptr) {}

ProxyBase::~ProxyBase() {}
