"""错误类型：每个错误码对应契约 §5.3 的一张表，并带 CLI 退出码。"""


class OddsError(Exception):
    """所有可预期错误的基类。"""
    code = "internal"
    exit_code = 1


class BadRequest(OddsError):
    """请求 JSON / 参数不符 schema。"""
    code = "bad_request"
    exit_code = 2


class NoOptions(OddsError):
    """choice / score 题没给选项。"""
    code = "no_options"
    exit_code = 2


class UnknownProfile(OddsError):
    """注册表里没有该模型档位。"""
    code = "unknown_profile"
    exit_code = 2


class ModelUnavailable(OddsError):
    """连不上本地服务 / 服务端报错。"""
    code = "model_unavailable"
    exit_code = 3


class LogprobsUnavailable(OddsError):
    """服务不支持 logprobs / n_probs（如 LM Studio 自带 HTTP 服务）。"""
    code = "logprobs_unavailable"
    exit_code = 3


class OddsTimeout(OddsError):
    """超时（含重试后）。"""
    code = "timeout"
    exit_code = 4


#: 错误码 → CLI 退出码（也供宿主脚本判断）
ERROR_EXIT = {
    OddsError.code: OddsError.exit_code,
    BadRequest.code: BadRequest.exit_code,
    NoOptions.code: NoOptions.exit_code,
    UnknownProfile.code: UnknownProfile.exit_code,
    ModelUnavailable.code: ModelUnavailable.exit_code,
    LogprobsUnavailable.code: LogprobsUnavailable.exit_code,
    OddsTimeout.code: OddsTimeout.exit_code,
}
