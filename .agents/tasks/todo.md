# JSON Extractor nested JSON 容错

- [x] 复现并确认当前非法尾逗号会抛 ValueError
- [x] 先写测试覆盖：合法嵌套 JSON、非法尾逗号容错、路径无效
- [x] 实现容错解析（仅作用于嵌套 JSON 字符串）
- [x] 运行测试并记录结果
- [x] 在本文件补充 review 结论

## Review
- 新增 `tests/test_json_extractor.py`，覆盖嵌套 JSON 正常提取、尾逗号容错、无效路径。
- 在 `nodes/text/json.py` 增加嵌套 JSON 的容错解析：严格解析失败后，尝试去掉对象/数组闭合前的尾逗号再解析。
- 回归：`uv run -m unittest discover -s tests -p 'test_json_extractor.py'` 通过；`uv run -m unittest discover -s tests -p 'test_*.py'` 全通过（9 tests）。

# xj_metadata 损坏 JSON 兼容与清理

- [x] 复核 `xj_metadata` 的读取与写入路径，确定最小改动点
- [x] 先写测试覆盖：损坏 JSON 恢复提取、写入前规范化、无法修复时不再写入坏 metadata
- [x] 抽出共享 JSON 兼容解析工具，供读取端和保存端复用
- [x] 更新 `JSON Extractor`：优先严格解析，失败时走兼容修复
- [x] 更新保存节点：写入前规范化 metadata，无法修复则跳过 `xj_metadata`
- [x] 用 `uv run` 跑定向与全量测试，并记录结果

## Review
- 新增 `json_compat.py`，统一处理严格解析、尾逗号修复，以及对顶层对象/数组的兼容解析；重点兼容字符串中未转义双引号这类历史坏数据。
- `nodes/text/json.py` 现在优先走共享兼容解析；当嵌套 JSON 可修复时继续提取字段，无法修复时降级返回空 tuple，不再中断执行链。
- `nodes/image/savers.py` 在写入 `xj_metadata` 前会先规范化 JSON；可修复则写入清洗后的合法 JSON，不可修复则跳过该 metadata 并打印一条提示，避免继续产出坏数据。
- 新增 `tests/test_save_image_metadata.py`，并扩充 `tests/test_json_extractor.py`，覆盖未转义双引号修复、不可修复数据降级、保存前规范化与跳过坏 metadata。
- 验证：`uv run -m unittest discover -s tests -p 'test_json_extractor.py'` 通过；`uv run -m unittest discover -s tests -p 'test_save_image_metadata.py'` 通过；`uv run -m unittest discover -s tests -p 'test_*.py'` 全通过（15 tests）。

# Text From List 随机选择行为核查

- [x] 读取 `nodes/text/loaders.py` 中 `XJRandomTextFromList` 和 `XJRandomTextFromFile` 的随机/最近历史逻辑
- [x] 用隔离加载方式复现固定 seed 下的输出序列
- [x] 判断当前实现是否真的“随机且不重复”
- [x] 记录结论与风险点

## Review
- 当前实现“有随机”，但不是“最近项绝不重复”的强保证。
- 原因是每次调用都会 `random.Random(seed)` 重建 RNG，seed 不变时候选序列也不变；最近历史只是在这个固定序列上做过滤，所以输出可能形成短周期重复。
- 最近历史窗口也比较小，`memory_size = max(1, len(list)//2 - 1)`，4 项列表时只会排除最近 1 项。
- 历史是类级别共享的，同内容列表会共享最近选择记录，跨节点/跨工作流也会互相影响。
- 复现结果：4 项列表、seed=0 时输出为 `d, a, d, a...`，说明当前实现能避免“紧邻重复”，但不能避免“看起来反复在少数项之间摆动”。

# Text From List 去除最近项窗口

- [x] 删除 `Random Text From List` 和 `Random Text From File` 的 recent-memory 逻辑
- [x] 保留 fixed / list / random 的原有输入输出形状
- [ ] 运行一个最小验证，确认随机分支不再依赖历史窗口

## Review
- `nodes/text/loaders.py` 现在不再维护 `_selection_memories`，随机选择直接从当前候选池中抽样。
- 这会把行为简化为“输入 seed + 当前列表内容”决定结果，不再有跨节点、跨调用的共享最近历史。

# Image browser symlink-preserving save

- [x] Confirm scope of the image browser save path and symlink behavior
- [x] Update metadata save logic so symlink paths write to the resolved target content without replacing the symlink
- [x] Add or update tests covering normal files and symlinked files
- [x] Run targeted verification and record results

## Review
- `_write_rating_metadata()` 现在会先判断 `image_path` 是否为符号链接；若是，则把临时文件写到真实目标同目录，并用 `os.replace()` 替换真实目标文件，保留链接本身。
- 新增 `tests/test_save_rating_metadata.py`，覆盖普通 PNG 元数据写入，以及通过 symlink 保存时“链接仍存在、目标内容已更新”的回归场景。
- 验证：`uv run -m unittest discover -s tests -p 'test_save_rating_metadata.py'` 通过；`uv run -m unittest discover -s tests -p 'test_*.py'` 全通过（11 tests）。
