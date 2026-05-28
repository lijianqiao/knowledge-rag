---
title: 数据库运维 FAQ
tags: [运维, FAQ, 数据库]
version: 1.0
updated: 2026-05-28
---

# 数据库运维 FAQ

## Q1：MySQL 慢查询突然增多？

**A：** 排查顺序：

1. `SHOW PROCESSLIST` 看是否有锁等待
2. 查 slow log 或 `performance_schema.events_statements_summary_by_digest`
3. 对比是否近期有 DDL、统计信息过期、流量上涨

```sql
ANALYZE TABLE orders;
EXPLAIN SELECT ...;
```

## Q2：主从延迟高如何应急？

**A：**

- Kill 非核心长查询（需审批）
- 临时把读流量切到其他从库
- 检查磁盘 IO 与 `sync_binlog` 配置
- 详见 [数据库故障应急预案](../应急预案/数据库故障应急预案.md)

## Q3：Redis 命中率下降？

**A：** 可能原因：内存淘汰（evicted_keys↑）、大 key、热点 key 过期策略变更、业务 key 模式变化。执行 `INFO stats` 与 `--bigkeys` 采样。

## Q4：能否在生产执行 DDL？

**A：** 需走变更流程。大表优先 pt-online-schema-change 或 gh-ost；禁止 peak 时段无评审 DDL。

## Q5：备份失败告警怎么处理？

**A：** 查看备份任务日志，常见：磁盘满、权限、网络至 OSS 超时。重跑备份并确认 binlog 连续性未断。
