---
title: 监控告警 FAQ
tags: [运维, FAQ, 监控, 告警]
version: 1.0
updated: 2026-05-28
---

# 监控告警 FAQ

## Q1：收到告警但 Grafana 看起来正常？

**A：** 可能原因：

- 告警评估窗口与 Dashboard 默认时间范围不一致
- 多副本中仅单实例异常（看 per-pod 面板）
- 告警已 auto-resolve 但通知延迟

建议：点击告警中的 `generatorURL` 跳转对应面板。

## Q2：如何静默维护窗口告警？

**A：** 在 Alertmanager 创建 Silence：

- matcher: `alertname=xxx` 或 `service=xxx`
- duration: 维护窗口长度
- comment: 关联变更工单号

## Q3：Prometheus 数据缺失？

**A：** 检查：target 是否 down、scrape 超时、Pod 是否重启导致 target 漂移、retention 是否已满。

```bash
kubectl logs -n monitoring prometheus-0 -c prometheus | tail -100
```

## Q4：P1 和 P2 告警区别？

**A：**

| 级别 | 定义 | 响应 |
|------|------|------|
| P1 | 核心功能受损 | 15min 响应，电话 |
| P2 | 性能降级或有风险 | 30min 响应，IM |
| P3 | 需关注 | 工作时间处理 |

## Q5：误告警太多怎么办？

**A：** 按 SRE 实践：调整 threshold、增加 `for` 持续时间、用 recording rule 降噪、定期 review 告警有效性（每月）。
